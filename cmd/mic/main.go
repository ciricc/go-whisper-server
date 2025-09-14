package main

import (
	"context"
	"encoding/binary"
	"flag"
	"fmt"
	"log"
	"math"
	"os"
	"os/signal"
	"sync/atomic"
	"time"

	"github.com/ciricc/go-whisper-server/pkg/whisper_lib"
	"github.com/gen2brain/malgo"
	"github.com/ggerganov/whisper.cpp/bindings/go/pkg/whisper"
)

func main() {
	var (
		modelPath = flag.String("model", "models/ggml-large-v3-turbo.bin", "path to whisper ggml model")
	)

	flag.Parse()

	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, os.Interrupt)

	model, err := whisper.NewModelContext(*modelPath)
	if err != nil {
		log.Fatalf("load model: %v", err)
	}
	defer model.Close()

	whisperParams, err := whisper.NewParameters(model, whisper.SAMPLING_GREEDY, func(p *whisper.Parameters) {
		p.SetLanguage("en")
		p.SetNoContext(true)
		p.SetSplitOnWord(true)
		p.SetTemperature(0.0)
		p.SetTemperatureFallback(0.2)
		p.SetVAD(true)
		p.SetVADThreshold(0.5)
	})
	if err != nil {
		log.Fatalf("create whisper parameters: %v", err)
	}

	whisperCtx, err := whisper.NewStatefulContext(model, whisperParams)
	if err != nil {
		log.Fatalf("create whisper context: %v", err)
	}
	defer whisperCtx.Close()

	task := whisper_lib.NewPCMTranscribeTask(whisperCtx)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	go func() {
		<-sigCh
		cancel()
	}()

	// Setup microphone capture
	ctxAudio, err := malgo.InitContext(nil, malgo.ContextConfig{}, func(message string) {})
	if err != nil {
		log.Fatalf("init malgo: %v", err)
	}
	defer ctxAudio.Uninit()

	cfg := malgo.DefaultDeviceConfig(malgo.Capture)
	cfg.Capture.Channels = 1
	cfg.Capture.Format = malgo.FormatS16
	cfg.SampleRate = 16000

	pcmCh := make(chan []float32, 128)
	task.Start(ctx, pcmCh)

	var stopped atomic.Bool
	// accumulate ~1000 ms (16000 samples at 16kHz) for more stable segments
	const targetSamples = 16000 * 2
	accum := make([]float32, 0, 16000)
	actualSampleRate := int(cfg.SampleRate)
	callbacks := malgo.DeviceCallbacks{
		Data: func(pOutput, pInput []byte, frameCount uint32) {
			// fmt.Printf("[malgo] data: %d\n", frameCount)
			if stopped.Load() {
				return
			}
			// Convert PCM16LE to float32
			floats := make([]float32, 0, len(pInput)/2)
			for i := 0; i+1 < len(pInput); i += 2 {
				v := int16(binary.LittleEndian.Uint16(pInput[i : i+2]))
				floats = append(floats, float32(v)/32768.0)
			}
			// Resample if needed
			if actualSampleRate != 16000 && len(floats) > 0 {
				floats = resampleMonoFloat32(floats, actualSampleRate, 16000)
			}
			// accumulate
			accum = append(accum, floats...)
			// While we have enough for a chunk, try to send
			for len(accum) >= targetSamples {
				out := make([]float32, targetSamples)
				copy(out, accum[:targetSamples])
				select {
				case pcmCh <- out:
					accum = accum[targetSamples:]
				default:
					return
				}
			}
		},
	}

	device, err := malgo.InitDevice(ctxAudio.Context, cfg, callbacks)
	if err != nil {
		log.Fatalf("init device: %v", err)
	}
	defer device.Uninit()

	// Query actual device sample rate
	if sr := int(device.SampleRate()); sr > 0 {
		actualSampleRate = sr
	}

	if err := device.Start(); err != nil {
		log.Fatalf("start device: %v", err)
	}

	// Segment printer
	done := make(chan struct{})
	go func() {
		for seg := range task.Segments() {
			fmt.Printf("[%6d -> %6d] %s\n", seg.Start.Milliseconds(), seg.End.Milliseconds(), seg.Text)
		}
		close(done)
	}()

	// Wait for cancel (Ctrl+C)
	<-ctx.Done()
	stopped.Store(true)
	_ = device.Stop()
	// flush remaining accumulated samples (pad to targetSamples)
	if len(accum) > 0 {
		out := make([]float32, targetSamples)
		copy(out, accum)
		select {
		case pcmCh <- out:
		default:
		}
	}
	close(pcmCh)

	if err := task.Wait(); err != nil {
		log.Printf("task error: %v", err)
	}
	<-done

	time.Sleep(200 * time.Millisecond)
}

func resampleMonoFloat32(in []float32, inRate, outRate int) []float32 {
	if inRate == outRate || len(in) == 0 {
		out := make([]float32, len(in))
		copy(out, in)
		return out
	}
	ratio := float64(outRate) / float64(inRate)
	outLen := int(math.Round(float64(len(in)) * ratio))
	if outLen <= 0 {
		return nil
	}
	out := make([]float32, outLen)
	for i := 0; i < outLen; i++ {
		x := float64(i) / ratio
		ix := int(math.Floor(x))
		fx := float32(x - float64(ix))
		if ix >= len(in)-1 {
			out[i] = in[len(in)-1]
			continue
		}
		v0 := in[ix]
		v1 := in[ix+1]
		out[i] = v0 + (v1-v0)*fx
	}
	return out
}
