package whisper_lib

import (
	"context"
	"errors"
	"fmt"
	"io"
	"sync"
	"time"

	"github.com/ciricc/go-whisper-server/internal/model/segment"
	"github.com/go-audio/audio"
	"github.com/go-audio/wav"
)

// WavTranscribeTask provides a wrapper around PCMTranscribeTask that accepts
// a WAV (PCM16LE, mono, 16kHz) io.ReadSeeker and streams it in chunks to Whisper.
//
// The WAV is decoded incrementally and sent to the underlying PCM task as float32
// samples in windows of size windowSeconds (default 30 seconds).
type WavTranscribeTask interface {
	Start(ctx context.Context, wavFile io.ReadSeeker)
	Done() <-chan error
	Wait() error
	Segments() <-chan *segment.Segment
	Close() error
}

type wavTranscribeTask struct {
	pcmTask       *PCMTranscribeTask
	windowSamples int
	done          chan error
	closeOnce     sync.Once
}

// NewWavTranscribeTask creates a WAV wrapper around an existing PCMTranscribeTask.
// windowSeconds configures the size of the PCM windows sent to Whisper (default 30).
func NewWavTranscribeTask(
	pcmTask *PCMTranscribeTask,
	windowSize time.Duration,
) WavTranscribeTask {
	return &wavTranscribeTask{
		pcmTask:       pcmTask,
		windowSamples: int(windowSize.Seconds() * 16000),
		done:          make(chan error, 1),
	}
}

func (t *wavTranscribeTask) Start(ctx context.Context, wavFile io.ReadSeeker) {
	runPCMWithProducer(ctx, t.pcmTask, func(ctx context.Context, pcmCh chan<- []float32) error {
		// Ensure we start reading from the beginning
		if _, err := wavFile.Seek(0, io.SeekStart); err != nil {
			return err
		}

		return streamWavPCM16LEMono16kGoAudio(
			ctx,
			wavFile,
			t.windowSamples,
			pcmCh,
		)
	}, t.finish)
}

func (t *wavTranscribeTask) Segments() <-chan *segment.Segment {
	return t.pcmTask.Segments()
}

func (t *wavTranscribeTask) Done() <-chan error {
	return t.done
}

func (t *wavTranscribeTask) Wait() error {
	return <-t.done
}

func (t *wavTranscribeTask) Close() error {
	var closeErr error
	t.closeOnce.Do(func() {
		closeErr = t.pcmTask.Close()
	})
	return closeErr
}

func (t *wavTranscribeTask) finish(err error) {
	select {
	case t.done <- err:
		close(t.done)
	default:
		// already finished
	}
}

func streamWavPCM16LEMono16kGoAudio(
	ctx context.Context,
	r io.ReadSeeker,
	windowSamples int,
	pcmCh chan<- []float32,
) error {
	dec := wav.NewDecoder(r)
	if !dec.IsValidFile() {
		return errors.New("invalid wav file")
	}
	dec.ReadInfo()

	// Validate required format: PCM16LE mono 16kHz
	if dec.WavAudioFormat != 1 {
		return fmt.Errorf("unsupported wav format: audioFormat=%d (need PCM=1)", dec.WavAudioFormat)
	}
	if dec.NumChans != 1 {
		return fmt.Errorf("unsupported channels: %d (need mono=1)", dec.NumChans)
	}
	if dec.SampleRate != 16000 {
		return fmt.Errorf("unsupported sample rate: %d (need 16000)", dec.SampleRate)
	}
	if dec.BitDepth != 16 {
		return fmt.Errorf("unsupported bits per sample: %d (need 16)", dec.BitDepth)
	}

	if windowSamples <= 0 {
		windowSamples = 60 * 16000
	}

	intBuf := &audio.IntBuffer{
		Format: &audio.Format{
			NumChannels: int(dec.NumChans),
			SampleRate:  int(dec.SampleRate),
		},
		Data:           make([]int, windowSamples),
		SourceBitDepth: int(dec.BitDepth),
	}

	fmt.Printf(
		"[streamWavPCM16LEMono16kGoAudio] windowSamples: %d, sampleRate: %d, numChans: %d, bitDepth: %d\n",
		windowSamples, dec.SampleRate, dec.NumChans, dec.BitDepth,
	)

	chunkI := 0

	for {
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
		}

		intBuf.Data = intBuf.Data[:cap(intBuf.Data)]
		n, err := dec.PCMBuffer(intBuf)
		if n > 0 {
			fb := intBuf.AsFloat32Buffer()
			out := make([]float32, n)
			copy(out, fb.Data[:n])
			fmt.Printf("[streamWavPCM16LEMono16kGoAudio] n: %d, chunkI: %d\n", n, chunkI)
			chunkI++
			fmt.Printf("[streamWavPCM16LEMono16kGoAudio] chunkI: %d\n", chunkI)
			select {
			case <-ctx.Done():
				return ctx.Err()
			case pcmCh <- out:
				fmt.Printf("[streamWavPCM16LEMono16kGoAudio] sent chunk: %d\n", chunkI)
			}
		}

		if err == io.EOF {
			fmt.Printf("[streamWavPCM16LEMono16kGoAudio] EOF\n")
			break
		}
		if err != nil {
			return fmt.Errorf("decode wav pcm: %w", err)
		}

		// Some decoders may return n == 0 with err == nil at exact EOF.
		// Treat this as EOF to avoid a tight infinite loop.
		if n == 0 {
			fmt.Printf("[streamWavPCM16LEMono16kGoAudio] n==0, assume EOF\n")
			break
		}
		fmt.Printf("[streamWavPCM16LEMono16kGoAudio] chunkI: %d, read len: %d\n", chunkI, n)
	}

	return nil
}
