package whisper_lib

import (
	"context"
	"errors"
	"fmt"
	"io"
	"log/slog"
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
	logger        *slog.Logger
	offset        time.Duration
	duration      time.Duration
}

// NewWavTranscribeTask creates a WAV wrapper around an existing PCMTranscribeTask.
// windowSeconds configures the size of the PCM windows sent to Whisper (default 30).
// offset specifies the start time in the audio file to begin transcription.
// duration specifies the maximum duration to transcribe from the offset.
func NewWavTranscribeTask(
	pcmTask *PCMTranscribeTask,
	windowSize time.Duration,
	logger *slog.Logger,
	offset time.Duration,
	duration time.Duration,
) WavTranscribeTask {
	return &wavTranscribeTask{
		pcmTask:       pcmTask,
		windowSamples: int(windowSize.Seconds() * 16000),
		done:          make(chan error, 1),
		logger:        logger,
		offset:        offset,
		duration:      duration,
	}
}

func (t *wavTranscribeTask) Start(ctx context.Context, wavFile io.ReadSeeker) {
	runPCMWithProducer(ctx, t.pcmTask, func(ctx context.Context, pcmCh chan<- []float32) error {
		// Ensure we start reading from the beginning
		if _, err := wavFile.Seek(0, io.SeekStart); err != nil {
			return err
		}

		return streamWavPCM16LEMono16kGoAudioWithOffset(
			ctx,
			wavFile,
			t.windowSamples,
			pcmCh,
			t.logger,
			t.offset,
			t.duration,
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
	logger *slog.Logger,
) error {
	// Create parameterized logger for this method
	log := logger.With("method", "streamWavPCM16LEMono16kGoAudio")

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

	log.DebugContext(ctx, "windowSamples",
		"windowSamples", windowSamples,
		"sampleRate", dec.SampleRate,
		"numChans", dec.NumChans,
		"bitDepth", dec.BitDepth,
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
			log.DebugContext(ctx, "read chunk",
				"n", n,
				"chunkI", chunkI,
			)
			chunkI++
			select {
			case <-ctx.Done():
				return ctx.Err()
			case pcmCh <- out:
				log.DebugContext(ctx, "sent chunk",
					"chunkI", chunkI,
				)
			}
		}

		if err == io.EOF {
			log.DebugContext(ctx, "EOF")
			break
		}
		if err != nil {
			return fmt.Errorf("decode wav pcm: %w", err)
		}

		// Some decoders may return n == 0 with err == nil at exact EOF.
		// Treat this as EOF to avoid a tight infinite loop.
		if n == 0 {
			log.DebugContext(ctx, "n==0, assume EOF")
			break
		}
	}

	return nil
}

func streamWavPCM16LEMono16kGoAudioWithOffset(
	ctx context.Context,
	r io.ReadSeeker,
	windowSamples int,
	pcmCh chan<- []float32,
	logger *slog.Logger,
	offset time.Duration,
	duration time.Duration,
) error {
	// Create parameterized logger for this method
	log := logger.With("method", "streamWavPCM16LEMono16kGoAudioWithOffset")

	// First, read WAV header to get format info
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

	// Calculate offset and duration in samples
	offsetSamples := int(offset.Seconds() * float64(dec.SampleRate))
	var durationSamples int
	if duration > 0 {
		durationSamples = int(duration.Seconds() * float64(dec.SampleRate))
	}

	log.DebugContext(ctx, "offset and duration",
		"offset", offset.String(),
		"duration", duration.String(),
		"offsetSamples", offsetSamples,
		"durationSamples", durationSamples,
	)

	// Find data chunk start and seek to offset
	dataStart, err := findWavDataStart(r)
	if err != nil {
		return fmt.Errorf("find wav data start: %w", err)
	}

	// Calculate offset in bytes (16-bit samples, mono)
	offsetBytes := offsetSamples * 2 // 2 bytes per sample
	seekPos := dataStart + int64(offsetBytes)

	if _, err := r.Seek(seekPos, io.SeekStart); err != nil {
		return fmt.Errorf("seek to offset: %w", err)
	}

	log.DebugContext(ctx, "seeked to offset",
		"dataStart", dataStart,
		"offsetBytes", offsetBytes,
		"seekPos", seekPos,
	)

	// Now read raw PCM data directly
	chunkI := 0
	totalSamplesRead := 0
	bytesPerSample := 2 // 16-bit samples

	for {
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
		}

		// Check duration limit
		if duration > 0 && totalSamplesRead >= durationSamples {
			log.DebugContext(ctx, "duration limit reached",
				"totalSamplesRead", totalSamplesRead,
				"durationSamples", durationSamples,
			)
			break
		}

		// Calculate chunk size in bytes
		chunkSizeBytes := windowSamples * bytesPerSample
		if duration > 0 {
			remainingSamples := durationSamples - totalSamplesRead
			if remainingSamples <= 0 {
				break
			}
			if remainingSamples < windowSamples {
				chunkSizeBytes = remainingSamples * bytesPerSample
			}
		}

		// Read raw PCM data
		rawData := make([]byte, chunkSizeBytes)
		n, err := r.Read(rawData)
		if n == 0 {
			if err == io.EOF {
				log.DebugContext(ctx, "EOF")
				break
			}
			if err != nil {
				return fmt.Errorf("read raw PCM: %w", err)
			}
			continue
		}

		// Convert bytes to samples (ensure even number of bytes)
		if n%2 != 0 {
			n-- // Drop odd byte
		}
		samplesRead := n / bytesPerSample

		// Convert int16 samples to float32
		chunk := make([]float32, samplesRead)
		for i := 0; i < samplesRead; i++ {
			// Little-endian int16 to float32
			sample := int16(rawData[i*2]) | int16(rawData[i*2+1])<<8
			chunk[i] = float32(sample) / 32768.0
		}

		totalSamplesRead += samplesRead

		log.DebugContext(ctx, "read chunk",
			"chunkI", chunkI,
			"samplesRead", samplesRead,
			"totalSamplesRead", totalSamplesRead,
		)

		chunkI++

		select {
		case <-ctx.Done():
			return ctx.Err()
		case pcmCh <- chunk:
			log.DebugContext(ctx, "sent chunk",
				"chunkI", chunkI,
			)
		}

		if err == io.EOF {
			log.DebugContext(ctx, "EOF")
			break
		}
		if err != nil {
			return fmt.Errorf("read raw PCM: %w", err)
		}
	}

	log.DebugContext(ctx, "finished reading",
		"chunkI", chunkI,
		"totalSamplesRead", totalSamplesRead,
	)

	return nil
}

// findWavDataStart finds the position where PCM data starts in a WAV file
func findWavDataStart(r io.ReadSeeker) (int64, error) {
	// Save current position
	currentPos, err := r.Seek(0, io.SeekCurrent)
	if err != nil {
		return 0, err
	}
	defer r.Seek(currentPos, io.SeekStart)

	// Go to beginning
	if _, err := r.Seek(0, io.SeekStart); err != nil {
		return 0, err
	}

	// Read RIFF header
	header := make([]byte, 12)
	if _, err := io.ReadFull(r, header); err != nil {
		return 0, err
	}

	// Verify RIFF header
	if string(header[0:4]) != "RIFF" || string(header[8:12]) != "WAVE" {
		return 0, errors.New("invalid WAV file")
	}

	// Read chunks until we find "data"
	for {
		chunkHeader := make([]byte, 8)
		if _, err := io.ReadFull(r, chunkHeader); err != nil {
			return 0, err
		}

		chunkID := string(chunkHeader[0:4])
		chunkSize := int64(chunkHeader[4]) | int64(chunkHeader[5])<<8 | int64(chunkHeader[6])<<16 | int64(chunkHeader[7])<<24

		if chunkID == "data" {
			pos, err := r.Seek(0, io.SeekCurrent)
			if err != nil {
				return 0, err
			}
			return pos, nil
		}

		// Skip chunk data
		if _, err := r.Seek(chunkSize, io.SeekCurrent); err != nil {
			return 0, err
		}
	}
}
