package whisper_lib

import (
	"context"
	"log/slog"
	"sync"
	"sync/atomic"
	"time"
	"unicode/utf8"

	"github.com/ciricc/go-whisper-server/internal/model/segment"
	"github.com/ggerganov/whisper.cpp/bindings/go/pkg/whisper"
	"golang.org/x/sync/errgroup"
)

// sanitizeUTF8 removes invalid UTF-8 sequences from text to prevent gRPC marshaling errors
func sanitizeUTF8(text string) string {
	if utf8.ValidString(text) {
		return text
	}

	// Replace invalid UTF-8 sequences with replacement character
	return string([]rune(text))
}

// PCMTranscribeTask is a task that transcribes PCM data.
// It is used to transcribe PCM data from a file or a stream of PCM data.
// So we give a flexible interface to transcribe data from different sources.
// If you want to transcribe a wav/flac/mp3 file, you can use the wrappers around this task.
type PCMTranscribeTask struct {
	whisperCtx     *whisper.StatefulContext
	segCh          chan *segment.Segment
	closed         sync.Once
	fullyRead      atomic.Bool
	done           chan error
	logger         *slog.Logger
	globalOffset   time.Duration
	globalDuration time.Duration
	totalSamples   int // Track total samples processed for virtual offset calculation
}

func NewPCMTranscribeTask(
	whisperCtx *whisper.StatefulContext,
	logger *slog.Logger,
	globalOffset time.Duration,
	globalDuration time.Duration,
) *PCMTranscribeTask {
	return &PCMTranscribeTask{
		whisperCtx:     whisperCtx,
		segCh:          make(chan *segment.Segment),
		fullyRead:      atomic.Bool{},
		done:           make(chan error, 1),
		logger:         logger,
		globalOffset:   globalOffset,
		globalDuration: globalDuration,
	}
}

func (t *PCMTranscribeTask) Close() error {
	var err error

	t.closed.Do(func() {
		err = t.close()
	})

	return err
}

func (t *PCMTranscribeTask) close() error {
	close(t.segCh)
	return t.whisperCtx.Close()
}

// Segments returns a receive-only channel of decoded segments.
func (t *PCMTranscribeTask) Segments() <-chan *segment.Segment {
	return t.segCh
}

// Done returns a channel that yields the terminal error when the task finishes.
func (t *PCMTranscribeTask) Done() <-chan error {
	return t.done
}

// Wait blocks until the task completes and returns the terminal error.
func (t *PCMTranscribeTask) Wait() error {
	return <-t.done
}

// Start runs the task in a background goroutine and signals completion on Done().
func (t *PCMTranscribeTask) Start(ctx context.Context, pcmCh <-chan []float32) {
	go func() {
		err := t.runPCM(ctx, pcmCh)
		t.done <- err
		close(t.done)
	}()
}

func (t *PCMTranscribeTask) runPCM(
	ctx context.Context,
	pcmCh <-chan []float32,
) error {
	errGroup, ctx := errgroup.WithContext(ctx)
	defer t.Close()

	errGroup.Go(func() error { return t.processAudioPCM(ctx, pcmCh) })

	return errGroup.Wait()
}

func (t *PCMTranscribeTask) processAudioPCM(
	ctx context.Context,
	pcmCh <-chan []float32,
) error {
	// Create parameterized logger for this method
	log := t.logger.With("method", "processAudioPCM")

	chunkI := 0
	for {
		select {
		case <-ctx.Done():
			log.DebugContext(ctx, "context done")
			return ctx.Err()
		case pcm, ok := <-pcmCh:
			if !ok {
				log.DebugContext(ctx, "pcm channel closed")
				t.fullyRead.Store(true)
				return nil
			}

			chunkI++
			currentChunkSamples := len(pcm)

			// Calculate virtual offset for this chunk based on total processed samples
			virtualChunkOffset := time.Duration(t.totalSamples) * time.Second / 16000

			log.DebugContext(ctx, "chunk",
				"chunkI", chunkI,
				"pcmLen", currentChunkSamples,
				"totalSamples", t.totalSamples,
				"virtualChunkOffset", virtualChunkOffset,
			)

			// Process chunk with zero offset in whisper (we handle offset virtually)
			if pErr := t.whisperCtx.Process(pcm, nil, func(s whisper.Segment) {
				// Apply virtual offset: global offset + chunk offset within file
				segStart := s.Start + t.globalOffset + virtualChunkOffset
				segEnd := s.End + t.globalOffset + virtualChunkOffset

				log.DebugContext(ctx, "processing segment",
					"chunkI", chunkI,
					"globalOffset", t.globalOffset,
					"globalDuration", t.globalDuration,
					"virtualChunkOffset", virtualChunkOffset,
					"s.Start", s.Start,
					"s.End", s.End,
					"segStart", segStart,
					"segEnd", segEnd,
					"text", s.Text)

				// Check if segment exceeds global duration limit
				if t.globalDuration > 0 && segStart >= t.globalDuration {
					log.DebugContext(ctx, "skipping segment beyond global duration",
						"segStart", segStart,
						"globalDuration", t.globalDuration)
					return
				}

				// Trim segment end if it exceeds global duration
				if t.globalDuration > 0 && segEnd > t.globalDuration {
					segEnd = t.globalDuration
					log.DebugContext(ctx, "trimmed segment end to global duration",
						"originalEnd", s.End+t.globalOffset+virtualChunkOffset,
						"trimmedEnd", segEnd)
				}

				seg := segment.NewSegment(
					segStart,
					segEnd,
					sanitizeUTF8(s.Text),
					s.SpeakerTurnNext,
				)
				t.segCh <- seg
			}, nil); pErr != nil {
				return pErr
			}

			log.DebugContext(ctx, "processed",
				"chunkI", chunkI,
				"pcmLen", currentChunkSamples,
				"totalSamples", t.totalSamples,
			)

			// Update total samples processed for next chunk's virtual offset
			t.totalSamples += currentChunkSamples
		}
	}
}
