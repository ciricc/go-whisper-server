package whisper_lib

import (
	"context"
	"fmt"
	"sync"
	"sync/atomic"
	"time"

	"github.com/ciricc/go-whisper-server/internal/model/segment"
	"github.com/ggerganov/whisper.cpp/bindings/go/pkg/whisper"
	"golang.org/x/sync/errgroup"
)

// PCMTranscribeTask is a task that transcribes PCM data.
// It is used to transcribe PCM data from a file or a stream of PCM data.
// So we give a flexible interface to transcribe data from different sources.
// If you want to transcribe a wav/flac/mp3 file, you can use the wrappers around this task.
type PCMTranscribeTask struct {
	whisperCtx *whisper.StatefulContext
	segCh      chan *segment.Segment
	closed     sync.Once
	fullyRead  atomic.Bool
	done       chan error
}

func NewPCMTranscribeTask(
	whisperCtx *whisper.StatefulContext,
) *PCMTranscribeTask {
	return &PCMTranscribeTask{
		whisperCtx: whisperCtx,
		segCh:      make(chan *segment.Segment),
		fullyRead:  atomic.Bool{},
		done:       make(chan error, 1),
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
	chunkI := 0
	var baseOffset time.Duration
	for {
		select {
		case <-ctx.Done():
			fmt.Println("[whisper] processAudioPCM: context done")
			return ctx.Err()
		case pcm, ok := <-pcmCh:
			if !ok {
				fmt.Println("[whisper] processAudioPCM: pcm channel closed")
				t.fullyRead.Store(true)
				return nil
			}
			chunkI++
			fmt.Printf("[whisper] processAudioPCM: chunk: %d\n", chunkI)
			// Emit segments via callback, applying absolute baseOffset
			if pErr := t.whisperCtx.Process(pcm, nil, func(s whisper.Segment) {
				seg := segment.NewSegment(
					s.Start+baseOffset,
					s.End+baseOffset,
					s.Text,
					s.SpeakerTurnNext,
				)
				t.segCh <- seg
			}, nil); pErr != nil {
				return pErr
			}
			fmt.Printf("[whisper] processAudioPCM: processed: %d\n", chunkI)
			// Advance absolute offset by this chunk length
			baseOffset += time.Duration(len(pcm)) * time.Second / 16000
		}
	}
}
