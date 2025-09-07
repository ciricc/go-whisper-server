package server

import (
	"bytes"
	"fmt"
	"log/slog"
	"runtime"
	"time"

	wh "github.com/ciricc/go-whisper-grpc/internal/whisper"
	transcriberv1 "github.com/ciricc/go-whisper-grpc/pkg/proto/transcriber/v1"
	whisperpkg "github.com/ggerganov/whisper.cpp/bindings/go/pkg/whisper"
	"github.com/mutablelogic/go-media/pkg/segmenter"
	"google.golang.org/grpc"
)

var punctuationPrompts = map[string]string{
	"ru": "Это расшифровка русской речи. Используй правильную пунктуацию: запятые, точки, вопросительные и восклицательные знаки. Начинай предложения с заглавной буквы.",
	"en": "This is a transcription of English speech. Use proper punctuation: commas, periods, question marks, and exclamation marks. Start sentences with a capital letter.",
}

const bytesPerSecond = 32000

type TranscriberServer struct {
	transcriberv1.UnimplementedTranscriberServer
	Engine *wh.Engine
	Log    *slog.Logger
}

func NewTranscriberServer(
	engine *wh.Engine,
	log *slog.Logger,
) *TranscriberServer {
	return &TranscriberServer{
		Engine: engine,
		Log:    log,
	}
}

func chunkPcm16Le(pcm16Le []byte, chunkSize int) [][]byte {
	if chunkSize <= 0 {
		return [][]byte{pcm16Le}
	}
	// Ensure 2-byte alignment for 16-bit samples
	if chunkSize%2 != 0 {
		chunkSize++
	}

	est := len(pcm16Le) / chunkSize
	if est == 0 {
		est = 1
	}
	chunks := make([][]byte, 0, est)

	for i := 0; i < len(pcm16Le); i += chunkSize {
		end := i + chunkSize
		if end > len(pcm16Le) {
			end = len(pcm16Le)
		}
		// keep even number of bytes per chunk
		if (end-i)%2 != 0 {
			end--
		}
		if end > i {
			chunks = append(chunks, pcm16Le[i:end])
		}
	}

	return chunks
}

func (s *TranscriberServer) Transcribe(
	req *transcriberv1.TranscribeRequest,
	stream grpc.ServerStreamingServer[transcriberv1.Segment],
) error {
	log := s.Log.With(
		slog.String("method", "Transcribe"),
		slog.String("language", req.Language),
	)

	wCtx, err := s.Engine.SpawnTranscribe(
		stream.Context(),
		wh.WithLanguage(req.Language),
	)
	if err != nil {
		return err
	}

	log.Info("Processing audio", "size", len(req.WavFile))

	wCtx.SetThreads(uint(runtime.NumCPU()))
	wCtx.SetTemperature(0.0)

	// Encourage punctuation for RU and allow longer sentences before hard split
	if _, ok := punctuationPrompts[req.Language]; ok {
		wCtx.SetInitialPrompt(punctuationPrompts[req.Language])
	}

	segmenter, err := segmenter.NewReader(
		bytes.NewReader(req.WavFile),
		16000,
	)
	if err != nil {
		return fmt.Errorf("failed to create segmenter: %w", err)
	}

	err = segmenter.DecodeFloat32(stream.Context(), func(t time.Duration, d []float32) error {
		cb := func(segment whisperpkg.Segment) {
			startMs := segment.Start.Milliseconds()
			endMs := segment.End.Milliseconds()
			log.Info(
				"Sending segment",
				"start", startMs,
				"end", endMs,
				"text", segment.Text,
			)
			resp := &transcriberv1.Segment{
				StartMs: startMs,
				EndMs:   endMs,
				Text:    segment.Text,
			}
			if err := stream.Send(resp); err != nil {
				log.Error("Error sending segment", "error", err)
			}
		}

		return wCtx.Process(d, func() bool {
			select {
			case <-stream.Context().Done():
				log.Error("Stream context done", "error", stream.Context().Err())
				return false
			default:
				return true
			}
		}, cb, func(i int) {
			log.Info("Progress", "progress", i)
		})
	})
	if err != nil {
		return fmt.Errorf("failed to decode segmenter: %w", err)
	}

	return nil
}
