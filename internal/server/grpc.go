package server

import (
	"bytes"
	"fmt"
	"io"
	"log/slog"
	"os"

	"github.com/ciricc/go-whisper-server/internal/service/transcribe_svc"
	transcriberv1 "github.com/ciricc/go-whisper-server/pkg/proto/transcriber/v1"
	"google.golang.org/grpc"
	durationpb "google.golang.org/protobuf/types/known/durationpb"
)

const bytesPerSecond = 32000

type TranscriberServer struct {
	transcriberv1.UnimplementedTranscriberServer
	Log           *slog.Logger
	transcribeSvc transcribe_svc.TranscribeService
}

func NewTranscriberServer(
	log *slog.Logger,
	transcribeSvc transcribe_svc.TranscribeService,
) *TranscriberServer {
	return &TranscriberServer{
		Log:           log,
		transcribeSvc: transcribeSvc,
	}
}

// chunkPcm16Le retained for potential future use.
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

func (s *TranscriberServer) getFileReader(
	file *transcriberv1.File,
) (io.ReadSeeker, func() error, error) {
	switch file.GetFile().(type) {
	case *transcriberv1.File_Bytes:
		return bytes.NewReader(file.GetBytes()), nil, nil
	case *transcriberv1.File_Url:
		return nil, nil, fmt.Errorf("url not supported")
	case *transcriberv1.File_Path:
		f, err := os.Open(file.GetPath())
		if err != nil {
			return nil, nil, err
		}
		return f, f.Close, nil
	default:
		return nil, nil, fmt.Errorf("unknown file type")
	}
}

func (s *TranscriberServer) TranscribeWav(
	req *transcriberv1.TranscribeWavRequest,
	stream grpc.ServerStreamingServer[transcriberv1.Segment],
) error {
	if req.GetWav_16KFile() == nil {
		return fmt.Errorf("pcm_file is required")
	}

	r, cleanup, err := s.getFileReader(req.GetWav_16KFile())
	if err != nil {
		return fmt.Errorf("failed to get file reader: %w", err)
	}

	if cleanup != nil {
		defer cleanup()
	}

	task, err := s.transcribeSvc.TranscribeWav(
		stream.Context(),
		r,
		mapWhisperParamsToTranscribeOpts(
			req.GetWhisperParams(),
			req.GetTranscribeWavParams(),
		)...,
	)
	if err != nil {
		return err
	}

	for {
		select {
		case <-stream.Context().Done():
			return stream.Context().Err()
		case seg, ok := <-task.Segments():
			if !ok {
				return task.Wait()
			}
			resp := &transcriberv1.Segment{
				Start:           durationpb.New(seg.Start),
				End:             durationpb.New(seg.End),
				Text:            seg.Text,
				SpeakerTurnNext: seg.SpeakerTurnNext,
			}
			if err := stream.Send(resp); err != nil {
				return err
			}
		}
	}
}
