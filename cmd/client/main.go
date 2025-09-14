package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"io"
	"log"
	"os"
	"strings"
	"time"

	"github.com/ciricc/go-whisper-server/internal/config"
	transcriberv1 "github.com/ciricc/go-whisper-server/pkg/proto/transcriber/v1"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/protobuf/types/known/wrapperspb"
)

func main() {
	var (
		cfgPath  = flag.String("config", "config.yaml", "path to config.yaml for server address")
		wavPath  = flag.String("wav", "testdata/output.wav", "path to 16-bit PCM WAV mono file")
		dialAddr = flag.String("addr", "localhost:50051", "override server address (e.g., :50051)")
	)

	flag.Parse()

	cfg, err := config.Load(*cfgPath)
	if err != nil {
		log.Fatalf("load config: %v", err)
	}
	addr := cfg.Server.Address
	if strings.TrimSpace(*dialAddr) != "" {
		addr = *dialAddr
	}

	f, err := os.Open(*wavPath)
	if err != nil {
		log.Fatalf("open wav: %v", err)
	}
	defer f.Close()

	wavData, err := io.ReadAll(f)
	if err != nil {
		log.Fatalf("read wav data: %v", err)
	}

	client, err := grpc.NewClient(addr, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		log.Fatalf("new client: %v", err)
	}
	defer client.Close()

	transcriber := transcriberv1.NewTranscriberClient(client)
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Minute)
	defer cancel()

	resp, err := transcriber.TranscribeWav(ctx, &transcriberv1.TranscribeWavRequest{
		Language: "en",
		Wav_16KFile: &transcriberv1.File{
			File: &transcriberv1.File_Bytes{
				Bytes: wavData,
			},
		},
		WhisperParams: &transcriberv1.WhisperParams{
			SplitOnWord:         wrapperspb.Bool(true),
			BeamSize:            wrapperspb.Int32(20),
			Temperature:         wrapperspb.Float(0.0),
			TemperatureFallback: wrapperspb.Float(0.2),
			MaxTokensPerSegment: wrapperspb.Int32(128),
			TokenThreshold:      wrapperspb.Float(0.2),
			Translate:           wrapperspb.Bool(false),
		},
	})
	if err != nil {
		log.Fatalf("Transcribe: %v", err)
	}

	for {
		seg, rerr := resp.Recv()
		if errors.Is(rerr, io.EOF) {
			break
		}

		if rerr != nil {
			log.Fatalf("recv: %v", rerr)
		}

		fmt.Printf(
			"[%6d -> %6d] %s\n",
			seg.GetStart().AsDuration().Milliseconds(),
			seg.GetEnd().AsDuration().Milliseconds(),
			seg.GetText(),
		)
	}
}
