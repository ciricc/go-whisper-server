package main

import (
	"context"
	"encoding/binary"
	"errors"
	"flag"
	"fmt"
	"io"
	"log"
	"os"
	"strings"
	"time"

	"github.com/ciricc/go-whisper-grpc/internal/config"
	transcriberv1 "github.com/ciricc/go-whisper-grpc/pkg/proto/transcriber/v1"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

func main() {
	var (
		cfgPath  = flag.String("config", "config.yaml", "path to config.yaml for server address")
		wavPath  = flag.String("wav", "testdata/audio.wav", "path to 16-bit PCM WAV mono file")
		dialAddr = flag.String("addr", "", "override server address (e.g., :50051)")
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

	// info, dataStart, err := parseWAVHeader(f)
	// if err != nil {
	// 	log.Fatalf("parse wav: %v", err)
	// }
	// if info.BitsPerSample != 16 || info.NumChannels != 1 {
	// 	log.Printf("warning: expected mono 16-bit PCM wav, got %d ch, %d bits", info.NumChannels, info.BitsPerSample)
	// }
	// if cfg.Model.SampleRateHz > 0 && info.SampleRate != cfg.Model.SampleRateHz {
	// 	log.Printf("warning: wav sample rate %d != model sample rate %d; results may degrade (no resample)", info.SampleRate, cfg.Model.SampleRateHz)
	// }
	// if _, err := f.Seek(dataStart, io.SeekStart); err != nil {
	// 	log.Fatalf("seek data: %v", err)
	// }
	pcm, err := io.ReadAll(f)
	if err != nil {
		log.Fatalf("read wav data: %v", err)
	}

	conn, err := grpc.Dial(addr, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		log.Fatalf("dial: %v", err)
	}
	defer conn.Close()

	client := transcriberv1.NewTranscriberClient(conn)
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Minute)
	defer cancel()

	resp, err := client.Transcribe(ctx, &transcriberv1.TranscribeRequest{
		Language: "ru",
		WavFile:  pcm,
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
		fmt.Printf("[%6d -> %6d] %s\n", seg.StartMs, seg.EndMs, seg.Text)
	}
}

type wavInfo struct {
	AudioFormat   uint16
	NumChannels   uint16
	SampleRate    int
	BitsPerSample int
}

func parseWAVHeader(r io.ReadSeeker) (wavInfo, int64, error) {
	var info wavInfo
	var hdr [12]byte
	if _, err := io.ReadFull(r, hdr[:]); err != nil {
		return info, 0, fmt.Errorf("read riff: %w", err)
	}
	if string(hdr[0:4]) != "RIFF" || string(hdr[8:12]) != "WAVE" {
		return info, 0, fmt.Errorf("not a WAVE RIFF file")
	}
	for {
		var chunkHdr [8]byte
		if _, err := io.ReadFull(r, chunkHdr[:]); err != nil {
			return info, 0, fmt.Errorf("read chunk: %w", err)
		}
		ckID := string(chunkHdr[0:4])
		ckSize := int64(binary.LittleEndian.Uint32(chunkHdr[4:8]))
		switch ckID {
		case "fmt ":
			data := make([]byte, ckSize)
			if _, err := io.ReadFull(r, data); err != nil {
				return info, 0, fmt.Errorf("read fmt: %w", err)
			}
			if len(data) < 16 {
				return info, 0, fmt.Errorf("fmt too short")
			}
			info.AudioFormat = binary.LittleEndian.Uint16(data[0:2])
			info.NumChannels = binary.LittleEndian.Uint16(data[2:4])
			info.SampleRate = int(binary.LittleEndian.Uint32(data[4:8]))
			info.BitsPerSample = int(binary.LittleEndian.Uint16(data[14:16]))
		case "data":
			pos, _ := r.Seek(0, io.SeekCurrent)
			return info, pos, nil
		default:
			if _, err := r.Seek(ckSize, io.SeekCurrent); err != nil {
				return info, 0, fmt.Errorf("skip %s: %w", ckID, err)
			}
		}
	}
}
