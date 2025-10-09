package app

import (
	"fmt"
	"log/slog"
	"os"

	"github.com/ciricc/go-whisper-server/internal/config"
	"github.com/ciricc/go-whisper-server/internal/server"
	"github.com/ciricc/go-whisper-server/internal/service/transcribe_svc"
	"github.com/ggerganov/whisper.cpp/bindings/go/pkg/whisper"
)

type Application struct {
	Config        config.Config
	Server        *server.TranscriberServer
	whisperModel  *whisper.ModelContext
	transcribeSvc transcribe_svc.TranscribeService
}

func New() (*Application, error) {
	cfg, err := config.Load("config.yaml")
	if err != nil {
		return nil, err
	}

	model, err := whisper.NewModelContext(cfg.Model.Path)
	if err != nil {
		return nil, fmt.Errorf("failed to create whisper model: %w", err)
	}

	log := slog.New(slog.NewTextHandler(os.Stdout, &slog.HandlerOptions{
		Level: slog.LevelDebug,
	}))
	svc := transcribe_svc.NewTranscribeService(model, log)
	grpcServer := server.NewTranscriberServer(log, svc)

	return &Application{
		Config:        cfg,
		Server:        grpcServer,
		whisperModel:  model,
		transcribeSvc: svc,
	}, nil
}

func (a *Application) Close() error {
	if a.whisperModel != nil {
		_ = a.whisperModel.Close()
	}
	return nil
}
