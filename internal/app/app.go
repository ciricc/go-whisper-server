package app

import (
	"log/slog"
	"os"

	"github.com/ciricc/go-whisper-grpc/internal/config"
	"github.com/ciricc/go-whisper-grpc/internal/server"
	wh "github.com/ciricc/go-whisper-grpc/internal/whisper"
)

type Application struct {
	Config config.Config
	Engine *wh.Engine
	Server *server.TranscriberServer
}

func New() (*Application, error) {
	cfg, err := config.Load("config.yaml")
	if err != nil {
		return nil, err
	}

	engine, err := wh.NewEngine(cfg.Model.Path)
	if err != nil {
		return nil, err
	}

	log := slog.New(slog.NewTextHandler(os.Stdout, nil))

	grpcServer := server.NewTranscriberServer(engine, log)
	return &Application{Config: cfg, Engine: engine, Server: grpcServer}, nil
}

func (a *Application) Close() error {
	if a.Engine != nil {
		_ = a.Engine.Close()
	}
	return nil
}
