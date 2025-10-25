package app

import (
	"context"
	"fmt"
	"log/slog"
	"os"

	"github.com/ciricc/go-whisper-server/internal/config"
	"github.com/ciricc/go-whisper-server/internal/health"
	"github.com/ciricc/go-whisper-server/internal/monitor"
	"github.com/ciricc/go-whisper-server/internal/server"
	"github.com/ciricc/go-whisper-server/internal/service/transcribe_svc"
	"github.com/ciricc/go-whisper-server/internal/telemetry"
	transcriberv1 "github.com/ciricc/go-whisper-server/pkg/proto/transcriber/v1"
	"github.com/ggerganov/whisper.cpp/bindings/go/pkg/whisper"
	"google.golang.org/grpc/health/grpc_health_v1"
)

type Application struct {
	Config        config.Config
	Server        *server.TranscriberServer
	HealthChecker *health.HealthChecker
	whisperModel  *whisper.ModelContext
	transcribeSvc transcribe_svc.TranscribeService
	loadMonitor   monitor.LoadMonitor
	telemetry     *telemetry.SDK
}

func New() (*Application, error) {
	cfg, err := config.Load("config.yaml")
	if err != nil {
		return nil, err
	}

	// Initialize OpenTelemetry from configuration file (optional)
	ctx := context.Background()
	otelSDK, err := telemetry.InitFromConfig(ctx, "otel-config.yaml")
	if err != nil {
		log := slog.New(slog.NewTextHandler(os.Stdout, &slog.HandlerOptions{
			Level: slog.LevelDebug,
		}))
		log.Warn("Failed to initialize telemetry, continuing without it", "error", err)
		otelSDK = nil
	}

	model, err := whisper.NewModelContext(cfg.Model.Path)
	if err != nil {
		return nil, fmt.Errorf("failed to create whisper model: %w", err)
	}

	log := slog.New(slog.NewTextHandler(os.Stdout, &slog.HandlerOptions{
		Level: slog.LevelDebug,
	}))

	// Create load monitor with health threshold
	loadMonitor := monitor.NewSemaphoreLoadMonitor(
		int64(cfg.Transcribe.MaxConcurrency),
		cfg.Health.LoadThreshold,
	)

	// Create transcribe service with load monitor
	svc := transcribe_svc.NewTranscribeService(
		model,
		log,
		loadMonitor,
	)

	// Create health checker
	var healthChecker *health.HealthChecker
	if cfg.Health.Enabled {
		healthChecker = health.NewHealthChecker(loadMonitor)
		// Register the transcriber service as healthy by default
		healthChecker.SetServingStatus(
			transcriberv1.Transcriber_ServiceDesc.ServiceName,
			grpc_health_v1.HealthCheckResponse_SERVING,
		)
	}

	grpcServer := server.NewTranscriberServer(
		log,
		svc,
	)

	return &Application{
		Config:        cfg,
		Server:        grpcServer,
		HealthChecker: healthChecker,
		whisperModel:  model,
		transcribeSvc: svc,
		loadMonitor:   loadMonitor,
		telemetry:     otelSDK,
	}, nil
}

func (a *Application) Close() error {
	ctx := context.Background()

	// Shutdown telemetry first to flush remaining data
	if a.telemetry != nil {
		if err := a.telemetry.Shutdown(ctx); err != nil {
			return fmt.Errorf("failed to shutdown telemetry: %w", err)
		}
	}

	if a.whisperModel != nil {
		_ = a.whisperModel.Close()
	}
	return nil
}
