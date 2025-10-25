package telemetry

import (
	"context"
	"fmt"
	"os"

	otelconf "go.opentelemetry.io/contrib/otelconf/v0.3.0"
	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/log/global"
)

// SDK wraps the OpenTelemetry SDK initialized from configuration file
type SDK struct {
	sdk otelconf.SDK
}

// InitFromConfig initializes OpenTelemetry SDK from a YAML configuration file
// using the opentelemetry-configuration schema.
// Returns nil SDK and nil error if telemetry is disabled or config file doesn't exist.
func InitFromConfig(ctx context.Context, configPath string) (*SDK, error) {
	// Check if telemetry is disabled via environment variable
	if os.Getenv("OTEL_SDK_DISABLED") == "true" {
		return nil, nil
	}

	// Read configuration file
	configBytes, err := os.ReadFile(configPath)
	if err != nil {
		// If config file doesn't exist, return nil (telemetry disabled)
		if os.IsNotExist(err) {
			return nil, nil
		}
		return nil, fmt.Errorf("failed to read otel config file: %w", err)
	}

	// Parse YAML configuration
	config, err := otelconf.ParseYAML(configBytes)
	if err != nil {
		return nil, fmt.Errorf("failed to parse otel config: %w", err)
	}

	// Check if disabled in config
	if config.Disabled != nil && *config.Disabled {
		return nil, nil
	}

	// Create SDK from configuration
	sdk, err := otelconf.NewSDK(
		otelconf.WithOpenTelemetryConfiguration(*config),
	)
	if err != nil {
		return nil, fmt.Errorf("failed to create otel SDK: %w", err)
	}

	// Set global providers
	otel.SetTracerProvider(sdk.TracerProvider())
	otel.SetMeterProvider(sdk.MeterProvider())
	global.SetLoggerProvider(sdk.LoggerProvider())

	fmt.Printf("OpenTelemetry initialized successfully\n")
	fmt.Printf("  TracerProvider: %T\n", sdk.TracerProvider())
	fmt.Printf("  MeterProvider: %T\n", sdk.MeterProvider())

	return &SDK{sdk: sdk}, nil
}

// Shutdown gracefully shuts down the OpenTelemetry SDK
func (s *SDK) Shutdown(ctx context.Context) error {
	if s == nil {
		return nil
	}
	return s.sdk.Shutdown(ctx)
}
