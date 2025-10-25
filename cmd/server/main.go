package main

import (
	"log"
	"net"

	"github.com/ciricc/go-whisper-server/internal/app"
	transcriberv1 "github.com/ciricc/go-whisper-server/pkg/proto/transcriber/v1"
	"google.golang.org/grpc"
	"google.golang.org/grpc/health/grpc_health_v1"
)

const MB = 1024 * 1024

func main() {
	application, err := app.New()
	if err != nil {
		log.Fatalf("init error: %v", err)
	}
	defer application.Close()

	lis, err := net.Listen("tcp", application.Config.Server.Address)
	if err != nil {
		log.Fatalf("listen: %v", err)
	}

	const maxMsgSize = 1024 * 10 * MB

	grpcServer := grpc.NewServer(
		grpc.MaxRecvMsgSize(maxMsgSize),
		grpc.MaxSendMsgSize(maxMsgSize),
	)

	// Register transcriber service
	transcriberv1.RegisterTranscriberServer(grpcServer, application.Server)

	// Register health check service if enabled
	if application.HealthChecker != nil {
		grpc_health_v1.RegisterHealthServer(grpcServer, application.HealthChecker)
		log.Printf("health check enabled with threshold: %.0f%%", application.Config.Health.LoadThreshold*100)
	}

	log.Printf("listening on %s", application.Config.Server.Address)
	if err := grpcServer.Serve(lis); err != nil {
		log.Fatalf("serve: %v", err)
	}
}
