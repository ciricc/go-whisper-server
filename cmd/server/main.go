package main

import (
	"log"
	"net"

	"github.com/ciricc/go-whisper-server/internal/app"
	transcriberv1 "github.com/ciricc/go-whisper-server/pkg/proto/transcriber/v1"
	"google.golang.org/grpc"
)

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

	const maxMsgSize = 1024 * 1024 * 1024

	grpcServer := grpc.NewServer(
		grpc.MaxRecvMsgSize(maxMsgSize),
		grpc.MaxSendMsgSize(maxMsgSize),
	)
	transcriberv1.RegisterTranscriberServer(grpcServer, application.Server)
	log.Printf("listening on %s", application.Config.Server.Address)
	if err := grpcServer.Serve(lis); err != nil {
		log.Fatalf("serve: %v", err)
	}
}
