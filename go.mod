module github.com/ciricc/go-whisper-server

go 1.24.0

require (
	github.com/gen2brain/malgo v0.11.23
	github.com/ggerganov/whisper.cpp/bindings/go v0.0.0-20250905132032-bb0e1fc60f26
	github.com/go-audio/audio v1.0.0
	github.com/go-audio/wav v1.1.0
	github.com/samber/lo v1.51.0
	google.golang.org/grpc v1.76.0
	google.golang.org/protobuf v1.36.8
	gopkg.in/yaml.v3 v3.0.1
)

require (
	github.com/go-audio/riff v1.0.0 // indirect
	github.com/stretchr/testify v1.10.0 // indirect
)

replace github.com/ggerganov/whisper.cpp/bindings/go => ./third_party/whisper.cpp/bindings/go

require (
	golang.org/x/net v0.42.0 // indirect
	golang.org/x/sync v0.17.0
	golang.org/x/sys v0.34.0 // indirect
	golang.org/x/text v0.27.0 // indirect
	google.golang.org/genproto/googleapis/rpc v0.0.0-20250804133106-a7a43d27e69b // indirect
)
