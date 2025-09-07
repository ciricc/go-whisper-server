module github.com/ciricc/go-whisper-grpc

go 1.24.0

require (
	github.com/ggerganov/whisper.cpp/bindings/go v0.0.0-20250905132032-bb0e1fc60f26
	google.golang.org/grpc v1.75.0
	google.golang.org/protobuf v1.36.8
	gopkg.in/yaml.v3 v3.0.1
)

require (
	github.com/djthorpe/go-errors v1.0.3 // indirect
	github.com/mutablelogic/go-media v1.7.7 // indirect
	golang.org/x/exp v0.0.0-20250506013437-ce4c2cf36ca6 // indirect
)

replace github.com/ggerganov/whisper.cpp/bindings/go => ./third_party/whisper.cpp/bindings/go

require (
	golang.org/x/net v0.41.0 // indirect
	golang.org/x/sync v0.17.0
	golang.org/x/sys v0.33.0 // indirect
	golang.org/x/text v0.26.0 // indirect
	google.golang.org/genproto/googleapis/rpc v0.0.0-20250707201910-8d1bb00bc6a7 // indirect
)
