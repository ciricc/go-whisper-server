## go-whisper-server

gRPC server that streams speech to text using whisper.cpp through the official Go bindings. It accepts 16 kHz mono 16-bit PCM WAV files and returns decoded segments as a server stream.

### Features
- CPU, Metal (macOS), and CUDA (Linux) builds
- Server streaming of segments while decoding
- Full control over whisper parameters via the API
- Example client and a microphone demo

### Requirements
- Go 1.22 or newer
- CMake 3.21 or newer and a C or C++ compiler
- macOS
  - Xcode command line tools
  - Metal build requires a Metal capable Mac (Taskfile provides flags)
- Linux (for CUDA build)
  - CUDA toolkit (nvcc) if you want CUDA acceleration
  - See Taskfile for optional FFmpeg dev packages used by some whisper.cpp builds

### Getting started
Clone the repo and enter the project directory.

```bash
git clone https://github.com/ciricc/go-whisper-server.git
cd go-whisper-server
```

#### 1) One time setup
This installs helper tools and prepares local bin.

```bash
task setup
```

#### 2) Download a model
You can download a model into the local models directory. The default is tiny.en.

```bash
# small example
task dl

# or pick an explicit file name from ggerganov whisper.cpp model list
task download-model MODEL=ggml-large-v3-turbo.bin
```

#### 3) Configure the server
Edit `config.yaml` to point at the model you want and choose the listen address.

```yaml
server:
  address: ":50051"
model:
  path: "domain/ggml-large-v3-turbo.bin"
  language: "en"
  sample_rate_hz: 16000
```

#### 4) Build and run the server
CPU build:

```bash
task run
```

Metal build on macOS (recommended on Apple Silicon):

```bash
task run:metal
```

CUDA build on Linux:

```bash
# install deps (Ubuntu) then build and run
task build:cuda:deps:ubuntu
task run:cuda
```

On success the server logs a line like:

```text
listening on :50051
```

### Try it with the example client
The service expects WAV 16 kHz mono 16-bit PCM. If your audio is not in that format, convert it first.

```bash
ffmpeg -i input_audio.wav -ac 1 -ar 16000 -c:a pcm_s16le testdata/out.wav
```

Run the client:

```bash
go run ./cmd/client -wav testdata/out.wav -addr localhost:50051
```

You should see streaming segment lines:

```text
[     0 ->   1200] hello world
```

### Microphone demo (local, offline)
This does not use gRPC. It opens your microphone, runs whisper locally, and prints segments.

```bash
go run ./cmd/mic -model domain/ggml-large-v3-turbo.bin
```

Notes
- You may need to grant microphone permission on macOS
- If your device sample rate is not 16 kHz the demo resamples on the fly

### API overview
Service definition: `pkg/proto/transcriber.proto`

```proto
service Transcriber {
  rpc TranscribeWav(TranscribeWavRequest) returns (stream Segment);
}

message TranscribeWavRequest {
  string language = 1;                  // spoken language code like en
  File wav_16k_file = 2;                // bytes or path; url is not supported yet
  WhisperParams whisper_params = 3;     // decoding parameters
  TranscribeWavParams transcribe_wav_params = 4; // windowing for WAV reader
}

message Segment {
  google.protobuf.Duration start = 1;
  google.protobuf.Duration end = 2;
  string text = 3;
  bool speaker_turn_next = 4;           // tinydiarize prediction flag
}
```

Important
- WAV input must be PCM16LE, mono, 16000 Hz
- `File.url` is not supported by the server yet

Whisper parameters
- Split on word, beam size, temperature and fallback, max tokens per segment, token thresholds, translate, diarize, VAD, max segment length, token timestamps, offset, duration, initial prompt

### Calling the API with grpcurl
The simplest way is to pass a local file path. Ensure the WAV is PCM16LE mono 16 kHz.

```bash
grpcurl -plaintext \
  -import-path pkg/proto \
  -proto pkg/proto/transcriber.proto \
  -d '{
        "language": "en",
        "wav_16k_file": { "path": "testdata/out.wav" },
        "whisper_params": { "split_on_word": { "value": true } },
        "transcribe_wav_params": { "window_size": "30s" }
      }' \
  localhost:50051 transcriber.v1.Transcriber/TranscribeWav
```

### Project structure
- `cmd/server` entry point for the gRPC server
- `cmd/client` example client for TranscribeWav
- `cmd/mic` microphone demo using the same whisper bindings
- `internal/app` wiring for config, whisper model, and server
- `internal/server` gRPC service implementation and request mapping
- `internal/service/transcribe_svc` service layer that sets whisper parameters and builds tasks
- `internal/model/segment` simple segment model used internally
- `pkg/whisper_lib` WAV and PCM task pipeline built over whisper.cpp Go bindings
- `pkg/proto` protobuf API and generated Go code
- `third_party/whisper.cpp` vendored whisper.cpp tree used by CGO build

### Build and development
Common tasks are defined in `Taskfile.yml`.

Build C++ libs and the Go server (CPU):

```bash
task build
```

Build with Metal (macOS):

```bash
task build:metal
```

Build with CUDA (Linux):

```bash
task build:cuda
```

Regenerate protobufs using EasyP:

```bash
task easyp
```

Clean build artifacts:

```bash
task clean
```

### Configuration reference
`config.yaml` fields:
- `server.address` listen address, example `:50051`
- `model.path` path to a ggml model file in `models`
- `model.language` default language for the model; API can override
- `model.sample_rate_hz` must be 16000 for this pipeline

### Troubleshooting
- Build fails with missing whisper symbols
  - Run `task build` or `task build:metal` again to ensure the C++ library is built and linked
- Runtime error about missing Metal resources on macOS
  - The Taskfile sets `GGML_METAL_PATH_RESOURCES` automatically, but ensure you run from the project root
- gRPC client cannot connect
  - Verify the server is logging `listening on :50051` and that you configured the same address in the client
- Audio rejected with unsupported format
  - Convert to PCM16LE mono 16 kHz using the ffmpeg command above

### License and models
This repository uses whisper.cpp. Make sure you comply with the upstream licenses for code and model weights.


