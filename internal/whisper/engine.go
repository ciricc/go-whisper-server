package whisper

import (
	"context"
	"sync"

	w "github.com/ggerganov/whisper.cpp/bindings/go/pkg/whisper"
)

type SpawnTranscribeOpts struct {
	Language string
}

type SpwanTranscribeOpt func(opts *SpawnTranscribeOpts)

func WithLanguage(language string) SpwanTranscribeOpt {
	return func(opts *SpawnTranscribeOpts) {
		opts.Language = language
	}
}

func buildOpts(defaultOpts SpawnTranscribeOpts, opts ...SpwanTranscribeOpt) SpawnTranscribeOpts {
	var o SpawnTranscribeOpts
	o = defaultOpts
	for _, opt := range opts {
		opt(&o)
	}

	return o
}

type Engine struct {
	model w.Model
	mu    sync.Mutex
}

func NewEngine(modelPath string) (*Engine, error) {
	m, err := w.New(modelPath)
	if err != nil {
		return nil, err
	}

	return &Engine{model: m}, nil
}

func (e *Engine) Close() error {
	if e.model == nil {
		return nil
	}

	return e.model.Close()
}

func (e *Engine) SpawnTranscribe(
	ctx context.Context,
	opts ...SpwanTranscribeOpt,
) (w.Context, error) {
	o := buildOpts(SpawnTranscribeOpts{}, opts...)

	wCtx, err := e.model.NewContext()
	if err != nil {
		return nil, err
	}

	wCtx.SetLanguage(o.Language)

	return wCtx, nil
}
