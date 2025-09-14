package transcribe_svc

import (
	"context"
	"fmt"
	"io"
	"log/slog"
	"time"

	"github.com/ciricc/go-whisper-server/pkg/whisper_lib"
	"github.com/ggerganov/whisper.cpp/bindings/go/pkg/whisper"
	"github.com/samber/lo"
)

type TranscribeService interface {
	// TranscribeWav transcribes a WAV file using a WAV decoder wrapper around the PCM task
	TranscribeWav(
		ctx context.Context,
		file io.ReadSeeker,
		opts ...TranscribeOpt,
	) (whisper_lib.WavTranscribeTask, error)
}

type TranscribeServiceImpl struct {
	whisperModel *whisper.ModelContext
}

func NewTranscribeService(
	whisperModel *whisper.ModelContext,
) *TranscribeServiceImpl {
	return &TranscribeServiceImpl{
		whisperModel: whisperModel,
	}
}

func setParam[T any](opt *T, setter func(opt T)) {
	if opt != nil {
		setter(*opt)
	}
}

func setParamErr[T any](opt *T, setter func(opt T) error) error {
	if opt != nil {
		return setter(*opt)
	}
	return nil
}

func (s *TranscribeServiceImpl) TranscribeWav(
	ctx context.Context,
	file io.ReadSeeker,
	opts ...TranscribeOpt,
) (whisper_lib.WavTranscribeTask, error) {
	o := buildOpts(TranscribeOpts{
		SplitOnWord: lo.ToPtr(true),
		NoContext:   lo.ToPtr(false),
		Temperature: lo.ToPtr(float32(0.0)),
	}, opts...)

	whisperParams, err := whisper.NewParameters(
		s.whisperModel,
		whisper.SamplingStrategy(whisper.SAMPLING_BEAM_SEARCH),
		func(p *whisper.Parameters) {
			slog.Info("opts", "opts", o)
			setParam(o.NoContext, p.SetNoContext)
			setParam(o.SplitOnWord, p.SetSplitOnWord)
			setParam(o.BeamSize, p.SetBeamSize)
			setParam(o.Temperature, p.SetTemperature)
			setParam(o.TemperatureFallback, p.SetTemperatureFallback)
			setParam(o.MaxTokensPerSegment, p.SetMaxTokensPerSegment)
			setParam(o.TokenThreshold, p.SetTokenThreshold)
			setParam(o.TokenSumThreshold, p.SetTokenSumThreshold)
			setParam(o.Translate, p.SetTranslate)
			setParam(o.Diarize, p.SetDiarize)
			setParam(o.Vad, p.SetVAD)
			setParam(o.MaxSegmentLength, p.SetMaxSegmentLength)
			setParam(o.TokenTimestamps, p.SetTokenTimestamps)
			setParam(o.Offset, p.SetOffset)
			setParam(o.Duration, p.SetDuration)
			setParam(o.InitialPrompt, p.SetInitialPrompt)
		},
	)
	if err != nil {
		return nil, fmt.Errorf("failed to create whisper parameters: %w", err)
	}

	if err := setParamErr(o.Language, func(lang string) error {
		return whisperParams.SetLanguage(lang)
	}); err != nil {
		return nil, fmt.Errorf("failed to set language: %w", err)
	}

	whisperCtx, err := whisper.NewStatefulContext(
		s.whisperModel,
		whisperParams,
	)
	if err != nil {
		return nil, fmt.Errorf("failed to create whisper context: %w", err)
	}

	// Create PCM task and WAV wrapper
	pcmTask := whisper_lib.NewPCMTranscribeTask(whisperCtx)
	var defaultWindowSize = time.Second * 30

	if o.WindowSize != nil && *o.WindowSize > 0 {
		defaultWindowSize = *o.WindowSize
	}

	wavTask := whisper_lib.NewWavTranscribeTask(pcmTask, defaultWindowSize)
	wavTask.Start(ctx, file)

	return wavTask, nil
}

func buildOpts(defaultOpts TranscribeOpts, opts ...TranscribeOpt) TranscribeOpts {
	o := defaultOpts
	for _, opt := range opts {
		opt(&o)
	}
	return o
}

var _ TranscribeService = (*TranscribeServiceImpl)(nil)
