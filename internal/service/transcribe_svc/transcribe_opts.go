package transcribe_svc

import "time"

type TranscribeOpts struct {
	Language            *string
	WindowSize          *time.Duration
	HopMs               *int
	InitialPrompt       *string
	NoContext           *bool
	SplitOnWord         *bool
	BeamSize            *int
	Temperature         *float32
	TemperatureFallback *float32
	MaxTokensPerSegment *uint
	TokenThreshold      *float32
	TokenSumThreshold   *float32
	Translate           *bool
	Diarize             *bool
	Vad                 *bool
	MaxSegmentLength    *uint
	TokenTimestamps     *bool
	Offset              *time.Duration
	Duration            *time.Duration
}

type TranscribeOpt func(opts *TranscribeOpts)

func WithLanguage(language string) TranscribeOpt {
	return func(opts *TranscribeOpts) {
		opts.Language = &language
	}
}

func WithWindowSize(v time.Duration) TranscribeOpt {
	return func(opts *TranscribeOpts) { opts.WindowSize = &v }
}

func WithHopMs(v int) TranscribeOpt {
	return func(opts *TranscribeOpts) { opts.HopMs = &v }
}

func WithBeamSize(v int) TranscribeOpt {
	return func(opts *TranscribeOpts) { opts.BeamSize = &v }
}

func WithTemperature(v float32) TranscribeOpt {
	return func(opts *TranscribeOpts) { opts.Temperature = &v }
}

func WithTemperatureFallback(v float32) TranscribeOpt {
	return func(opts *TranscribeOpts) { opts.TemperatureFallback = &v }
}

func WithMaxTokensPerSegment(v uint) TranscribeOpt {
	return func(opts *TranscribeOpts) { opts.MaxTokensPerSegment = &v }
}

func WithTokenThreshold(v float32) TranscribeOpt {
	return func(opts *TranscribeOpts) { opts.TokenThreshold = &v }
}

func WithTokenSumThreshold(v float32) TranscribeOpt {
	return func(opts *TranscribeOpts) { opts.TokenSumThreshold = &v }
}

func WithTranslate(v bool) TranscribeOpt {
	return func(opts *TranscribeOpts) { opts.Translate = &v }
}

func WithDiarize(v bool) TranscribeOpt {
	return func(opts *TranscribeOpts) { opts.Diarize = &v }
}

func WithVad(v bool) TranscribeOpt {
	return func(opts *TranscribeOpts) { opts.Vad = &v }
}

func WithMaxSegmentLength(v uint) TranscribeOpt {
	return func(opts *TranscribeOpts) { opts.MaxSegmentLength = &v }
}

func WithTokenTimestamps(v bool) TranscribeOpt {
	return func(opts *TranscribeOpts) { opts.TokenTimestamps = &v }
}

func WithOffset(v time.Duration) TranscribeOpt {
	return func(opts *TranscribeOpts) { opts.Offset = &v }
}

func WithDuration(v time.Duration) TranscribeOpt {
	return func(opts *TranscribeOpts) { opts.Duration = &v }
}

func WithInitialPrompt(v string) TranscribeOpt {
	return func(opts *TranscribeOpts) { opts.InitialPrompt = &v }
}

func WithNoContext(v bool) TranscribeOpt {
	return func(opts *TranscribeOpts) { opts.NoContext = &v }
}

func WithSplitOnWord(v bool) TranscribeOpt {
	return func(opts *TranscribeOpts) { opts.SplitOnWord = &v }
}
