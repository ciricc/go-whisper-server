package server

import (
	"github.com/ciricc/go-whisper-server/internal/service/transcribe_svc"
	transcriberv1 "github.com/ciricc/go-whisper-server/pkg/proto/transcriber/v1"
)

func mapWhisperParamsToTranscribeOpts(
	whisperParams *transcriberv1.WhisperParams,
	transcribeWavParams *transcriberv1.TranscribeWavParams,
) []transcribe_svc.TranscribeOpt {
	transcribeOpts := []transcribe_svc.TranscribeOpt{}

	if whisperParams.GetSplitOnWord() != nil {
		transcribeOpts = append(transcribeOpts, transcribe_svc.WithSplitOnWord(whisperParams.GetSplitOnWord().GetValue()))
	}

	if transcribeWavParams.GetWindowSize() != nil {
		transcribeOpts = append(transcribeOpts, transcribe_svc.WithWindowSize(transcribeWavParams.GetWindowSize().AsDuration()))
	}

	if whisperParams.GetBeamSize() != nil {
		transcribeOpts = append(transcribeOpts, transcribe_svc.WithBeamSize(int(whisperParams.GetBeamSize().GetValue())))
	}

	if whisperParams.GetTranslate() != nil {
		transcribeOpts = append(transcribeOpts, transcribe_svc.WithTranslate(whisperParams.GetTranslate().GetValue()))
	}

	if whisperParams.GetDiarize() != nil {
		transcribeOpts = append(transcribeOpts, transcribe_svc.WithDiarize(whisperParams.GetDiarize().GetValue()))
	}

	if whisperParams.GetTemperature() != nil {
		transcribeOpts = append(transcribeOpts, transcribe_svc.WithTemperature(whisperParams.GetTemperature().GetValue()))
	}

	if whisperParams.GetTemperatureFallback() != nil {
		transcribeOpts = append(transcribeOpts, transcribe_svc.WithTemperatureFallback(whisperParams.GetTemperatureFallback().GetValue()))
	}

	if whisperParams.GetMaxTokensPerSegment() != nil {
		transcribeOpts = append(transcribeOpts, transcribe_svc.WithMaxTokensPerSegment(uint(whisperParams.GetMaxTokensPerSegment().GetValue())))
	}

	if whisperParams.GetTokenThreshold() != nil {
		transcribeOpts = append(transcribeOpts, transcribe_svc.WithTokenThreshold(whisperParams.GetTokenThreshold().GetValue()))
	}

	if whisperParams.GetTokenSumThreshold() != nil {
		transcribeOpts = append(transcribeOpts, transcribe_svc.WithTokenSumThreshold(whisperParams.GetTokenSumThreshold().GetValue()))
	}

	if whisperParams.GetVad() != nil {
		transcribeOpts = append(transcribeOpts, transcribe_svc.WithVad(whisperParams.GetVad().GetValue()))
	}

	if whisperParams.GetMaxSegmentLength() != nil {
		transcribeOpts = append(transcribeOpts, transcribe_svc.WithMaxSegmentLength(uint(whisperParams.GetMaxSegmentLength().GetValue())))
	}

	if whisperParams.GetTokenTimestamps() != nil {
		transcribeOpts = append(transcribeOpts, transcribe_svc.WithTokenTimestamps(whisperParams.GetTokenTimestamps().GetValue()))
	}

	if whisperParams.GetOffset() != nil {
		transcribeOpts = append(transcribeOpts, transcribe_svc.WithOffset(whisperParams.GetOffset().AsDuration()))
	}

	if whisperParams.GetDuration() != nil {
		transcribeOpts = append(transcribeOpts, transcribe_svc.WithDuration(whisperParams.GetDuration().AsDuration()))
	}

	if whisperParams.GetInitialPrompt() != nil {
		transcribeOpts = append(transcribeOpts, transcribe_svc.WithInitialPrompt(whisperParams.GetInitialPrompt().GetValue()))
	}

	return transcribeOpts
}
