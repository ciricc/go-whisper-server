package whisper_lib

import (
	"context"
)

type pcmProducer func(context.Context, chan<- []float32) error

func runPCMWithProducer(
	ctx context.Context,
	task *PCMTranscribeTask,
	produce pcmProducer,
	finish func(error),
) {
	// Create parameterized logger for this method
	log := task.logger.With("method", "runPCMWithProducer")

	pcmCh := make(chan []float32)

	// Start underlying PCM pipeline
	task.Start(ctx, pcmCh)

	go func() {
		if err := produce(ctx, pcmCh); err != nil {
			log.DebugContext(ctx, "produce error",
				"error", err,
			)
			close(pcmCh)
			finish(err)
			return
		}

		log.DebugContext(ctx, "produce done")

		// Signal end-of-input to the PCM task before waiting for it
		close(pcmCh)

		// Wait for the PCM task to finish processing and propagate its terminal error
		finish(task.Wait())
	}()
}
