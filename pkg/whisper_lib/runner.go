package whisper_lib

import (
	"context"
	"fmt"
)

type pcmProducer func(context.Context, chan<- []float32) error

func runPCMWithProducer(
	ctx context.Context,
	task *PCMTranscribeTask,
	produce pcmProducer,
	finish func(error),
) {
	pcmCh := make(chan []float32)

	// Start underlying PCM pipeline
	task.Start(ctx, pcmCh)

	go func() {
		if err := produce(ctx, pcmCh); err != nil {
			fmt.Printf("[runPCMWithProducer] produce error: %v\n", err)
			close(pcmCh)
			finish(err)
			return
		}

		fmt.Printf("[runPCMWithProducer] produce done\n")

		// Signal end-of-input to the PCM task before waiting for it
		close(pcmCh)

		// Wait for the PCM task to finish processing and propagate its terminal error
		finish(task.Wait())
	}()
}
