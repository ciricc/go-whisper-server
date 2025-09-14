package segment

import (
	"time"
)

type Segment struct {
	Start           time.Duration
	End             time.Duration
	Text            string
	SpeakerTurnNext bool
}

func (s Segment) Validate() error {
	return nil
}

func NewSegment(
	start time.Duration,
	end time.Duration,
	text string,
	speakerTurnNext bool,
) *Segment {
	return &Segment{
		Start:           start,
		End:             end,
		Text:            text,
		SpeakerTurnNext: speakerTurnNext,
	}
}
