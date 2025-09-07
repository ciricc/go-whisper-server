package service

import (
	"encoding/binary"
)

// PCM16LEToFloat32 converts little-endian int16 PCM to float32 in [-1,1].
func PCM16LEToFloat32(pcm []byte) []float32 {
	count := len(pcm) / 2
	out := make([]float32, count)

	for i := range count {
		v := int16(binary.LittleEndian.Uint16(pcm[2*i:]))
		out[i] = float32(v) / 32768.0
	}

	return out
}
