package main

import (
	"bytes"
	"context"
	"encoding/xml"
	"fmt"
	"os"
	"os/exec"
	"strconv"
	"strings"
	"time"
)

// GPUSample contains a subset of NVIDIA SMI metrics we care about.
type GPUSample struct {
	Name        string
	Index       int
	UtilPercent int
	MemUsedMB   float64
	MemTotalMB  float64
	PowerWatt   float64
	SMClockMHz  int
}

// Minimal XML mapping for nvidia-smi -x -q
type smiLog struct {
	XMLName xml.Name `xml:"nvidia_smi_log"`
	GPU     smiGPU   `xml:"gpu"`
}

type smiGPU struct {
	ProductName string         `xml:"product_name"`
	MinorNumber string         `xml:"minor_number"`
	Util        smiUtilization `xml:"utilization"`
	FBMem       smiFBMemory    `xml:"fb_memory_usage"`
	Power       smiPower       `xml:"power_readings"`
	Clocks      smiClocks      `xml:"clocks"`
}

type smiUtilization struct {
	GPU    string `xml:"gpu_util"`
	Memory string `xml:"memory_util"`
}

type smiFBMemory struct {
	Total string `xml:"total"`
	Used  string `xml:"used"`
	Free  string `xml:"free"`
}

type smiPower struct {
	Draw string `xml:"power_draw"`
}

type smiClocks struct {
	SMClock string `xml:"sm_clock"`
}

func hasNvidiaSMI() bool {
	_, err := exec.LookPath("nvidia-smi")
	return err == nil
}

func parsePercentInt(s string) int {
	s = strings.TrimSpace(s)
	s = strings.TrimSuffix(s, "%")
	s = strings.TrimSpace(s)
	if v, err := strconv.Atoi(s); err == nil {
		return v
	}
	// Some fields can be like "66 %"
	fields := strings.Fields(s)
	if len(fields) > 0 {
		if v, err := strconv.Atoi(fields[0]); err == nil {
			return v
		}
	}
	return 0
}

func parseMiBFloat(s string) float64 {
	s = strings.TrimSpace(s)
	s = strings.TrimSuffix(s, "MiB")
	s = strings.TrimSuffix(s, "MiB ")
	s = strings.TrimSpace(s)
	if v, err := strconv.ParseFloat(s, 64); err == nil {
		return v
	}
	fields := strings.Fields(s)
	if len(fields) > 0 {
		if v, err := strconv.ParseFloat(fields[0], 64); err == nil {
			return v
		}
	}
	return 0
}

func parseWattFloat(s string) float64 {
	s = strings.TrimSpace(s)
	s = strings.TrimSuffix(s, "W")
	s = strings.TrimSpace(s)
	if v, err := strconv.ParseFloat(s, 64); err == nil {
		return v
	}
	fields := strings.Fields(s)
	if len(fields) > 0 {
		if v, err := strconv.ParseFloat(fields[0], 64); err == nil {
			return v
		}
	}
	return 0
}

func parseMHzInt(s string) int {
	s = strings.TrimSpace(s)
	s = strings.TrimSuffix(s, "MHz")
	s = strings.TrimSpace(s)
	if v, err := strconv.Atoi(s); err == nil {
		return v
	}
	fields := strings.Fields(s)
	if len(fields) > 0 {
		if v, err := strconv.Atoi(fields[0]); err == nil {
			return v
		}
	}
	return 0
}

// sampleNvidiaSMIXML executes a single nvidia-smi -x -q sample and parses it.
func sampleNvidiaSMIXML(ctx context.Context, device int) (GPUSample, error) {
	var sample GPUSample
	cmd := exec.CommandContext(ctx, "nvidia-smi", "-x", "-q", "-i", strconv.Itoa(device))
	b, err := cmd.Output()
	if err != nil {
		return sample, err
	}
	dec := xml.NewDecoder(bytes.NewReader(b))
	var log smiLog
	if err := dec.Decode(&log); err != nil {
		return sample, err
	}
	gpu := log.GPU
	sample = GPUSample{
		Name:        strings.TrimSpace(gpu.ProductName),
		Index:       device,
		UtilPercent: parsePercentInt(gpu.Util.GPU),
		MemUsedMB:   parseMiBFloat(gpu.FBMem.Used),
		MemTotalMB:  parseMiBFloat(gpu.FBMem.Total),
		PowerWatt:   parseWattFloat(gpu.Power.Draw),
		SMClockMHz:  parseMHzInt(gpu.Clocks.SMClock),
	}
	return sample, nil
}

// monitorNvidiaSMI prints real-time utilization using XML sampling every interval until stop is closed.
func monitorNvidiaSMI(device int, interval time.Duration, stop <-chan struct{}) {
	if !hasNvidiaSMI() {
		return
	}
	ticker := time.NewTicker(interval)
	defer ticker.Stop()
	for {
		select {
		case <-stop:
			return
		case <-ticker.C:
			ctx, cancel := context.WithTimeout(context.Background(), 1500*time.Millisecond)
			s, err := sampleNvidiaSMIXML(ctx, device)
			cancel()
			if err != nil {
				fmt.Fprintf(os.Stderr, "\n[nvidia-smi] error: %v\n", err)
				continue
			}
			memPct := 0.0
			if s.MemTotalMB > 0 {
				memPct = (s.MemUsedMB / s.MemTotalMB) * 100.0
			}
			// Print one concise line; stderr to avoid colliding with JSON output
			fmt.Fprintf(
				os.Stderr,
				"\rGPU%d %s util=%3d%%, mem=%.0f/%.0f MiB (%2.0f%%), power=%.0f W, sm=%d MHz",
				s.Index, s.Name, s.UtilPercent, s.MemUsedMB, s.MemTotalMB, memPct, s.PowerWatt, s.SMClockMHz,
			)
		}
	}
}
