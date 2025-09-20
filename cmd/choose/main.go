package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/ciricc/go-whisper-server/pkg/benchreport"
)

type BenchmarkResult = benchreport.BenchmarkResult

type Choice struct {
	Path   string
	Report BenchmarkResult
}

func main() {
	var (
		dir        = flag.String("dir", "reports", "directory containing JSON benchmark reports")
		maxResults = flag.Int("n", 5, "number of top configurations to print")
		prefer     = flag.String("prefer", "cost", "preference if tie: cost|speed")
	)
	flag.Parse()

	choices, err := loadReports(*dir)
	if err != nil {
		fatalf("load reports: %v", err)
	}
	if len(choices) == 0 {
		fatalf("no reports found in %s", *dir)
	}

	// Ensure derived metrics exist
	for i := range choices {
		if choices[i].Report.AvgRealTimeFactor <= 0 && choices[i].Report.WAVDurationSeconds > 0 && choices[i].Report.AvgProcessingSeconds > 0 {
			choices[i].Report.AvgRealTimeFactor = choices[i].Report.AvgProcessingSeconds / choices[i].Report.WAVDurationSeconds
		}
		if choices[i].Report.CostPerAudioHourUSD <= 0 && choices[i].Report.MonthlyPriceUSD > 0 && choices[i].Report.AvgRealTimeFactor > 0 {
			choices[i].Report.CostPerAudioHourUSD = costPerAudioHour(choices[i].Report.MonthlyPriceUSD, choices[i].Report.AvgRealTimeFactor)
		}
	}

	// Pareto front: minimize both cost per hour (if available) and RTF
	pareto := paretoFront(choices)

	// Sort for output
	sort.Slice(pareto, func(i, j int) bool {
		ai, aj := pareto[i].Report, pareto[j].Report
		// Primary: available cost
		if ai.CostPerAudioHourUSD > 0 && aj.CostPerAudioHourUSD > 0 {
			if ai.CostPerAudioHourUSD != aj.CostPerAudioHourUSD {
				return ai.CostPerAudioHourUSD < aj.CostPerAudioHourUSD
			}
		}
		if strings.EqualFold(*prefer, "speed") {
			if ai.AvgRealTimeFactor != aj.AvgRealTimeFactor {
				return ai.AvgRealTimeFactor < aj.AvgRealTimeFactor
			}
		} else {
			// Default prefer cost; then speed
			if ai.AvgRealTimeFactor != aj.AvgRealTimeFactor {
				return ai.AvgRealTimeFactor < aj.AvgRealTimeFactor
			}
		}
		// Tiebreak: lower RAM, then threads
		if ai.PeakRSSMegabytes != aj.PeakRSSMegabytes {
			return ai.PeakRSSMegabytes < aj.PeakRSSMegabytes
		}
		return ai.Threads < aj.Threads
	})

	if *maxResults > len(pareto) {
		*maxResults = len(pareto)
	}

	for i := 0; i < *maxResults; i++ {
		r := pareto[i].Report
		fmt.Printf("%d) %s\n", i+1, pareto[i].Path)
		fmt.Printf("   label=%s cpu=\"%s\" threads=%d gpu=%v gpu_name=\"%s\" gpu_device=%d\n", r.Label, r.CPUModel, r.Threads, r.UseGPU, r.GPUName, r.GPUDevice)
		fmt.Printf("   model=%s lang=%s rtf=%.4f wall_s_per_audio_hour=%.1f avg_proc_s=%.2f peak_rss_mb=%.1f\n", r.ModelPath, r.ModelLanguage, r.AvgRealTimeFactor, r.WallSecondsPerAudioHour, r.AvgProcessingSeconds, r.PeakRSSMegabytes)
		if r.CostPerAudioHourUSD > 0 && r.MonthlyPriceUSD > 0 {
			fmt.Printf("   price=$%.2f/mo cost_per_audio_hour=$%.4f\n", r.MonthlyPriceUSD, r.CostPerAudioHourUSD)
		}
	}
}

func loadReports(dir string) ([]Choice, error) {
	var out []Choice
	walk := func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if d.IsDir() {
			return nil
		}
		if !strings.HasSuffix(strings.ToLower(d.Name()), ".json") {
			return nil
		}
		f, err := os.Open(path)
		if err != nil {
			return err
		}
		defer f.Close()
		var r benchreport.BenchmarkResult
		if err := json.NewDecoder(f).Decode(&r); err != nil {
			return nil
		}
		out = append(out, Choice{Path: path, Report: r})
		return nil
	}
	if err := filepath.WalkDir(dir, walk); err != nil {
		return nil, err
	}
	return out, nil
}

func paretoFront(in []Choice) []Choice {
	var out []Choice
	for i := range in {
		dominated := false
		for j := range in {
			if i == j {
				continue
			}
			if dominates(in[j].Report, in[i].Report) {
				dominated = true
				break
			}
		}
		if !dominated {
			out = append(out, in[i])
		}
	}
	return out
}

func dominates(a, b BenchmarkResult) bool {
	// Smaller is better for both cost and RTF; if cost missing, compare RTF only
	betterOrEqualCost := (a.CostPerAudioHourUSD == 0 && b.CostPerAudioHourUSD == 0) || (a.CostPerAudioHourUSD > 0 && b.CostPerAudioHourUSD > 0 && a.CostPerAudioHourUSD <= b.CostPerAudioHourUSD)
	strictlyBetterCost := a.CostPerAudioHourUSD > 0 && b.CostPerAudioHourUSD > 0 && a.CostPerAudioHourUSD < b.CostPerAudioHourUSD
	betterOrEqualRTF := a.AvgRealTimeFactor <= b.AvgRealTimeFactor
	strictlyBetterRTF := a.AvgRealTimeFactor < b.AvgRealTimeFactor

	return (betterOrEqualCost && strictlyBetterRTF) || (strictlyBetterCost && betterOrEqualRTF)
}

func costPerAudioHour(monthlyPriceUSD, avgRTF float64) float64 {
	if monthlyPriceUSD <= 0 || avgRTF <= 0 {
		return 0
	}
	monthlyWallMinutes := float64(30 * 24 * 60)
	monthlyAudioHours := (monthlyWallMinutes / avgRTF) / 60.0
	if monthlyAudioHours <= 0 {
		return 0
	}
	return monthlyPriceUSD / monthlyAudioHours
}

func fatalf(format string, a ...any) {
	_, _ = fmt.Fprintf(os.Stderr, format+"\n", a...)
	os.Exit(1)
}
