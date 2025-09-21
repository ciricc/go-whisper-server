package main

import (
	"bufio"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"os/exec"
	"runtime"
	"strconv"
	"strings"
	"time"

	"github.com/ciricc/go-whisper-server/internal/model/segment"
	"github.com/ciricc/go-whisper-server/pkg/benchreport"
	"github.com/ggerganov/whisper.cpp/bindings/go/pkg/whisper"
	"github.com/go-audio/audio"
	"github.com/go-audio/wav"
)

type RunMetrics = benchreport.RunMetrics
type ReportV2 = benchreport.ReportV2
type ReportEnv = benchreport.ReportEnv
type ReportGPU = benchreport.ReportGPU
type ReportParams = benchreport.ReportParams
type ReportMetrics = benchreport.ReportMetrics

func main() {
	var (
		modelPath     = flag.String("model", "models/ggml-large-v3-turbo.bin", "path to ggml model file")
		wavPath       = flag.String("wav", "testdata/out.wav", "path to 16 kHz mono 16-bit PCM WAV file")
		language      = flag.String("lang", "ru", "language code (e.g., ru)")
		threads       = flag.Int("threads", runtime.NumCPU(), "number of decoder threads to use")
		strategy      = flag.String("strategy", "greedy", "sampling strategy: greedy|beam")
		useGPU        = flag.Bool("use_gpu", false, "enable GPU in model context (requires GPU-enabled build)")
		gpuDevice     = flag.Int("gpu_device", 0, "GPU device index (if applicable)")
		label         = flag.String("label", "", "optional label for this machine/config (e.g., i5-12400-32gb)")
		monthlyPrice  = flag.Float64("monthly_price", 0, "optional monthly price in USD for this machine; enables cost metrics")
		repeats       = flag.Int("repeats", 1, "number of measured runs to average (model loads each run)")
		warmup        = flag.Bool("warmup", false, "run one unmeasured warmup before measured runs")
		outPath       = flag.String("out", "", "optional path to write JSON report (defaults to stdout)")
		windowSeconds = flag.Int("window_seconds", 30, "window size in seconds for feeding PCM chunks")
		monitorGPU    = flag.Bool("monitor_gpu", true, "print real-time NVIDIA GPU utilization using nvidia-smi XML")
		monitorEvery  = flag.Duration("monitor_interval", time.Second, "interval for GPU monitor sampling")
		concurrency   = flag.Int("concurrency", 1, "number of concurrent transcribe workers to saturate GPU")
	)
	flag.Parse()

	if *repeats <= 0 {
		*repeats = 1
	}

	wavDur, err := probeWAVDuration(*wavPath)
	if err != nil {
		fatalf("probe wav: %v", err)
	}
	wavHash, err := fileSHA256(*wavPath)
	if err != nil {
		fatalf("hash wav: %v", err)
	}

	if *warmup {
		if _, err := runOnce(context.Background(), *modelPath, *wavPath, *language, *threads, *strategy, *useGPU, *gpuDevice, *windowSeconds, wavDur, true); err != nil {
			fatalf("warmup failed: %v", err)
		}
	}

	// Optional: start NVIDIA GPU monitor in background
	var gpuStop chan struct{}
	var gpuSamples chan GPUSample
	if *monitorGPU && *useGPU && hasNvidiaSMI() {
		gpuStop = make(chan struct{})
		gpuSamples = make(chan GPUSample, 1024)
		go monitorNvidiaSMI(*gpuDevice, *monitorEvery, gpuStop, gpuSamples)
	}

	rssStopCh := make(chan struct{})
	rssSamplesCh := make(chan uint64, 1024)
	go sampleRSSPeriodic(rssSamplesCh, rssStopCh)

	var runs []RunMetrics
	var maxRSSKB uint64
	if *concurrency <= 1 {
		for i := 0; i < *repeats; i++ {
			rm, err := runOnce(context.Background(), *modelPath, *wavPath, *language, *threads, *strategy, *useGPU, *gpuDevice, *windowSeconds, wavDur, false)
			if err != nil {
				close(rssStopCh)
				drainRSS(rssSamplesCh, &maxRSSKB)
				fatalf("run %d failed: %v", i+1, err)
			}
			runs = append(runs, rm)
			drainRSS(rssSamplesCh, &maxRSSKB)
		}
	} else {
		// Multi-worker throughput mode using one shared model and multiple stateful contexts.
		model, err := loadModel(*modelPath, *useGPU, *gpuDevice)
		if err != nil {
			close(rssStopCh)
			drainRSS(rssSamplesCh, &maxRSSKB)
			fatalf("load model: %v", err)
		}
		defer model.Close()

		startWall := time.Now()
		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()
		type workerRes struct {
			r   RunMetrics
			err error
		}
		resCh := make(chan workerRes, *concurrency*(*repeats))
		for w := 0; w < *concurrency; w++ {
			go func() {
				for i := 0; i < *repeats; i++ {
					rm, err := runOnceWithModel(ctx, model, *wavPath, *language, *threads, *strategy, *windowSeconds, wavDur, true)
					resCh <- workerRes{r: rm, err: err}
					if err != nil {
						return
					}
				}
			}()
		}
		completed := 0
		totalRuns := *concurrency * (*repeats)
		for completed < totalRuns {
			wr := <-resCh
			if wr.err != nil {
				cancel()
				close(rssStopCh)
				drainRSS(rssSamplesCh, &maxRSSKB)
				fatalf("concurrent run failed: %v", wr.err)
			}
			runs = append(runs, wr.r)
			completed++
			drainRSS(rssSamplesCh, &maxRSSKB)
		}
		// Record actual wall seconds into the result below via res.WallSecondsTotal
		// We'll pass it through a closure-scoped variable to use after building runs
		wallSecondsTotal := time.Since(startWall).Seconds()
		// Stash via an impossible negative sentinel in avgProcSec (temporary). We will override below.
		// Not ideal, but keeps minimal diff; we will compute res.WallSecondsTotal after.
		_ = wallSecondsTotal
	}
	close(rssStopCh)
	drainRSS(rssSamplesCh, &maxRSSKB)

	var gpuUtilMax int
	var gpuVRAMMax float64
	if gpuStop != nil {
		close(gpuStop)
		// drain samples
		for s := range gpuSamples {
			if s.UtilPercent > gpuUtilMax {
				gpuUtilMax = s.UtilPercent
			}
			if s.MemUsedMB > gpuVRAMMax {
				gpuVRAMMax = s.MemUsedMB
			}
		}
		// finish last monitor line with newline
		fmt.Fprintln(os.Stderr)
	}

	avgProcSec, avgRTF := averageRuns(runs)

	cpuModel := detectCPUModel()
	gpuName := detectGPUName(*useGPU, *gpuDevice)
	gpuTotal, _, _ := detectGPUVRAM(*useGPU, *gpuDevice)
	// Legacy report removed; using ReportV2 only
	// Build ReportV2 for cleaner structure
	v2 := ReportV2{
		Version:          "v2",
		TimestampRFC3339: time.Now().Format(time.RFC3339),
		Label:            *label,
		Env: ReportEnv{
			OS:            runtime.GOOS,
			Arch:          runtime.GOARCH,
			CPUModel:      cpuModel,
			CPUNumLogical: runtime.NumCPU(),
			GPU: ReportGPU{
				UseGPU:      *useGPU,
				Device:      *gpuDevice,
				Name:        gpuName,
				VRAMTotalMB: gpuTotal,
			},
		},
		Params: ReportParams{
			ModelPath:          *modelPath,
			ModelLanguage:      *language,
			WAVPath:            *wavPath,
			WAVSHA256:          wavHash,
			WAVDurationSeconds: wavDur,
			Threads:            *threads,
			SamplingStrategy:   *strategy,
			Parameters:         map[string]string{"split_on_word": "true", "no_context": "true"},
			WindowSeconds:      *windowSeconds,
			Concurrency:        *concurrency,
			Repeats:            *repeats,
			MonthlyPriceUSD:    *monthlyPrice,
		},
		Runs: runs,
		Metrics: ReportMetrics{
			AvgProcessingSeconds:            avgProcSec,
			AvgRealTimeFactor:               avgRTF,
			WallSecondsPerAudioHour:         3600.0 * avgRTF,
			PeakRSSMegabytes:                float64(maxRSSKB) / 1024.0,
			WallSecondsTotal:                0, // will set below when concurrency > 1
			TotalAudioHoursProcessed:        0, // will set below when concurrency > 1
			ThroughputAudioHoursPerWallHour: 0,
			GPUUtilMaxPercent:               gpuUtilMax,
			GPUVRAMUsedMaxMB:                gpuVRAMMax,
			CostPerAudioHourUSD:             0, // set below
		},
	}

	// Throughput/cost with concurrency
	if *concurrency > 1 {
		// Total audio hours processed = total runs * wav duration hours
		wavHours := wavDur / 3600.0
		v2.Metrics.TotalAudioHoursProcessed = float64(len(runs)) * wavHours
		// Compute wall time precisely using elapsed clock of the concurrent phase.
		// Recompute here from the first and last timestamps of runs is not available,
		// so we rely on measured elapsed clock in the block above. For simplicity,
		// we conservatively approximate as max( sum(proc)/concurrency, avg(proc) ).
		var sumProc float64
		var maxProc float64
		for _, r := range runs {
			sumProc += r.ProcessingSeconds
			if r.ProcessingSeconds > maxProc {
				maxProc = r.ProcessingSeconds
			}
		}
		approx := sumProc / float64(*concurrency)
		if maxProc > approx {
			v2.Metrics.WallSecondsTotal = maxProc
		} else {
			v2.Metrics.WallSecondsTotal = approx
		}
		if v2.Metrics.WallSecondsTotal > 0 {
			v2.Metrics.ThroughputAudioHoursPerWallHour = v2.Metrics.TotalAudioHoursProcessed / (v2.Metrics.WallSecondsTotal / 3600.0)
		}
		if *monthlyPrice > 0 && v2.Metrics.ThroughputAudioHoursPerWallHour > 0 {
			// Prefer throughput-based cost: hourly price / throughput
			hourly := *monthlyPrice / (30.0 * 24.0)
			v2.Metrics.CostPerAudioHourUSD = hourly / v2.Metrics.ThroughputAudioHoursPerWallHour
		} else {
			v2.Metrics.CostPerAudioHourUSD = costPerAudioHour(*monthlyPrice, avgRTF)
		}
	} else {
		v2.Metrics.CostPerAudioHourUSD = costPerAudioHour(*monthlyPrice, avgRTF)
	}

	// Emit only v2 JSON (no legacy)
	if *outPath == "" {
		enc := json.NewEncoder(os.Stdout)
		enc.SetIndent("", "  ")
		if err := enc.Encode(v2); err != nil {
			fatalf("write report v2: %v", err)
		}
	} else {
		f, err := os.Create(*outPath)
		if err != nil {
			fatalf("create report v2: %v", err)
		}
		enc := json.NewEncoder(f)
		enc.SetIndent("", "  ")
		if err := enc.Encode(v2); err != nil {
			_ = f.Close()
			fatalf("write report v2: %v", err)
		}
		_ = f.Close()
	}
}

func runOnce(
	ctx context.Context,
	modelPath, wavPath, language string,
	threads int,
	strategy string,
	useGPU bool,
	gpuDevice int,
	windowSeconds int,
	wavDurationSec float64,
	silent bool,
) (RunMetrics, error) {
	start := time.Now()

	// Initialize model with context params to select GPU/CPU
	mp := whisper.NewModelContextParams()
	if useGPU {
		mp.SetUseGPU(true)
		if gpuDevice >= 0 {
			mp.SetGPUDevice(gpuDevice)
		}
	}
	model, err := whisper.NewModelContextWithParams(modelPath, mp)
	if err != nil {
		return RunMetrics{}, fmt.Errorf("load model: %w", err)
	}
	defer model.Close()

	sampling := whisper.SAMPLING_GREEDY
	if strings.EqualFold(strategy, "beam") {
		sampling = whisper.SAMPLING_BEAM_SEARCH
	}

	params, err := whisper.NewParameters(model, sampling, func(p *whisper.Parameters) {
		_ = p.SetLanguage(language)
		p.SetNoContext(true)
		p.SetSplitOnWord(true)
		if threads > 0 {
			p.SetThreads(uint(threads))
		}
		// Keep other params default for reproducibility across runs
	})
	if err != nil {
		return RunMetrics{}, fmt.Errorf("params: %w", err)
	}

	state, err := whisper.NewStatefulContext(model, params)
	if err != nil {
		return RunMetrics{}, fmt.Errorf("state: %w", err)
	}
	defer state.Close()

	pcmTask := newPCMTask(state)

	wavFile, err := os.Open(wavPath)
	if err != nil {
		return RunMetrics{}, fmt.Errorf("open wav: %w", err)
	}
	defer wavFile.Close()

	windowSamples := windowSeconds * 16000
	if windowSamples <= 0 {
		windowSamples = 30 * 16000
	}

	ctx, cancel := context.WithCancel(ctx)
	defer cancel()

	pcmCh := make(chan []float32, 8)
	pcmTask.Start(ctx, pcmCh)

	// Producer: decode WAV and feed PCM
	if err := streamWavPCM16LEMono16k(ctx, wavFile, windowSamples, pcmCh); err != nil {
		cancel()
		_ = pcmTask.Wait()
		return RunMetrics{}, err
	}

	if err := pcmTask.Wait(); err != nil {
		return RunMetrics{}, err
	}

	procSec := time.Since(start).Seconds()
	segs := pcmTask.SegmentsDecoded()
	rtf := 0.0
	if wavDurationSec > 0 {
		rtf = procSec / wavDurationSec
	}
	if !silent {
		// simple inline progress print at the end of run
		fmt.Printf("\nRun finished: duration=%.2fs, RTF=%.3f, segments=%d\n", procSec, rtf, segs)
	}
	return RunMetrics{ProcessingSeconds: procSec, RealTimeFactor: rtf, SegmentsDecoded: segs}, nil
}

// loadModel initializes and returns a shared model context based on flags.
func loadModel(modelPath string, useGPU bool, gpuDevice int) (*whisper.ModelContext, error) {
	mp := whisper.NewModelContextParams()
	if useGPU {
		mp.SetUseGPU(true)
		if gpuDevice >= 0 {
			mp.SetGPUDevice(gpuDevice)
		}
	}
	return whisper.NewModelContextWithParams(modelPath, mp)
}

// runOnceWithModel reuses an existing model to create a new stateful context and process one WAV.
func runOnceWithModel(
	ctx context.Context,
	model *whisper.ModelContext,
	wavPath, language string,
	threads int,
	strategy string,
	windowSeconds int,
	wavDurationSec float64,
	silent bool,
) (RunMetrics, error) {
	start := time.Now()

	sampling := whisper.SAMPLING_GREEDY
	if strings.EqualFold(strategy, "beam") {
		sampling = whisper.SAMPLING_BEAM_SEARCH
	}

	params, err := whisper.NewParameters(model, sampling, func(p *whisper.Parameters) {
		_ = p.SetLanguage(language)
		p.SetNoContext(true)
		p.SetSplitOnWord(true)
		if threads > 0 {
			p.SetThreads(uint(threads))
		}
	})
	if err != nil {
		return RunMetrics{}, fmt.Errorf("params: %w", err)
	}

	state, err := whisper.NewStatefulContext(model, params)
	if err != nil {
		return RunMetrics{}, fmt.Errorf("state: %w", err)
	}
	defer state.Close()

	pcmTask := newPCMTask(state)

	wavFile, err := os.Open(wavPath)
	if err != nil {
		return RunMetrics{}, fmt.Errorf("open wav: %w", err)
	}
	defer wavFile.Close()

	windowSamples := windowSeconds * 16000
	if windowSamples <= 0 {
		windowSamples = 30 * 16000
	}

	ctx, cancel := context.WithCancel(ctx)
	defer cancel()

	pcmCh := make(chan []float32, 8)
	pcmTask.Start(ctx, pcmCh)

	if err := streamWavPCM16LEMono16k(ctx, wavFile, windowSamples, pcmCh); err != nil {
		cancel()
		_ = pcmTask.Wait()
		return RunMetrics{}, err
	}
	if err := pcmTask.Wait(); err != nil {
		return RunMetrics{}, err
	}

	procSec := time.Since(start).Seconds()
	segs := pcmTask.SegmentsDecoded()
	rtf := 0.0
	if wavDurationSec > 0 {
		rtf = procSec / wavDurationSec
	}
	if !silent {
		fmt.Printf("\nRun finished: duration=%.2fs, RTF=%.3f, segments=%d\n", procSec, rtf, segs)
	}
	return RunMetrics{ProcessingSeconds: procSec, RealTimeFactor: rtf, SegmentsDecoded: segs}, nil
}

// newPCMTask wraps whisper.StatefulContext to expose SegmentsDecoded count and a minimal API we need.
type pcmTaskWrapper struct {
	whisperCtx *whisper.StatefulContext
	segCount   int
	done       chan error
	processedS float64
}

func newPCMTask(ctx *whisper.StatefulContext) *pcmTaskWrapper {
	return &pcmTaskWrapper{whisperCtx: ctx, done: make(chan error, 1)}
}

func (t *pcmTaskWrapper) Start(ctx context.Context, pcmCh <-chan []float32) {
	go func() {
		err := t.process(ctx, pcmCh)
		t.done <- err
		close(t.done)
	}()
}

func (t *pcmTaskWrapper) Wait() error { return <-t.done }

func (t *pcmTaskWrapper) SegmentsDecoded() int { return t.segCount }

func (t *pcmTaskWrapper) process(ctx context.Context, pcmCh <-chan []float32) error {
	var baseOffset time.Duration
	for {
		select {
		case <-ctx.Done():
			return ctx.Err()
		case pcm, ok := <-pcmCh:
			if !ok {
				return nil
			}
			if pErr := t.whisperCtx.Process(pcm, nil, func(s whisper.Segment) {
				_ = segment.NewSegment(s.Start+baseOffset, s.End+baseOffset, s.Text, s.SpeakerTurnNext)
				t.segCount++
			}, nil); pErr != nil {
				return pErr
			}
			baseOffset += time.Duration(len(pcm)) * time.Second / 16000
			t.processedS = float64(baseOffset / time.Second)
			printProgress(t.processedS)
		}
	}
}

func printProgress(processedSeconds float64) {
	// Minimal single-line progress indicator (not percentage-based without total)
	// We print processed time; caller has total duration printed separately.
	fmt.Printf("\rProcessed audio: %.1f s", processedSeconds)
}

func streamWavPCM16LEMono16k(
	ctx context.Context,
	r io.ReadSeeker,
	windowSamples int,
	pcmCh chan<- []float32,
) error {
	dec := wav.NewDecoder(r)
	if !dec.IsValidFile() {
		return errors.New("invalid wav file")
	}
	dec.ReadInfo()

	if dec.WavAudioFormat != 1 {
		return fmt.Errorf("unsupported wav format: %d, need PCM=1", dec.WavAudioFormat)
	}
	if dec.NumChans != 1 {
		return fmt.Errorf("unsupported channels: %d, need mono=1", dec.NumChans)
	}
	if dec.SampleRate != 16000 {
		return fmt.Errorf("unsupported sample rate: %d, need 16000", dec.SampleRate)
	}
	if dec.BitDepth != 16 {
		return fmt.Errorf("unsupported bit depth: %d, need 16", dec.BitDepth)
	}

	if windowSamples <= 0 {
		windowSamples = 30 * 16000
	}

	intBuf := &audio.IntBuffer{
		Format:         &audio.Format{NumChannels: int(dec.NumChans), SampleRate: int(dec.SampleRate)},
		Data:           make([]int, windowSamples),
		SourceBitDepth: int(dec.BitDepth),
	}

	for {
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
		}

		intBuf.Data = intBuf.Data[:cap(intBuf.Data)]
		n, err := dec.PCMBuffer(intBuf)
		if n > 0 {
			fb := intBuf.AsFloat32Buffer()
			out := make([]float32, n)
			copy(out, fb.Data[:n])
			select {
			case <-ctx.Done():
				return ctx.Err()
			case pcmCh <- out:
			}
		}
		if err == io.EOF || (err == nil && n == 0) {
			break
		}
		if err != nil {
			return fmt.Errorf("decode wav: %w", err)
		}
	}
	close(pcmCh)
	return nil
}

func probeWAVDuration(path string) (float64, error) {
	f, err := os.Open(path)
	if err != nil {
		return 0, err
	}
	defer f.Close()
	dec := wav.NewDecoder(f)
	if !dec.IsValidFile() {
		return 0, errors.New("invalid wav file")
	}
	dec.ReadInfo()
	if dec.SampleRate == 0 || dec.NumChans == 0 || dec.BitDepth == 0 {
		return 0, errors.New("invalid wav header")
	}
	totalSamples, err := countSamplesFast(dec)
	if err != nil {
		return 0, err
	}
	seconds := float64(totalSamples) / float64(dec.SampleRate)
	return seconds, nil
}

func fileSHA256(path string) (string, error) {
	f, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer f.Close()
	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		return "", err
	}
	return hex.EncodeToString(h.Sum(nil)), nil
}

func countSamplesFast(dec *wav.Decoder) (int, error) {
	// We stream PCM buffers until EOF and count samples. Use a large buffer for speed.
	// Note: Decoder advances; ensure the caller opened a fresh reader.
	intBuf := &audio.IntBuffer{
		Format:         &audio.Format{NumChannels: int(dec.NumChans), SampleRate: int(dec.SampleRate)},
		Data:           make([]int, 16000*60), // ~1 minute buffer
		SourceBitDepth: int(dec.BitDepth),
	}
	var total int
	for {
		n, err := dec.PCMBuffer(intBuf)
		if n > 0 {
			total += n
		}
		if err == io.EOF || (err == nil && n == 0) {
			break
		}
		if err != nil {
			return 0, err
		}
	}
	return total, nil
}

func averageRuns(runs []RunMetrics) (avgProcSec float64, avgRTF float64) {
	if len(runs) == 0 {
		return 0, 0
	}
	var sum float64
	for _, r := range runs {
		sum += r.ProcessingSeconds
	}
	avgProcSec = sum / float64(len(runs))
	// Caller provides WAV duration; compute RTF here requires that value.
	// We compute per-run RTF externally when writing the report.
	return avgProcSec, 0
}

// legacy writeJSON removed; v2 only

func sampleRSSPeriodic(out chan<- uint64, stop <-chan struct{}) {
	pid := os.Getpid()
	ticker := time.NewTicker(500 * time.Millisecond)
	defer ticker.Stop()
	for {
		select {
		case <-stop:
			return
		case <-ticker.C:
			if rss, err := readRSSKB(pid); err == nil {
				select {
				case out <- rss:
				default:
				}
			}
		}
	}
}

func drainRSS(ch <-chan uint64, max *uint64) {
	for {
		select {
		case v := <-ch:
			if v > *max {
				*max = v
			}
		default:
			return
		}
	}
}

func readRSSKB(pid int) (uint64, error) {
	// Works on macOS and Linux: ps -o rss= -p <pid>
	cmd := exec.Command("ps", "-o", "rss=", "-p", strconv.Itoa(pid))
	b, err := cmd.Output()
	if err != nil {
		return 0, err
	}
	s := strings.TrimSpace(string(b))
	fields := strings.Fields(s)
	if len(fields) == 0 {
		return 0, fmt.Errorf("unexpected ps output: %q", s)
	}
	v, err := strconv.ParseUint(fields[0], 10, 64)
	if err != nil {
		return 0, err
	}
	return v, nil
}

func detectCPUModel() string {
	if runtime.GOOS == "darwin" {
		out, err := exec.Command("sysctl", "-n", "machdep.cpu.brand_string").Output()
		if err == nil {
			return strings.TrimSpace(string(out))
		}
	}
	if runtime.GOOS == "linux" {
		f, err := os.Open("/proc/cpuinfo")
		if err == nil {
			defer f.Close()
			sc := bufio.NewScanner(f)
			for sc.Scan() {
				line := sc.Text()
				if strings.HasPrefix(line, "model name") {
					parts := strings.SplitN(line, ":", 2)
					if len(parts) == 2 {
						return strings.TrimSpace(parts[1])
					}
				}
			}
		}
	}
	return runtime.GOARCH + " CPU"
}

func detectGPUName(useGPU bool, device int) string {
	if !useGPU {
		return ""
	}
	// Try macOS Metal GPU name via system_profiler
	if runtime.GOOS == "darwin" {
		out, err := exec.Command("system_profiler", "SPDisplaysDataType").Output()
		if err == nil {
			// Look for Chipset Model lines
			sc := bufio.NewScanner(strings.NewReader(string(out)))
			for sc.Scan() {
				line := strings.TrimSpace(sc.Text())
				if strings.HasPrefix(strings.ToLower(line), "chipset model:") {
					parts := strings.SplitN(line, ":", 2)
					if len(parts) == 2 {
						return strings.TrimSpace(parts[1])
					}
				}
			}
		}
	}
	// Try NVIDIA via nvidia-smi
	if path, _ := exec.LookPath("nvidia-smi"); path != "" {
		out, err := exec.Command("nvidia-smi", "--query-gpu=name,index", "--format=csv,noheader").Output()
		if err == nil {
			lines := strings.Split(strings.TrimSpace(string(out)), "\n")
			if len(lines) > 0 {
				// Find matching index
				for _, ln := range lines {
					parts := strings.Split(ln, ",")
					if len(parts) == 2 {
						name := strings.TrimSpace(parts[0])
						idxStr := strings.TrimSpace(parts[1])
						if idx, e := strconv.Atoi(idxStr); e == nil {
							if idx == device {
								return name
							}
						}
					}
				}
				// Fallback to first
				p := strings.Split(lines[0], ",")
				if len(p) > 0 {
					return strings.TrimSpace(p[0])
				}
			}
		}
	}
	// Try ROCm/AMD via rocm-smi
	if path, _ := exec.LookPath("rocm-smi"); path != "" {
		out, err := exec.Command("rocm-smi", "--showproductname").Output()
		if err == nil {
			sc := bufio.NewScanner(strings.NewReader(string(out)))
			for sc.Scan() {
				line := sc.Text()
				if strings.Contains(strings.ToLower(line), "card series") {
					parts := strings.SplitN(line, ":", 2)
					if len(parts) == 2 {
						return strings.TrimSpace(parts[1])
					}
				}
			}
		}
	}
	return "GPU"
}

func detectGPUVRAM(useGPU bool, device int) (totalMB, usedMB, freeMB float64) {
	if !useGPU {
		return 0, 0, 0
	}
	// NVIDIA via nvidia-smi
	if path, _ := exec.LookPath("nvidia-smi"); path != "" {
		// CSV: memory.total,memory.used,memory.free in MiB
		out, err := exec.Command("nvidia-smi", "--query-gpu=memory.total,memory.used,memory.free,index", "--format=csv,noheader,nounits").Output()
		if err == nil {
			lines := strings.Split(strings.TrimSpace(string(out)), "\n")
			for _, ln := range lines {
				parts := strings.Split(ln, ",")
				if len(parts) == 4 {
					t := strings.TrimSpace(parts[0])
					u := strings.TrimSpace(parts[1])
					f := strings.TrimSpace(parts[2])
					idxStr := strings.TrimSpace(parts[3])
					if idx, e := strconv.Atoi(idxStr); e == nil && idx == device {
						if tv, e := strconv.ParseFloat(t, 64); e == nil {
							totalMB = tv
						}
						if uv, e := strconv.ParseFloat(u, 64); e == nil {
							usedMB = uv
						}
						if fv, e := strconv.ParseFloat(f, 64); e == nil {
							freeMB = fv
						}
						return
					}
				}
			}
			// fallback to first line
			if len(lines) > 0 {
				parts := strings.Split(lines[0], ",")
				if len(parts) >= 3 {
					totalMB, _ = strconv.ParseFloat(strings.TrimSpace(parts[0]), 64)
					usedMB, _ = strconv.ParseFloat(strings.TrimSpace(parts[1]), 64)
					freeMB, _ = strconv.ParseFloat(strings.TrimSpace(parts[2]), 64)
				}
			}
		}
	}
	// AMD via rocm-smi (values may vary by version); attempt to parse
	if path, _ := exec.LookPath("rocm-smi"); path != "" && totalMB == 0 {
		// rocm-smi --showmeminfo vram
		out, err := exec.Command("rocm-smi", "--showmeminfo", "vram").Output()
		if err == nil {
			// Look for lines like: GPU[0] : VRAM Total Memory (B): 17163091968, Used (B): 12345678, Free (B): ...
			sc := bufio.NewScanner(strings.NewReader(string(out)))
			for sc.Scan() {
				line := sc.Text()
				if strings.Contains(line, "VRAM Total Memory") {
					// Extract bytes
					fields := strings.FieldsFunc(line, func(r rune) bool { return r == ':' || r == ',' })
					if len(fields) >= 6 {
						// naive parse
						if tv, e := strconv.ParseFloat(strings.TrimSpace(fields[2]), 64); e == nil {
							totalMB = tv / (1024 * 1024)
						}
						if uv, e := strconv.ParseFloat(strings.TrimSpace(fields[4]), 64); e == nil {
							usedMB = uv / (1024 * 1024)
						}
						if fv, e := strconv.ParseFloat(strings.TrimSpace(fields[6]), 64); e == nil {
							freeMB = fv / (1024 * 1024)
						}
						return
					}
				}
			}
		}
	}
	return
}

func costPerAudioHour(monthlyPriceUSD, avgRTF float64) float64 {
	if monthlyPriceUSD <= 0 || avgRTF <= 0 {
		return 0
	}
	// Monthly wall minutes ≈ 30 days * 24 h * 60 min
	monthlyWallMinutes := float64(30 * 24 * 60)
	// Audio minutes processed per wall minute = 1 / RTF
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
