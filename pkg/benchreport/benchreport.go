package benchreport

type RunMetrics struct {
	ProcessingSeconds float64 `json:"processing_seconds"`
	RealTimeFactor    float64 `json:"real_time_factor"`
	SegmentsDecoded   int     `json:"segments_decoded"`
}

type BenchmarkResult struct {
	TimestampRFC3339        string            `json:"timestamp_rfc3339"`
	Label                   string            `json:"label"`
	UseGPU                  bool              `json:"use_gpu"`
	GPUDevice               int               `json:"gpu_device"`
	GPUName                 string            `json:"gpu_name"`
	ModelPath               string            `json:"model_path"`
	ModelLanguage           string            `json:"model_language"`
	WAVPath                 string            `json:"wav_path"`
	WAVSHA256               string            `json:"wav_sha256"`
	WAVDurationSeconds      float64           `json:"wav_duration_seconds"`
	Threads                 int               `json:"threads"`
	SamplingStrategy        string            `json:"sampling_strategy"`
	Parameters              map[string]string `json:"parameters"`
	Runs                    []RunMetrics      `json:"runs"`
	AvgProcessingSeconds    float64           `json:"avg_processing_seconds"`
	AvgRealTimeFactor       float64           `json:"avg_real_time_factor"`
	WallSecondsPerAudioHour float64           `json:"wall_seconds_per_audio_hour"`
	// Concurrency/throughput fields (set when using multi-stream runs)
	Concurrency                     int     `json:"concurrency"`
	WallSecondsTotal                float64 `json:"wall_seconds_total"`
	TotalAudioHoursProcessed        float64 `json:"total_audio_hours_processed"`
	ThroughputAudioHoursPerWallHour float64 `json:"throughput_audio_hours_per_wall_hour"`
	PeakRSSMegabytes                float64 `json:"peak_rss_mb"`
	GPUVRAMTotalMB                  float64 `json:"gpu_vram_total_mb"`
	GPUVRAMUsedMB                   float64 `json:"gpu_vram_used_mb"`
	GPUVRAMFreeMB                   float64 `json:"gpu_vram_free_mb"`
	// Aggregated GPU metrics over the whole run (if monitored)
	GPUUtilAvgPercent   float64 `json:"gpu_util_avg_percent"`
	GPUUtilMaxPercent   int     `json:"gpu_util_max_percent"`
	GPUVRAMUsedAvgMB    float64 `json:"gpu_vram_used_avg_mb"`
	GPUVRAMUsedMaxMB    float64 `json:"gpu_vram_used_max_mb"`
	CPUModel            string  `json:"cpu_model"`
	CPUNumLogical       int     `json:"cpu_num_logical"`
	OS                  string  `json:"os"`
	Arch                string  `json:"arch"`
	MonthlyPriceUSD     float64 `json:"monthly_price_usd"`
	CostPerAudioHourUSD float64 `json:"cost_per_audio_hour_usd"`
}
