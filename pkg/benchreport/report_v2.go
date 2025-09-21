package benchreport

type ReportGPU struct {
	UseGPU      bool    `json:"use_gpu"`
	Device      int     `json:"device"`
	Name        string  `json:"name"`
	VRAMTotalMB float64 `json:"vram_total_mb"`
}

type ReportEnv struct {
	OS            string    `json:"os"`
	Arch          string    `json:"arch"`
	CPUModel      string    `json:"cpu_model"`
	CPUNumLogical int       `json:"cpu_num_logical"`
	GPU           ReportGPU `json:"gpu"`
}

type ReportParams struct {
	ModelPath          string            `json:"model_path"`
	ModelLanguage      string            `json:"model_language"`
	WAVPath            string            `json:"wav_path"`
	WAVSHA256          string            `json:"wav_sha256"`
	WAVDurationSeconds float64           `json:"wav_duration_seconds"`
	Threads            int               `json:"threads"`
	SamplingStrategy   string            `json:"sampling_strategy"`
	Parameters         map[string]string `json:"parameters"`
	WindowSeconds      int               `json:"window_seconds"`
	Concurrency        int               `json:"concurrency"`
	Repeats            int               `json:"repeats"`
	MonthlyPriceUSD    float64           `json:"monthly_price_usd"`
}

type ReportMetrics struct {
	AvgProcessingSeconds    float64 `json:"avg_processing_seconds"`
	AvgRealTimeFactor       float64 `json:"avg_real_time_factor"`
	WallSecondsPerAudioHour float64 `json:"wall_seconds_per_audio_hour"`
	// Peak process memory (RSS)
	PeakRSSMegabytes float64 `json:"peak_rss_mb"`
	// Concurrency/throughput metrics (optional)
	WallSecondsTotal                float64 `json:"wall_seconds_total"`
	TotalAudioHoursProcessed        float64 `json:"total_audio_hours_processed"`
	ThroughputAudioHoursPerWallHour float64 `json:"throughput_audio_hours_per_wall_hour"`
	// GPU peaks across the run (if monitored)
	GPUUtilMaxPercent int     `json:"gpu_util_max_percent"`
	GPUVRAMUsedMaxMB  float64 `json:"gpu_vram_used_max_mb"`
	// Cost
	CostPerAudioHourUSD float64 `json:"cost_per_audio_hour_usd"`
}

type ReportV2 struct {
	Version          string        `json:"version"`
	TimestampRFC3339 string        `json:"timestamp_rfc3339"`
	Label            string        `json:"label"`
	Env              ReportEnv     `json:"env"`
	Params           ReportParams  `json:"params"`
	Runs             []RunMetrics  `json:"runs"`
	Metrics          ReportMetrics `json:"metrics"`
}
