package config

import (
	"os"

	"gopkg.in/yaml.v3"
)

type Config struct {
	Server struct {
		Address string `yaml:"address"`
	} `yaml:"server"`

	Model struct {
		Path         string `yaml:"path"`
		SampleRateHz int    `yaml:"sample_rate_hz"`
	} `yaml:"model"`

	Transcribe struct {
		MaxConcurrency int `yaml:"max_concurrency"`
	} `yaml:"transcribe"`

	Health struct {
		Enabled         bool    `yaml:"enabled"`
		LoadThreshold   float64 `yaml:"load_threshold"`
	} `yaml:"health"`
}

func Load(path string) (Config, error) {
	var c Config
	b, err := os.ReadFile(path)
	if err != nil {
		return c, err
	}

	if err := yaml.Unmarshal(b, &c); err != nil {
		return c, err
	}

	return c, nil
}
