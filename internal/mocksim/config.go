package mocksim

import (
	"errors"
	"fmt"
	"os"
	"strings"
	"time"

	"gopkg.in/yaml.v3"
)

// Duration mirrors config.Duration so mock backend YAML can say "250us".
type Duration struct {
	time.Duration
}

// UnmarshalYAML decodes a Go duration string.
func (d *Duration) UnmarshalYAML(value *yaml.Node) error {
	var s string
	if err := value.Decode(&s); err != nil {
		return fmt.Errorf("duration must be a string: %w", err)
	}
	parsed, err := time.ParseDuration(s)
	if err != nil {
		return fmt.Errorf("invalid duration %q: %w", s, err)
	}
	d.Duration = parsed
	return nil
}

// MarshalYAML renders the duration back to its string form.
func (d Duration) MarshalYAML() (any, error) { return d.String(), nil }

// Config is the mock backend's YAML document. Every simulation parameter lives
// here so a benchmark run is reproducible from a committed file.
type Config struct {
	Name   string   `yaml:"name"`
	Addr   string   `yaml:"addr"`
	Seed   int64    `yaml:"seed"`
	Models []string `yaml:"models"`

	Prefill     Prefill    `yaml:"prefill"`
	Decode      Decode     `yaml:"decode"`
	Batching    Batching   `yaml:"batching"`
	PrefixCache PrefixConf `yaml:"prefix_cache"`
	Output      Output     `yaml:"output"`
	Faults      FaultState `yaml:"faults"`
}

// Prefill models the cost of processing prompt tokens before the first output
// token is produced.
type Prefill struct {
	// PerToken is the cost of each *uncached* prompt token.
	PerToken Duration `yaml:"per_token"`
	// Base is fixed per-request overhead (scheduling, sampling setup).
	Base Duration `yaml:"base"`
}

// Decode models per-output-token cost and how it degrades with batch size.
type Decode struct {
	// TokensPerSecond is single-sequence decode throughput at batch size 1.
	TokensPerSecond float64 `yaml:"tokens_per_second"`
	// Alpha scales batch-pressure slowdown.
	Alpha float64 `yaml:"alpha"`
	// Beta is the exponent on batch pressure.
	Beta float64 `yaml:"beta"`
}

// Interval converts TokensPerSecond into a per-token duration.
func (d Decode) Interval() time.Duration {
	if d.TokensPerSecond <= 0 {
		return 0
	}
	return time.Duration(float64(time.Second) / d.TokensPerSecond)
}

// Batching bounds concurrent decoding.
type Batching struct {
	MaxBatch int `yaml:"max_batch"`
	MaxQueue int `yaml:"max_queue"`
}

// PrefixConf sizes the simulated KV prefix cache.
type PrefixConf struct {
	BlockSize      int `yaml:"block_size"`
	CapacityBlocks int `yaml:"capacity_blocks"`
}

// Output bounds generated completion length.
type Output struct {
	MinTokens int `yaml:"min_tokens"`
	MaxTokens int `yaml:"max_tokens"`
}

// DefaultConfig returns a configuration tuned so that routing effects are
// visible on a laptop: prefill dominates TTFT for long prompts, and batch
// pressure is strong enough that load imbalance shows up in p99.
func DefaultConfig() Config {
	return Config{
		Name:   "mock",
		Addr:   ":9001",
		Seed:   1,
		Models: []string{"mock-7b"},
		Prefill: Prefill{
			PerToken: Duration{250 * time.Microsecond},
			Base:     Duration{5 * time.Millisecond},
		},
		Decode: Decode{
			TokensPerSecond: 120,
			Alpha:           0.08,
			Beta:            1.0,
		},
		Batching: Batching{
			MaxBatch: 8,
			MaxQueue: 256,
		},
		PrefixCache: PrefixConf{
			BlockSize:      16,
			CapacityBlocks: 2048,
		},
		Output: Output{
			MinTokens: 48,
			MaxTokens: 256,
		},
	}
}

// LoadConfig reads YAML over DefaultConfig and validates the result.
func LoadConfig(path string) (Config, error) {
	cfg := DefaultConfig()

	raw, err := os.ReadFile(path)
	if err != nil {
		return Config{}, fmt.Errorf("read mock backend config: %w", err)
	}
	dec := yaml.NewDecoder(strings.NewReader(string(raw)))
	dec.KnownFields(true)
	if err := dec.Decode(&cfg); err != nil {
		return Config{}, fmt.Errorf("parse mock backend config %s: %w", path, err)
	}

	if err := cfg.Validate(); err != nil {
		return Config{}, err
	}
	return cfg, nil
}

// Validate reports configuration that would make the simulation meaningless.
func (c *Config) Validate() error {
	var errs []error

	if c.Name == "" {
		errs = append(errs, errors.New("name is required"))
	}
	if c.Addr == "" {
		errs = append(errs, errors.New("addr is required"))
	}
	if len(c.Models) == 0 {
		errs = append(errs, errors.New("at least one model is required"))
	}
	if c.Decode.TokensPerSecond <= 0 {
		errs = append(errs, errors.New("decode.tokens_per_second must be > 0"))
	}
	if c.Decode.Alpha < 0 {
		errs = append(errs, errors.New("decode.alpha must be >= 0"))
	}
	if c.Batching.MaxBatch < 1 {
		errs = append(errs, errors.New("batching.max_batch must be >= 1"))
	}
	if c.PrefixCache.BlockSize < 1 {
		errs = append(errs, errors.New("prefix_cache.block_size must be >= 1"))
	}
	if c.PrefixCache.CapacityBlocks < 0 {
		errs = append(errs, errors.New("prefix_cache.capacity_blocks must be >= 0"))
	}
	if c.Output.MinTokens < 1 {
		errs = append(errs, errors.New("output.min_tokens must be >= 1"))
	}
	if c.Output.MaxTokens < c.Output.MinTokens {
		errs = append(errs, errors.New("output.max_tokens must be >= output.min_tokens"))
	}
	if c.Prefill.PerToken.Duration < 0 {
		errs = append(errs, errors.New("prefill.per_token must be >= 0"))
	}

	return errors.Join(errs...)
}
