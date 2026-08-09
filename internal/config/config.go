// Package config loads and validates Waypoint's YAML configuration.
package config

import (
	"errors"
	"fmt"
	"net/url"
	"os"
	"strconv"
	"strings"
	"time"

	"gopkg.in/yaml.v3"
)

// Duration wraps time.Duration so it can be expressed in YAML as "5s" / "1h".
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

// Policy names understood by internal/router.
const (
	PolicyRoundRobin              = "round_robin"
	PolicyLeastLoaded             = "least_loaded"
	PolicyPrefixAffinity          = "prefix_affinity"
	PolicyPrefixAffinityLoadAware = "prefix_affinity_load_aware"
)

// ValidPolicies lists every routing policy the gateway can be configured with.
var ValidPolicies = []string{
	PolicyRoundRobin,
	PolicyLeastLoaded,
	PolicyPrefixAffinity,
	PolicyPrefixAffinityLoadAware,
}

// Config is the root configuration document.
type Config struct {
	Server     Server     `yaml:"server"`
	Routing    Routing    `yaml:"routing"`
	Backends   []Backend  `yaml:"backends"`
	Resilience Resilience `yaml:"resilience"`
	RateLimit  RateLimit  `yaml:"ratelimit"`
	Cache      Cache      `yaml:"cache"`
	Redis      Redis      `yaml:"redis"`
	Telemetry  Telemetry  `yaml:"telemetry"`
	Auth       Auth       `yaml:"auth"`
}

// Server holds listener and admission-control settings.
type Server struct {
	Port int `yaml:"port"`
	// MaxQueue bounds the admission queue; requests beyond it are shed with 503.
	MaxQueue int `yaml:"max_queue"`
	// MaxConcurrent bounds in-flight dispatches; 0 means unbounded.
	MaxConcurrent     int      `yaml:"max_concurrent"`
	ShutdownGrace     Duration `yaml:"shutdown_grace"`
	ReadHeaderTimeout Duration `yaml:"read_header_timeout"`
}

// Routing selects and tunes the backend-selection policy.
type Routing struct {
	Policy string `yaml:"policy"`
	// PrefixTokens is how many leading prompt tokens form the affinity key.
	PrefixTokens int `yaml:"prefix_tokens"`
	// LoadFactor is the bounded-load multiplier (c) for prefix affinity.
	LoadFactor float64 `yaml:"load_factor"`
	// VirtualNodes is the replication factor on the hash ring.
	VirtualNodes int `yaml:"virtual_nodes"`
}

// Backend describes one upstream model server.
type Backend struct {
	Name              string `yaml:"name"`
	URL               string `yaml:"url"`
	MaxInflightTokens int    `yaml:"max_inflight_tokens"`
	// Models optionally pins the model list; empty means discover via /v1/models.
	Models []string `yaml:"models"`
}

// Resilience configures breakers, retries and timeouts.
type Resilience struct {
	RetryBudgetPct float64  `yaml:"retry_budget_pct"`
	MaxAttempts    int      `yaml:"max_attempts"`
	ConnectTimeout Duration `yaml:"connect_timeout"`
	TTFTTimeout    Duration `yaml:"ttft_timeout"`
	RequestTimeout Duration `yaml:"request_timeout"`
	Backoff        Backoff  `yaml:"backoff"`
	Breaker        Breaker  `yaml:"breaker"`
	HealthCheck    Health   `yaml:"health_check"`
}

// Backoff tunes exponential backoff with full jitter.
type Backoff struct {
	Base Duration `yaml:"base"`
	Max  Duration `yaml:"max"`
}

// Breaker tunes the per-backend circuit breaker.
type Breaker struct {
	Window           Duration `yaml:"window"`
	Buckets          int      `yaml:"buckets"`
	MinRequests      int      `yaml:"min_requests"`
	ErrorRate        float64  `yaml:"error_rate"`
	TimeoutRate      float64  `yaml:"timeout_rate"`
	OpenDuration     Duration `yaml:"open_duration"`
	HalfOpenRequests int      `yaml:"half_open_requests"`
}

// Health tunes active probing of backends.
type Health struct {
	Interval           Duration `yaml:"interval"`
	Timeout            Duration `yaml:"timeout"`
	Path               string   `yaml:"path"`
	HealthyThreshold   int      `yaml:"healthy_threshold"`
	UnhealthyThreshold int      `yaml:"unhealthy_threshold"`
}

// RateLimit configures the Redis-backed token buckets.
type RateLimit struct {
	Enabled  bool `yaml:"enabled"`
	RPM      int  `yaml:"rpm"`
	TPM      int  `yaml:"tpm"`
	FailOpen bool `yaml:"fail_open"`
}

// Cache configures the exact-match and semantic response caches.
type Cache struct {
	Enabled  bool     `yaml:"enabled"`
	ExactTTL Duration `yaml:"exact_ttl"`
	Semantic Semantic `yaml:"semantic"`
}

// Semantic configures the embedding-based cache.
type Semantic struct {
	Enabled      bool     `yaml:"enabled"`
	Threshold    float64  `yaml:"threshold"`
	TTL          Duration `yaml:"ttl"`
	EmbedderURL  string   `yaml:"embedder_url"`
	EmbedTimeout Duration `yaml:"embed_timeout"`
	Dimensions   int      `yaml:"dimensions"`
	IndexName    string   `yaml:"index_name"`
	TopK         int      `yaml:"top_k"`
}

// Redis points at the shared Redis Stack instance.
type Redis struct {
	Addr      string   `yaml:"addr"`
	Password  string   `yaml:"password"`
	DB        int      `yaml:"db"`
	Timeout   Duration `yaml:"timeout"`
	KeyPrefix string   `yaml:"key_prefix"`
}

// Telemetry configures metrics, tracing and logging.
type Telemetry struct {
	ServiceName   string  `yaml:"service_name"`
	OTLPEndpoint  string  `yaml:"otlp_endpoint"`
	TraceSampling float64 `yaml:"trace_sampling"`
	LogLevel      string  `yaml:"log_level"`
}

// Auth lists accepted API keys. An empty list disables authentication.
type Auth struct {
	APIKeys []string `yaml:"api_keys"`
}

// Default returns a configuration with production-sane defaults applied.
// Load starts from these, so YAML files only need to specify overrides.
func Default() Config {
	return Config{
		Server: Server{
			Port:              8080,
			MaxQueue:          512,
			MaxConcurrent:     0,
			ShutdownGrace:     Duration{20 * time.Second},
			ReadHeaderTimeout: Duration{10 * time.Second},
		},
		Routing: Routing{
			Policy:       PolicyRoundRobin,
			PrefixTokens: 256,
			LoadFactor:   1.25,
			VirtualNodes: 150,
		},
		Resilience: Resilience{
			RetryBudgetPct: 10,
			MaxAttempts:    3,
			ConnectTimeout: Duration{2 * time.Second},
			TTFTTimeout:    Duration{5 * time.Second},
			RequestTimeout: Duration{120 * time.Second},
			Backoff: Backoff{
				Base: Duration{20 * time.Millisecond},
				Max:  Duration{2 * time.Second},
			},
			Breaker: Breaker{
				Window:           Duration{10 * time.Second},
				Buckets:          10,
				MinRequests:      20,
				ErrorRate:        0.5,
				TimeoutRate:      0.5,
				OpenDuration:     Duration{5 * time.Second},
				HalfOpenRequests: 3,
			},
			HealthCheck: Health{
				Interval:           Duration{2 * time.Second},
				Timeout:            Duration{1 * time.Second},
				Path:               "/healthz",
				HealthyThreshold:   2,
				UnhealthyThreshold: 3,
			},
		},
		RateLimit: RateLimit{
			Enabled:  true,
			RPM:      600,
			TPM:      200000,
			FailOpen: true,
		},
		Cache: Cache{
			Enabled:  true,
			ExactTTL: Duration{time.Hour},
			Semantic: Semantic{
				Enabled:      false,
				Threshold:    0.92,
				TTL:          Duration{time.Hour},
				EmbedderURL:  "http://embeddings:8000",
				EmbedTimeout: Duration{500 * time.Millisecond},
				Dimensions:   384,
				IndexName:    "waypoint_semantic",
				TopK:         3,
			},
		},
		Redis: Redis{
			Addr:      "redis:6379",
			DB:        0,
			Timeout:   Duration{200 * time.Millisecond},
			KeyPrefix: "waypoint",
		},
		Telemetry: Telemetry{
			ServiceName:   "waypoint-gateway",
			TraceSampling: 1.0,
			LogLevel:      "info",
		},
	}
}

// Load reads YAML from path, applies environment overrides and validates.
func Load(path string) (Config, error) {
	cfg := Default()

	raw, err := os.ReadFile(path)
	if err != nil {
		return Config{}, fmt.Errorf("read config: %w", err)
	}
	dec := yaml.NewDecoder(strings.NewReader(string(raw)))
	dec.KnownFields(true)
	if err := dec.Decode(&cfg); err != nil {
		return Config{}, fmt.Errorf("parse config %s: %w", path, err)
	}

	if err := cfg.applyEnv(); err != nil {
		return Config{}, err
	}
	if err := cfg.Validate(); err != nil {
		return Config{}, err
	}
	return cfg, nil
}

// applyEnv applies WAYPOINT_* overrides. Only operationally useful knobs are
// overridable; everything else stays in YAML so runs stay reproducible.
func (c *Config) applyEnv() error {
	if v := os.Getenv("WAYPOINT_PORT"); v != "" {
		n, err := strconv.Atoi(v)
		if err != nil {
			return fmt.Errorf("WAYPOINT_PORT: %w", err)
		}
		c.Server.Port = n
	}
	if v := os.Getenv("WAYPOINT_ROUTING_POLICY"); v != "" {
		c.Routing.Policy = v
	}
	if v := os.Getenv("WAYPOINT_REDIS_ADDR"); v != "" {
		c.Redis.Addr = v
	}
	if v := os.Getenv("WAYPOINT_REDIS_PASSWORD"); v != "" {
		c.Redis.Password = v
	}
	if v := os.Getenv("WAYPOINT_OTLP_ENDPOINT"); v != "" {
		c.Telemetry.OTLPEndpoint = v
	}
	if v := os.Getenv("WAYPOINT_LOG_LEVEL"); v != "" {
		c.Telemetry.LogLevel = v
	}
	if v := os.Getenv("WAYPOINT_EMBEDDER_URL"); v != "" {
		c.Cache.Semantic.EmbedderURL = v
	}
	if v := os.Getenv("WAYPOINT_SEMANTIC_CACHE"); v != "" {
		b, err := strconv.ParseBool(v)
		if err != nil {
			return fmt.Errorf("WAYPOINT_SEMANTIC_CACHE: %w", err)
		}
		c.Cache.Semantic.Enabled = b
	}
	if v := os.Getenv("WAYPOINT_API_KEYS"); v != "" {
		c.Auth.APIKeys = splitAndTrim(v)
	}
	return nil
}

func splitAndTrim(v string) []string {
	parts := strings.Split(v, ",")
	out := make([]string, 0, len(parts))
	for _, p := range parts {
		if trimmed := strings.TrimSpace(p); trimmed != "" {
			out = append(out, trimmed)
		}
	}
	return out
}

// Validate reports configuration that would fail at runtime.
func (c *Config) Validate() error {
	var errs []error

	if c.Server.Port < 1 || c.Server.Port > 65535 {
		errs = append(errs, fmt.Errorf("server.port %d out of range", c.Server.Port))
	}
	if c.Server.MaxQueue < 1 {
		errs = append(errs, errors.New("server.max_queue must be >= 1"))
	}
	if !isValidPolicy(c.Routing.Policy) {
		errs = append(errs, fmt.Errorf("routing.policy %q must be one of %s",
			c.Routing.Policy, strings.Join(ValidPolicies, ", ")))
	}
	if c.Routing.PrefixTokens < 1 {
		errs = append(errs, errors.New("routing.prefix_tokens must be >= 1"))
	}
	if c.Routing.LoadFactor < 1 {
		errs = append(errs, errors.New("routing.load_factor must be >= 1.0"))
	}
	if c.Routing.VirtualNodes < 1 {
		errs = append(errs, errors.New("routing.virtual_nodes must be >= 1"))
	}
	if len(c.Backends) == 0 {
		errs = append(errs, errors.New("at least one backend is required"))
	}

	seen := make(map[string]struct{}, len(c.Backends))
	for i, b := range c.Backends {
		if b.Name == "" {
			errs = append(errs, fmt.Errorf("backends[%d].name is required", i))
		}
		if _, dup := seen[b.Name]; dup {
			errs = append(errs, fmt.Errorf("duplicate backend name %q", b.Name))
		}
		seen[b.Name] = struct{}{}

		u, err := url.Parse(b.URL)
		if err != nil || u.Scheme == "" || u.Host == "" {
			errs = append(errs, fmt.Errorf("backends[%d].url %q is not an absolute URL", i, b.URL))
		}
		if b.MaxInflightTokens < 0 {
			errs = append(errs, fmt.Errorf("backends[%d].max_inflight_tokens must be >= 0", i))
		}
	}

	if c.Resilience.MaxAttempts < 1 {
		errs = append(errs, errors.New("resilience.max_attempts must be >= 1"))
	}
	if c.Resilience.RetryBudgetPct < 0 || c.Resilience.RetryBudgetPct > 100 {
		errs = append(errs, errors.New("resilience.retry_budget_pct must be within [0,100]"))
	}
	if c.Resilience.Breaker.Buckets < 1 {
		errs = append(errs, errors.New("resilience.breaker.buckets must be >= 1"))
	}
	if r := c.Resilience.Breaker.ErrorRate; r <= 0 || r > 1 {
		errs = append(errs, errors.New("resilience.breaker.error_rate must be within (0,1]"))
	}
	if r := c.Resilience.Breaker.TimeoutRate; r <= 0 || r > 1 {
		errs = append(errs, errors.New("resilience.breaker.timeout_rate must be within (0,1]"))
	}
	if c.RateLimit.Enabled {
		if c.RateLimit.RPM < 1 {
			errs = append(errs, errors.New("ratelimit.rpm must be >= 1 when enabled"))
		}
		if c.RateLimit.TPM < 1 {
			errs = append(errs, errors.New("ratelimit.tpm must be >= 1 when enabled"))
		}
	}
	if c.Cache.Semantic.Enabled {
		if t := c.Cache.Semantic.Threshold; t <= 0 || t > 1 {
			errs = append(errs, errors.New("cache.semantic.threshold must be within (0,1]"))
		}
		if c.Cache.Semantic.Dimensions < 1 {
			errs = append(errs, errors.New("cache.semantic.dimensions must be >= 1"))
		}
		if c.Cache.Semantic.TopK < 1 {
			errs = append(errs, errors.New("cache.semantic.top_k must be >= 1"))
		}
		if c.Cache.Semantic.EmbedderURL == "" {
			errs = append(errs, errors.New("cache.semantic.embedder_url is required when semantic cache is enabled"))
		}
	}
	if s := c.Telemetry.TraceSampling; s < 0 || s > 1 {
		errs = append(errs, errors.New("telemetry.trace_sampling must be within [0,1]"))
	}

	return errors.Join(errs...)
}

func isValidPolicy(p string) bool {
	for _, valid := range ValidPolicies {
		if p == valid {
			return true
		}
	}
	return false
}
