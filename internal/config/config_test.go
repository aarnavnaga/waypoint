package config

import (
	"os"
	"path/filepath"
	"testing"
	"time"
)

func writeTemp(t *testing.T, body string) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "waypoint.yaml")
	if err := os.WriteFile(path, []byte(body), 0o600); err != nil {
		t.Fatalf("write temp config: %v", err)
	}
	return path
}

const minimalConfig = `
backends:
  - { name: mock-1, url: "http://localhost:9001", max_inflight_tokens: 32000 }
`

func TestLoadAppliesDefaults(t *testing.T) {
	cfg, err := Load(writeTemp(t, minimalConfig))
	if err != nil {
		t.Fatalf("Load: %v", err)
	}

	if cfg.Server.Port != 8080 {
		t.Errorf("Server.Port = %d, want 8080", cfg.Server.Port)
	}
	if cfg.Routing.Policy != PolicyRoundRobin {
		t.Errorf("Routing.Policy = %q, want %q", cfg.Routing.Policy, PolicyRoundRobin)
	}
	if cfg.Cache.ExactTTL.Duration != time.Hour {
		t.Errorf("Cache.ExactTTL = %s, want 1h", cfg.Cache.ExactTTL)
	}
	if len(cfg.Backends) != 1 || cfg.Backends[0].Name != "mock-1" {
		t.Fatalf("Backends = %+v, want one backend named mock-1", cfg.Backends)
	}
}

func TestLoadParsesDurations(t *testing.T) {
	cfg, err := Load(writeTemp(t, minimalConfig+`
resilience: { ttft_timeout: 5s, request_timeout: 2m }
cache: { exact_ttl: 30m }
`))
	if err != nil {
		t.Fatalf("Load: %v", err)
	}

	tests := []struct {
		name string
		got  time.Duration
		want time.Duration
	}{
		{"ttft_timeout", cfg.Resilience.TTFTTimeout.Duration, 5 * time.Second},
		{"request_timeout", cfg.Resilience.RequestTimeout.Duration, 2 * time.Minute},
		{"exact_ttl", cfg.Cache.ExactTTL.Duration, 30 * time.Minute},
	}
	for _, tc := range tests {
		if tc.got != tc.want {
			t.Errorf("%s = %s, want %s", tc.name, tc.got, tc.want)
		}
	}
}

func TestLoadRejectsBadDuration(t *testing.T) {
	_, err := Load(writeTemp(t, minimalConfig+"\nresilience: { ttft_timeout: \"5 bananas\" }\n"))
	if err == nil {
		t.Fatal("Load accepted an invalid duration")
	}
}

func TestLoadRejectsUnknownFields(t *testing.T) {
	_, err := Load(writeTemp(t, minimalConfig+"\nserver: { prot: 8080 }\n"))
	if err == nil {
		t.Fatal("Load accepted an unknown field; typos would be silently ignored")
	}
}

func TestEnvOverrides(t *testing.T) {
	t.Setenv("WAYPOINT_PORT", "9999")
	t.Setenv("WAYPOINT_ROUTING_POLICY", PolicyLeastLoaded)
	t.Setenv("WAYPOINT_REDIS_ADDR", "redis.internal:6380")
	t.Setenv("WAYPOINT_API_KEYS", " k1 , k2 ,")

	cfg, err := Load(writeTemp(t, minimalConfig))
	if err != nil {
		t.Fatalf("Load: %v", err)
	}

	if cfg.Server.Port != 9999 {
		t.Errorf("Server.Port = %d, want 9999", cfg.Server.Port)
	}
	if cfg.Routing.Policy != PolicyLeastLoaded {
		t.Errorf("Routing.Policy = %q, want %q", cfg.Routing.Policy, PolicyLeastLoaded)
	}
	if cfg.Redis.Addr != "redis.internal:6380" {
		t.Errorf("Redis.Addr = %q", cfg.Redis.Addr)
	}
	if len(cfg.Auth.APIKeys) != 2 || cfg.Auth.APIKeys[0] != "k1" || cfg.Auth.APIKeys[1] != "k2" {
		t.Errorf("Auth.APIKeys = %q, want [k1 k2]", cfg.Auth.APIKeys)
	}
}

func TestValidate(t *testing.T) {
	tests := []struct {
		name    string
		mutate  func(*Config)
		wantErr bool
	}{
		{"default is valid", func(*Config) {}, false},
		{"bad port", func(c *Config) { c.Server.Port = 0 }, true},
		{"unknown policy", func(c *Config) { c.Routing.Policy = "magic" }, true},
		{"no backends", func(c *Config) { c.Backends = nil }, true},
		{"duplicate backend names", func(c *Config) {
			c.Backends = append(c.Backends, Backend{Name: "mock-1", URL: "http://b:1"})
		}, true},
		{"relative backend url", func(c *Config) { c.Backends[0].URL = "mock-1:9001" }, true},
		{"load factor below one", func(c *Config) { c.Routing.LoadFactor = 0.9 }, true},
		{"retry budget over 100", func(c *Config) { c.Resilience.RetryBudgetPct = 101 }, true},
		{"zero attempts", func(c *Config) { c.Resilience.MaxAttempts = 0 }, true},
		{"breaker error rate above one", func(c *Config) { c.Resilience.Breaker.ErrorRate = 1.5 }, true},
		{"semantic threshold out of range", func(c *Config) {
			c.Cache.Semantic.Enabled = true
			c.Cache.Semantic.Threshold = 1.5
		}, true},
		{"semantic enabled without embedder", func(c *Config) {
			c.Cache.Semantic.Enabled = true
			c.Cache.Semantic.EmbedderURL = ""
		}, true},
		{"sampling above one", func(c *Config) { c.Telemetry.TraceSampling = 1.1 }, true},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			cfg := Default()
			cfg.Backends = []Backend{{Name: "mock-1", URL: "http://localhost:9001"}}
			tc.mutate(&cfg)

			err := cfg.Validate()
			if tc.wantErr && err == nil {
				t.Error("Validate() = nil, want error")
			}
			if !tc.wantErr && err != nil {
				t.Errorf("Validate() = %v, want nil", err)
			}
		})
	}
}

func TestLoadMissingFile(t *testing.T) {
	if _, err := Load(filepath.Join(t.TempDir(), "absent.yaml")); err == nil {
		t.Fatal("Load accepted a missing file")
	}
}
