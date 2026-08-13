package mocksim

import (
	"os"
	"path/filepath"
	"testing"
	"time"
)

// The committed configs/mockbackend.yaml must actually load. Strict decoding
// means a missing yaml tag is a startup failure, not a silent default.
func TestRepoConfigLoads(t *testing.T) {
	path := filepath.Join("..", "..", "configs", "mockbackend.yaml")
	if _, err := os.Stat(path); err != nil {
		t.Skipf("config not present: %v", err)
	}

	cfg, err := LoadConfig(path)
	if err != nil {
		t.Fatalf("LoadConfig(%s): %v", path, err)
	}
	if cfg.PrefixCache.BlockSize != 16 {
		t.Errorf("BlockSize = %d, want 16", cfg.PrefixCache.BlockSize)
	}
	if cfg.Prefill.PerToken.Duration != 250*time.Microsecond {
		t.Errorf("Prefill.PerToken = %s, want 250us", cfg.Prefill.PerToken)
	}
}

func TestLoadConfigFaultFieldsMapFromYAML(t *testing.T) {
	path := filepath.Join(t.TempDir(), "mb.yaml")
	body := `
name: m
faults:
  error_rate: 0.25
  extra_latency_ms: 40
  hang: true
  slow_tokens: 2.5
`
	if err := os.WriteFile(path, []byte(body), 0o600); err != nil {
		t.Fatalf("write: %v", err)
	}

	cfg, err := LoadConfig(path)
	if err != nil {
		t.Fatalf("LoadConfig: %v", err)
	}
	if cfg.Faults.ErrorRate != 0.25 || cfg.Faults.ExtraLatencyMS != 40 ||
		!cfg.Faults.Hang || cfg.Faults.SlowTokens != 2.5 {
		t.Errorf("faults = %+v, want all fields populated from YAML", cfg.Faults)
	}
}

func TestLoadConfigRejectsUnknownField(t *testing.T) {
	path := filepath.Join(t.TempDir(), "mb.yaml")
	if err := os.WriteFile(path, []byte("naem: typo\n"), 0o600); err != nil {
		t.Fatalf("write: %v", err)
	}
	if _, err := LoadConfig(path); err == nil {
		t.Error("LoadConfig accepted an unknown field")
	}
}

func TestLoadConfigMissingFile(t *testing.T) {
	if _, err := LoadConfig(filepath.Join(t.TempDir(), "nope.yaml")); err == nil {
		t.Error("LoadConfig accepted a missing file")
	}
}
