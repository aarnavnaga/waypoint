package openai

import (
	"encoding/json"
	"strings"
	"testing"
)

func TestChatRequestPreservesUnknownFields(t *testing.T) {
	// A backend may support parameters Waypoint does not model. Proxying must
	// not silently drop them.
	raw := `{
		"model": "mock-7b",
		"messages": [{"role":"user","content":"hi"}],
		"temperature": 0,
		"guided_json": {"type":"object"},
		"repetition_penalty": 1.1
	}`

	var req ChatRequest
	if err := json.Unmarshal([]byte(raw), &req); err != nil {
		t.Fatalf("Unmarshal: %v", err)
	}

	if req.Model != "mock-7b" {
		t.Errorf("Model = %q", req.Model)
	}
	if len(req.Extra) != 2 {
		t.Fatalf("Extra = %v, want guided_json and repetition_penalty", req.Extra)
	}

	out, err := json.Marshal(req)
	if err != nil {
		t.Fatalf("Marshal: %v", err)
	}

	var round map[string]any
	if err := json.Unmarshal(out, &round); err != nil {
		t.Fatalf("round trip Unmarshal: %v", err)
	}
	if _, ok := round["guided_json"]; !ok {
		t.Error("guided_json was dropped in round trip")
	}
	if got, ok := round["repetition_penalty"].(float64); !ok || got != 1.1 {
		t.Errorf("repetition_penalty = %v, want 1.1", round["repetition_penalty"])
	}
	if round["model"] != "mock-7b" {
		t.Errorf("model = %v after round trip", round["model"])
	}
}

func TestChatRequestExtraCannotOverrideKnownFields(t *testing.T) {
	req := ChatRequest{
		Model:    "real-model",
		Messages: []Message{{Role: RoleUser, Content: "hi"}},
		Extra:    map[string]json.RawMessage{"model": json.RawMessage(`"spoofed"`)},
	}

	out, err := json.Marshal(req)
	if err != nil {
		t.Fatalf("Marshal: %v", err)
	}

	var round map[string]any
	if err := json.Unmarshal(out, &round); err != nil {
		t.Fatalf("Unmarshal: %v", err)
	}
	if round["model"] != "real-model" {
		t.Errorf("model = %v; Extra must not shadow a known field", round["model"])
	}
}

func TestChatRequestOmitsAbsentOptionals(t *testing.T) {
	out, err := json.Marshal(ChatRequest{
		Model:    "m",
		Messages: []Message{{Role: RoleUser, Content: "x"}},
	})
	if err != nil {
		t.Fatalf("Marshal: %v", err)
	}

	for _, field := range []string{"max_tokens", "temperature", "top_p", "seed", "stop"} {
		if strings.Contains(string(out), field) {
			t.Errorf("absent optional %q was serialized: %s", field, out)
		}
	}
}

func TestChatRequestValidate(t *testing.T) {
	valid := func() ChatRequest {
		return ChatRequest{Model: "m", Messages: []Message{{Role: RoleUser, Content: "hi"}}}
	}
	ptrInt := func(n int) *int { return &n }
	ptrFloat := func(f float64) *float64 { return &f }

	tests := []struct {
		name    string
		mutate  func(*ChatRequest)
		wantErr bool
	}{
		{"valid", func(*ChatRequest) {}, false},
		{"no model", func(r *ChatRequest) { r.Model = "" }, true},
		{"no messages", func(r *ChatRequest) { r.Messages = nil }, true},
		{"message without role", func(r *ChatRequest) { r.Messages[0].Role = "" }, true},
		{"zero max_tokens", func(r *ChatRequest) { r.MaxTokens = ptrInt(0) }, true},
		{"valid max_tokens", func(r *ChatRequest) { r.MaxTokens = ptrInt(10) }, false},
		{"temperature too high", func(r *ChatRequest) { r.Temperature = ptrFloat(2.5) }, true},
		{"temperature negative", func(r *ChatRequest) { r.Temperature = ptrFloat(-1) }, true},
		{"temperature zero is valid", func(r *ChatRequest) { r.Temperature = ptrFloat(0) }, false},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			req := valid()
			tc.mutate(&req)
			if err := req.Validate(); tc.wantErr != (err != nil) {
				t.Errorf("Validate() = %v, wantErr %v", err, tc.wantErr)
			}
		})
	}
}

func TestTemperatureOrDefault(t *testing.T) {
	var req ChatRequest
	if got := req.TemperatureOrDefault(); got != 1.0 {
		t.Errorf("default temperature = %v, want 1.0", got)
	}

	zero := 0.0
	req.Temperature = &zero
	if got := req.TemperatureOrDefault(); got != 0 {
		t.Errorf("explicit zero temperature = %v, want 0", got)
	}
}

func TestChunkFinishReasonIsNullable(t *testing.T) {
	// Clients distinguish "still streaming" (null) from "stopped", so the field
	// must serialize as null rather than be omitted.
	out, err := json.Marshal(ChatChunk{
		Choices: []ChunkChoice{{Delta: Delta{Content: "x"}}},
	})
	if err != nil {
		t.Fatalf("Marshal: %v", err)
	}
	if !strings.Contains(string(out), `"finish_reason":null`) {
		t.Errorf("expected explicit null finish_reason, got %s", out)
	}
}

func TestErrorEnvelopeShape(t *testing.T) {
	out, err := json.Marshal(NewError("over quota", ErrTypeRateLimit, "rate_limit_exceeded"))
	if err != nil {
		t.Fatalf("Marshal: %v", err)
	}

	var parsed struct {
		Error struct {
			Message string `json:"message"`
			Type    string `json:"type"`
			Code    string `json:"code"`
		} `json:"error"`
	}
	if err := json.Unmarshal(out, &parsed); err != nil {
		t.Fatalf("Unmarshal: %v", err)
	}
	if parsed.Error.Message != "over quota" || parsed.Error.Type != ErrTypeRateLimit {
		t.Errorf("unexpected envelope: %s", out)
	}
}

func TestChatRequestRejectsMalformedJSON(t *testing.T) {
	var req ChatRequest
	if err := json.Unmarshal([]byte(`{"model": 42}`), &req); err == nil {
		t.Error("accepted a non-string model")
	}
}

// FuzzParseChatRequest ensures decoding attacker-controlled bodies never panics.
func FuzzParseChatRequest(f *testing.F) {
	seeds := []string{
		`{"model":"m","messages":[{"role":"user","content":"hi"}]}`,
		`{"model":"m","messages":[],"stream":true}`,
		`{"model":"m","messages":[{"role":"user","content":"hi"}],"extra":{"a":[1,2,3]}}`,
		`{}`,
		`[]`,
		`null`,
		`{"messages":[{"role":"user","content":"` + strings.Repeat("x", 500) + `"}]}`,
	}
	for _, s := range seeds {
		f.Add(s)
	}

	f.Fuzz(func(t *testing.T, body string) {
		var req ChatRequest
		if err := json.Unmarshal([]byte(body), &req); err != nil {
			return
		}
		_ = req.Validate()
		_ = req.TemperatureOrDefault()

		// Anything that decodes must also re-encode without panicking.
		if _, err := json.Marshal(req); err != nil {
			t.Fatalf("decoded request failed to re-encode: %v", err)
		}
	})
}
