// Package openai defines the subset of the OpenAI chat-completions wire format
// that Waypoint speaks, both as a server (to clients) and a client (to backends).
package openai

import (
	"encoding/json"
	"fmt"
)

// Role values used in chat messages.
const (
	RoleSystem    = "system"
	RoleUser      = "user"
	RoleAssistant = "assistant"
	RoleTool      = "tool"
)

// Message is one turn in a conversation.
type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
	Name    string `json:"name,omitempty"`
}

// ChatRequest is a POST /v1/chat/completions body.
//
// Unknown fields are preserved in Extra so the gateway can proxy
// backend-specific parameters without needing to know them.
type ChatRequest struct {
	Model       string    `json:"model"`
	Messages    []Message `json:"messages"`
	Stream      bool      `json:"stream,omitempty"`
	MaxTokens   *int      `json:"max_tokens,omitempty"`
	Temperature *float64  `json:"temperature,omitempty"`
	TopP        *float64  `json:"top_p,omitempty"`
	Stop        []string  `json:"stop,omitempty"`
	Seed        *int      `json:"seed,omitempty"`
	User        string    `json:"user,omitempty"`

	// Extra holds any fields not named above, so proxying is lossless.
	Extra map[string]json.RawMessage `json:"-"`
}

// knownChatFields are the fields ChatRequest models explicitly.
var knownChatFields = map[string]struct{}{
	"model": {}, "messages": {}, "stream": {}, "max_tokens": {},
	"temperature": {}, "top_p": {}, "stop": {}, "seed": {}, "user": {},
}

// UnmarshalJSON decodes known fields and retains the rest in Extra.
func (r *ChatRequest) UnmarshalJSON(data []byte) error {
	type alias ChatRequest
	var base alias
	if err := json.Unmarshal(data, &base); err != nil {
		return err
	}

	var all map[string]json.RawMessage
	if err := json.Unmarshal(data, &all); err != nil {
		return err
	}
	for k := range knownChatFields {
		delete(all, k)
	}
	if len(all) > 0 {
		base.Extra = all
	}

	*r = ChatRequest(base)
	return nil
}

// MarshalJSON re-emits known fields plus anything retained in Extra.
func (r ChatRequest) MarshalJSON() ([]byte, error) {
	type alias ChatRequest
	data, err := json.Marshal(alias(r))
	if err != nil {
		return nil, err
	}
	if len(r.Extra) == 0 {
		return data, nil
	}

	var merged map[string]json.RawMessage
	if err := json.Unmarshal(data, &merged); err != nil {
		return nil, err
	}
	for k, v := range r.Extra {
		if _, reserved := knownChatFields[k]; reserved {
			continue
		}
		merged[k] = v
	}
	return json.Marshal(merged)
}

// Validate reports whether the request is well-formed enough to serve.
func (r *ChatRequest) Validate() error {
	if r.Model == "" {
		return fmt.Errorf("model is required")
	}
	if len(r.Messages) == 0 {
		return fmt.Errorf("messages must contain at least one message")
	}
	for i, m := range r.Messages {
		if m.Role == "" {
			return fmt.Errorf("messages[%d].role is required", i)
		}
	}
	if r.MaxTokens != nil && *r.MaxTokens < 1 {
		return fmt.Errorf("max_tokens must be >= 1")
	}
	if r.Temperature != nil && (*r.Temperature < 0 || *r.Temperature > 2) {
		return fmt.Errorf("temperature must be within [0,2]")
	}
	return nil
}

// TemperatureOrDefault returns the requested temperature, defaulting to 1.0.
func (r *ChatRequest) TemperatureOrDefault() float64 {
	if r.Temperature == nil {
		return 1.0
	}
	return *r.Temperature
}

// Object type values used in responses.
const (
	ObjectChatCompletion = "chat.completion"
	ObjectChatChunk      = "chat.completion.chunk"
	ObjectList           = "list"
	ObjectModel          = "model"
)

// Usage reports token accounting for a completion.
type Usage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

// Choice is one non-streaming completion alternative.
type Choice struct {
	Index        int     `json:"index"`
	Message      Message `json:"message"`
	FinishReason string  `json:"finish_reason"`
}

// ChatResponse is a non-streaming completion body.
type ChatResponse struct {
	ID      string   `json:"id"`
	Object  string   `json:"object"`
	Created int64    `json:"created"`
	Model   string   `json:"model"`
	Choices []Choice `json:"choices"`
	Usage   Usage    `json:"usage"`
	// SystemFingerprint identifies the serving backend; Waypoint sets it to the
	// backend name so responses are traceable to a server.
	SystemFingerprint string `json:"system_fingerprint,omitempty"`
}

// Delta is the incremental content of a streaming chunk.
type Delta struct {
	Role    string `json:"role,omitempty"`
	Content string `json:"content,omitempty"`
}

// ChunkChoice is one streaming alternative.
type ChunkChoice struct {
	Index        int     `json:"index"`
	Delta        Delta   `json:"delta"`
	FinishReason *string `json:"finish_reason"`
}

// ChatChunk is a single SSE `data:` payload in a streaming completion.
type ChatChunk struct {
	ID                string        `json:"id"`
	Object            string        `json:"object"`
	Created           int64         `json:"created"`
	Model             string        `json:"model"`
	Choices           []ChunkChoice `json:"choices"`
	Usage             *Usage        `json:"usage,omitempty"`
	SystemFingerprint string        `json:"system_fingerprint,omitempty"`
}

// Model describes an available model in GET /v1/models.
type Model struct {
	ID      string `json:"id"`
	Object  string `json:"object"`
	Created int64  `json:"created"`
	OwnedBy string `json:"owned_by"`
}

// ModelList is the GET /v1/models response.
type ModelList struct {
	Object string  `json:"object"`
	Data   []Model `json:"data"`
}

// Error types used in the OpenAI error envelope.
const (
	ErrTypeInvalidRequest = "invalid_request_error"
	ErrTypeAuthentication = "authentication_error"
	ErrTypeRateLimit      = "rate_limit_error"
	ErrTypeServer         = "server_error"
	ErrTypeOverloaded     = "overloaded_error"
)

// APIError is the body of an OpenAI-shaped error response.
type APIError struct {
	Message string `json:"message"`
	Type    string `json:"type"`
	Param   string `json:"param,omitempty"`
	Code    string `json:"code,omitempty"`
}

// ErrorEnvelope wraps APIError as `{"error": {...}}`.
type ErrorEnvelope struct {
	Error APIError `json:"error"`
}

// NewError builds an error envelope.
func NewError(msg, typ, code string) ErrorEnvelope {
	return ErrorEnvelope{Error: APIError{Message: msg, Type: typ, Code: code}}
}
