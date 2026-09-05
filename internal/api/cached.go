package api

import (
	"encoding/json"
	"fmt"
	"net/http"
	"time"

	"github.com/aarnavnaga/waypoint/internal/cache"
	"github.com/aarnavnaga/waypoint/internal/openai"
	"github.com/aarnavnaga/waypoint/internal/sse"
)

// serveFromCache writes a cached entry to the client in the format the request
// asked for.
//
// The stored chunk boundaries are replayed rather than emitting the whole answer
// as one chunk, so a cached streaming response is structurally identical to a
// live one. A client that renders tokens incrementally should not be able to
// tell the difference apart from it arriving faster.
func (s *Server) serveFromCache(w http.ResponseWriter, req *openai.ChatRequest, entry *cache.Entry) {
	w.Header().Set("X-Waypoint-Cache", entry.Source)
	if entry.Backend != "" {
		w.Header().Set("X-Waypoint-Backend", entry.Backend)
	}
	if entry.Source == cache.SourceSemantic {
		w.Header().Set("X-Waypoint-Cache-Similarity", fmt.Sprintf("%.4f", entry.Similarity))
	}

	if !req.Stream {
		writeJSON(w, http.StatusOK, openai.ChatResponse{
			ID:                "chatcmpl-cached",
			Object:            "chat.completion",
			Created:           time.Now().Unix(),
			Model:             req.Model,
			SystemFingerprint: entry.Backend,
			Choices: []openai.Choice{{
				Index:        0,
				Message:      openai.Message{Role: openai.RoleAssistant, Content: entry.Content},
				FinishReason: "stop",
			}},
			Usage: entry.Usage(),
		})
		return
	}

	flusher, ok := w.(http.Flusher)
	if !ok {
		writeAPIError(w, http.StatusInternalServerError,
			"streaming unsupported", openai.ErrTypeServer, "")
		return
	}

	writeStreamHeaders(w, entry.Backend)
	w.Header().Set("X-Waypoint-Cache", entry.Source)

	enc := sse.NewEncoder(&flushWriter{w: w, f: flusher})
	id := "chatcmpl-cached"
	created := time.Now().Unix()

	roleChunk := openai.ChatChunk{
		ID: id, Object: "chat.completion.chunk", Created: created, Model: req.Model,
		SystemFingerprint: entry.Backend,
		Choices: []openai.ChunkChoice{{
			Index: 0, Delta: openai.Delta{Role: openai.RoleAssistant},
		}},
	}
	if err := writeChunk(enc, roleChunk); err != nil {
		return
	}

	chunks := entry.Chunks
	if len(chunks) == 0 {
		// An entry stored without chunk boundaries still replays correctly,
		// just as a single chunk.
		chunks = []string{entry.Content}
	}

	for _, fragment := range chunks {
		chunk := openai.ChatChunk{
			ID: id, Object: "chat.completion.chunk", Created: created, Model: req.Model,
			SystemFingerprint: entry.Backend,
			Choices: []openai.ChunkChoice{{
				Index: 0, Delta: openai.Delta{Content: fragment},
			}},
		}
		if err := writeChunk(enc, chunk); err != nil {
			return
		}
	}

	stop := "stop"
	usage := entry.Usage()
	finalChunk := openai.ChatChunk{
		ID: id, Object: "chat.completion.chunk", Created: created, Model: req.Model,
		SystemFingerprint: entry.Backend,
		Choices: []openai.ChunkChoice{{
			Index: 0, Delta: openai.Delta{}, FinishReason: &stop,
		}},
		Usage: &usage,
	}
	if err := writeChunk(enc, finalChunk); err != nil {
		return
	}
	_ = enc.WriteDone()
}

func writeChunk(enc *sse.Encoder, chunk openai.ChatChunk) error {
	payload, err := json.Marshal(chunk)
	if err != nil {
		return err
	}
	return enc.WriteData(string(payload))
}
