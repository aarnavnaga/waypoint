// Command mockbackend simulates an LLM inference server with prefill/decode
// costs, a prefix KV cache and continuous batching. Fully implemented in M1.
package main

import (
	"fmt"
	"log/slog"
	"net/http"
	"os"
)

var version = "dev"

func main() {
	mux := http.NewServeMux()
	mux.HandleFunc("GET /healthz", func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte("ok\n"))
	})

	addr := ":9001"
	if v := os.Getenv("MOCKBACKEND_ADDR"); v != "" {
		addr = v
	}
	slog.Info("mock backend listening", "addr", addr, "version", version)

	srv := &http.Server{Addr: addr, Handler: mux, ReadHeaderTimeout: 0}
	if err := srv.ListenAndServe(); err != nil {
		slog.Error("mock backend exited", "error", err)
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
