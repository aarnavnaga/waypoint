package mocksim

import (
	"io"
	"log/slog"
)

// discardLogger silences expected warn/error logs in tests.
func discardLogger() *slog.Logger {
	return slog.New(slog.NewTextHandler(io.Discard, &slog.HandlerOptions{Level: slog.LevelError + 1}))
}
