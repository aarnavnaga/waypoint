package sse

import (
	"bytes"
	"errors"
	"io"
	"strings"
	"testing"
)

func decodeAll(t *testing.T, input string) []Event {
	t.Helper()
	dec := NewDecoder(strings.NewReader(input))

	var events []Event
	for {
		ev, err := dec.Next()
		if errors.Is(err, io.EOF) {
			return events
		}
		if err != nil {
			t.Fatalf("Next: %v", err)
		}
		events = append(events, ev)
	}
}

func TestDecodeSingleEvent(t *testing.T) {
	events := decodeAll(t, "data: hello\n\n")
	if len(events) != 1 {
		t.Fatalf("got %d events, want 1", len(events))
	}
	if events[0].Data != "hello" {
		t.Errorf("Data = %q, want %q", events[0].Data, "hello")
	}
}

func TestDecodeFields(t *testing.T) {
	events := decodeAll(t, "event: token\nid: 42\nretry: 1000\ndata: payload\n\n")
	if len(events) != 1 {
		t.Fatalf("got %d events, want 1", len(events))
	}

	ev := events[0]
	if ev.Name != "token" {
		t.Errorf("Name = %q", ev.Name)
	}
	if ev.ID != "42" {
		t.Errorf("ID = %q", ev.ID)
	}
	if ev.Retry != "1000" {
		t.Errorf("Retry = %q", ev.Retry)
	}
	if ev.Data != "payload" {
		t.Errorf("Data = %q", ev.Data)
	}
}

func TestDecodeMultilineDataIsJoined(t *testing.T) {
	events := decodeAll(t, "data: line one\ndata: line two\n\n")
	if len(events) != 1 {
		t.Fatalf("got %d events, want 1", len(events))
	}
	if want := "line one\nline two"; events[0].Data != want {
		t.Errorf("Data = %q, want %q", events[0].Data, want)
	}
}

func TestDecodeStripsExactlyOneLeadingSpace(t *testing.T) {
	tests := []struct {
		in   string
		want string
	}{
		{"data: hello\n\n", "hello"},
		{"data:hello\n\n", "hello"},
		{"data:  hello\n\n", " hello"}, // only one space is stripped
	}
	for _, tc := range tests {
		events := decodeAll(t, tc.in)
		if len(events) != 1 || events[0].Data != tc.want {
			t.Errorf("decode(%q) Data = %q, want %q", tc.in, events[0].Data, tc.want)
		}
	}
}

func TestDecodeMultipleEvents(t *testing.T) {
	events := decodeAll(t, "data: one\n\ndata: two\n\ndata: three\n\n")
	if len(events) != 3 {
		t.Fatalf("got %d events, want 3", len(events))
	}
	for i, want := range []string{"one", "two", "three"} {
		if events[i].Data != want {
			t.Errorf("event %d Data = %q, want %q", i, events[i].Data, want)
		}
	}
}

func TestDecodeComments(t *testing.T) {
	events := decodeAll(t, ": keepalive\ndata: real\n\n")
	if len(events) != 1 {
		t.Fatalf("got %d events, want 1", len(events))
	}
	if events[0].Data != "real" {
		t.Errorf("Data = %q", events[0].Data)
	}
	if !strings.Contains(events[0].Comment, "keepalive") {
		t.Errorf("Comment = %q, want it to contain keepalive", events[0].Comment)
	}
}

func TestDecodeCommentOnlyProducesNoEvent(t *testing.T) {
	if events := decodeAll(t, ": ping\n\n"); len(events) != 0 {
		t.Errorf("comment-only stream produced %d events, want 0", len(events))
	}
}

func TestDecodeCRLF(t *testing.T) {
	events := decodeAll(t, "data: hello\r\n\r\n")
	if len(events) != 1 || events[0].Data != "hello" {
		t.Fatalf("CRLF stream decoded as %+v", events)
	}
}

// Upstreams do not reliably send a trailing blank line; a pending event must
// still be delivered rather than silently dropped.
func TestDecodeUnterminatedFinalEvent(t *testing.T) {
	events := decodeAll(t, "data: one\n\ndata: truncated\n")
	if len(events) != 2 {
		t.Fatalf("got %d events, want 2", len(events))
	}
	if events[1].Data != "truncated" {
		t.Errorf("final Data = %q, want %q", events[1].Data, "truncated")
	}
}

func TestDecodeIgnoresUnknownFields(t *testing.T) {
	events := decodeAll(t, "unknown: whatever\ndata: kept\n\n")
	if len(events) != 1 || events[0].Data != "kept" {
		t.Fatalf("unexpected decode: %+v", events)
	}
}

func TestDecodeFieldWithoutColon(t *testing.T) {
	// A bare "data" line is a field with an empty value.
	events := decodeAll(t, "data\n\n")
	if len(events) != 1 {
		t.Fatalf("got %d events, want 1", len(events))
	}
	if events[0].Data != "" {
		t.Errorf("Data = %q, want empty", events[0].Data)
	}
}

func TestDecodeExtraBlankLines(t *testing.T) {
	events := decodeAll(t, "\n\n\ndata: hello\n\n\n\n")
	if len(events) != 1 || events[0].Data != "hello" {
		t.Fatalf("unexpected decode: %+v", events)
	}
}

func TestIsDone(t *testing.T) {
	events := decodeAll(t, "data: [DONE]\n\n")
	if len(events) != 1 || !events[0].IsDone() {
		t.Fatal("[DONE] sentinel not recognized")
	}
	if (Event{Data: "not done"}).IsDone() {
		t.Error("non-sentinel reported as done")
	}
}

func TestDecodeRejectsOversizedEvent(t *testing.T) {
	dec := NewDecoder(strings.NewReader("data: " + strings.Repeat("x", 5000) + "\n\n"))
	dec.SetMaxEventBytes(1000)

	if _, err := dec.Next(); !errors.Is(err, ErrEventTooLarge) {
		t.Errorf("Next error = %v, want ErrEventTooLarge", err)
	}
}

func TestDecoderRecoversAfterOversizedEvent(t *testing.T) {
	dec := NewDecoder(strings.NewReader(
		"data: " + strings.Repeat("x", 5000) + "\n\ndata: small\n\n"))
	dec.SetMaxEventBytes(1000)

	if _, err := dec.Next(); !errors.Is(err, ErrEventTooLarge) {
		t.Fatalf("first Next error = %v, want ErrEventTooLarge", err)
	}
	ev, err := dec.Next()
	if err != nil {
		t.Fatalf("second Next: %v", err)
	}
	if ev.Data != "small" {
		t.Errorf("Data = %q, want %q", ev.Data, "small")
	}
}

type flushRecorder struct {
	bytes.Buffer
	flushes int
}

func (f *flushRecorder) Flush() { f.flushes++ }

func TestEncoderWriteData(t *testing.T) {
	var buf bytes.Buffer
	if err := NewEncoder(&buf).WriteData("hello"); err != nil {
		t.Fatalf("WriteData: %v", err)
	}
	if got, want := buf.String(), "data: hello\n\n"; got != want {
		t.Errorf("output = %q, want %q", got, want)
	}
}

func TestEncoderWriteDone(t *testing.T) {
	var buf bytes.Buffer
	if err := NewEncoder(&buf).WriteDone(); err != nil {
		t.Fatalf("WriteDone: %v", err)
	}
	if got, want := buf.String(), "data: [DONE]\n\n"; got != want {
		t.Errorf("output = %q, want %q", got, want)
	}
}

func TestEncoderFlushesEachEvent(t *testing.T) {
	rec := &flushRecorder{}
	enc := NewEncoder(rec)

	for i := 0; i < 3; i++ {
		if err := enc.WriteData("tok"); err != nil {
			t.Fatalf("WriteData: %v", err)
		}
	}
	if rec.flushes != 3 {
		t.Errorf("flushes = %d, want 3 (streaming requires per-event flush)", rec.flushes)
	}
}

func TestEncoderRoundTrip(t *testing.T) {
	var buf bytes.Buffer
	enc := NewEncoder(&buf)

	original := Event{Name: "token", ID: "7", Data: "multi\nline"}
	if err := enc.WriteEvent(original); err != nil {
		t.Fatalf("WriteEvent: %v", err)
	}

	events := decodeAll(t, buf.String())
	if len(events) != 1 {
		t.Fatalf("got %d events, want 1", len(events))
	}
	got := events[0]
	if got.Name != original.Name || got.ID != original.ID || got.Data != original.Data {
		t.Errorf("round trip mismatch: got %+v, want %+v", got, original)
	}
}

func TestEncoderWriteComment(t *testing.T) {
	var buf bytes.Buffer
	if err := NewEncoder(&buf).WriteComment("ping"); err != nil {
		t.Fatalf("WriteComment: %v", err)
	}
	if got, want := buf.String(), ": ping\n\n"; got != want {
		t.Errorf("output = %q, want %q", got, want)
	}
}

type errWriter struct{}

func (errWriter) Write([]byte) (int, error) { return 0, errors.New("broken pipe") }

func TestEncoderPropagatesWriteErrors(t *testing.T) {
	enc := NewEncoder(errWriter{})
	if err := enc.WriteData("x"); err == nil {
		t.Error("WriteData swallowed a write error")
	}
	if err := enc.WriteEvent(Event{Data: "x"}); err == nil {
		t.Error("WriteEvent swallowed a write error")
	}
	if err := enc.WriteComment("x"); err == nil {
		t.Error("WriteComment swallowed a write error")
	}
}

// FuzzSSEDecoder checks the parser never panics on arbitrary input, which
// matters because the data comes from an upstream the gateway does not control.
func FuzzSSEDecoder(f *testing.F) {
	seeds := []string{
		"data: hello\n\n",
		"event: x\ndata: y\n\n",
		": comment\n\n",
		"data\n\n",
		"data: [DONE]\n\n",
		"\n\n\n",
		"data: a\rdata: b\n\n",
		"id: 1\nretry: 50\ndata: {}\n\n",
	}
	for _, s := range seeds {
		f.Add(s)
	}

	f.Fuzz(func(t *testing.T, input string) {
		dec := NewDecoder(strings.NewReader(input))
		dec.SetMaxEventBytes(4096)

		for i := 0; i < 1000; i++ {
			_, err := dec.Next()
			if err != nil {
				return
			}
		}
		t.Fatal("decoder did not terminate on bounded input")
	})
}
