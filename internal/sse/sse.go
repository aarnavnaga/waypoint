// Package sse implements the Server-Sent Events wire format used by
// OpenAI-compatible streaming APIs.
//
// The parser follows the WHATWG EventSource rules closely enough for that use:
// fields are `name: value`, a single leading space after the colon is stripped,
// multiple `data` lines in one event are joined with newlines, a blank line
// dispatches the event, and lines starting with `:` are comments (commonly used
// as keep-alive heartbeats).
package sse

import (
	"bufio"
	"bytes"
	"errors"
	"fmt"
	"io"
	"strings"
)

// DonePayload is the sentinel data value OpenAI uses to close a stream.
const DonePayload = "[DONE]"

// Event is one dispatched server-sent event.
type Event struct {
	// Name is the `event:` field, empty for the default "message" type.
	Name string
	// Data is the concatenated `data:` payload with no trailing newline.
	Data string
	// ID is the `id:` field, if present.
	ID string
	// Retry is the `retry:` field verbatim, if present.
	Retry string
	// Comment holds `:` lines seen before this event, used for heartbeats.
	Comment string
}

// IsDone reports whether this event is the OpenAI stream terminator.
func (e Event) IsDone() bool { return strings.TrimSpace(e.Data) == DonePayload }

// DefaultMaxEventBytes bounds a single event, so a hostile or broken upstream
// cannot drive the gateway out of memory on one unterminated event.
const DefaultMaxEventBytes = 1 << 20 // 1 MiB

// ErrEventTooLarge is returned when an event exceeds the configured cap.
var ErrEventTooLarge = errors.New("sse: event exceeds maximum size")

// Decoder reads Events from a stream.
type Decoder struct {
	scanner *bufio.Scanner
	maxData int

	data    bytes.Buffer
	name    string
	id      string
	retry   string
	comment strings.Builder
	started bool
}

// NewDecoder returns a Decoder reading from r.
func NewDecoder(r io.Reader) *Decoder {
	sc := bufio.NewScanner(r)
	// Allow long single lines; a token here is one SSE line.
	sc.Buffer(make([]byte, 0, 64*1024), DefaultMaxEventBytes)
	return &Decoder{scanner: sc, maxData: DefaultMaxEventBytes}
}

// SetMaxEventBytes overrides the per-event size cap.
func (d *Decoder) SetMaxEventBytes(n int) {
	if n > 0 {
		d.maxData = n
	}
}

// Next returns the next event, or io.EOF when the stream ends.
//
// A trailing event that was accumulating when the stream ended is returned
// before io.EOF, because upstreams do not always send a final blank line.
func (d *Decoder) Next() (Event, error) {
	for d.scanner.Scan() {
		line := d.scanner.Text()
		// Tolerate CRLF line endings.
		line = strings.TrimSuffix(line, "\r")

		// A blank line dispatches whatever has accumulated.
		if line == "" {
			if !d.started {
				// Blank line with nothing pending: ignore.
				continue
			}
			return d.dispatch(), nil
		}

		// Comment line.
		if strings.HasPrefix(line, ":") {
			d.comment.WriteString(strings.TrimPrefix(line, ":"))
			continue
		}

		name, value := splitField(line)
		switch name {
		case "data":
			if d.data.Len() > 0 {
				d.data.WriteByte('\n')
			}
			if d.data.Len()+len(value) > d.maxData {
				d.reset()
				return Event{}, ErrEventTooLarge
			}
			d.data.WriteString(value)
			d.started = true
		case "event":
			d.name = value
			d.started = true
		case "id":
			d.id = value
			d.started = true
		case "retry":
			d.retry = value
			d.started = true
		default:
			// Unknown field: ignored per spec.
		}
	}

	if err := d.scanner.Err(); err != nil {
		if errors.Is(err, bufio.ErrTooLong) {
			return Event{}, ErrEventTooLarge
		}
		return Event{}, fmt.Errorf("sse: read: %w", err)
	}

	// Flush a final event that never got its blank line.
	if d.started {
		return d.dispatch(), nil
	}
	return Event{}, io.EOF
}

func (d *Decoder) dispatch() Event {
	ev := Event{
		Name:    d.name,
		Data:    d.data.String(),
		ID:      d.id,
		Retry:   d.retry,
		Comment: d.comment.String(),
	}
	d.reset()
	return ev
}

func (d *Decoder) reset() {
	d.data.Reset()
	d.comment.Reset()
	d.name, d.id, d.retry = "", "", ""
	d.started = false
}

// splitField parses `name: value`, stripping one optional space after the colon.
// A line with no colon is a field name with an empty value.
func splitField(line string) (name, value string) {
	idx := strings.IndexByte(line, ':')
	if idx < 0 {
		return line, ""
	}
	name = line[:idx]
	value = line[idx+1:]
	return name, strings.TrimPrefix(value, " ")
}

// Encoder writes Events, flushing after each one so clients see tokens as they
// are produced rather than when a buffer happens to fill.
type Encoder struct {
	w       io.Writer
	flusher interface{ Flush() }
}

// NewEncoder returns an Encoder writing to w. If w implements Flush(), it is
// called after every event.
func NewEncoder(w io.Writer) *Encoder {
	enc := &Encoder{w: w}
	if f, ok := w.(interface{ Flush() }); ok {
		enc.flusher = f
	}
	return enc
}

// WriteData writes a bare `data:` event.
func (e *Encoder) WriteData(data string) error {
	if _, err := fmt.Fprintf(e.w, "data: %s\n\n", data); err != nil {
		return err
	}
	e.flush()
	return nil
}

// WriteEvent writes a full event, emitting each field that is set.
func (e *Encoder) WriteEvent(ev Event) error {
	var b strings.Builder
	if ev.Name != "" {
		fmt.Fprintf(&b, "event: %s\n", ev.Name)
	}
	if ev.ID != "" {
		fmt.Fprintf(&b, "id: %s\n", ev.ID)
	}
	if ev.Retry != "" {
		fmt.Fprintf(&b, "retry: %s\n", ev.Retry)
	}
	// Multi-line payloads must be emitted as one `data:` line each.
	for _, line := range strings.Split(ev.Data, "\n") {
		fmt.Fprintf(&b, "data: %s\n", line)
	}
	b.WriteByte('\n')

	if _, err := io.WriteString(e.w, b.String()); err != nil {
		return err
	}
	e.flush()
	return nil
}

// WriteDone writes the OpenAI stream terminator.
func (e *Encoder) WriteDone() error { return e.WriteData(DonePayload) }

// WriteComment writes a `:` heartbeat line.
func (e *Encoder) WriteComment(text string) error {
	if _, err := fmt.Fprintf(e.w, ": %s\n\n", text); err != nil {
		return err
	}
	e.flush()
	return nil
}

func (e *Encoder) flush() {
	if e.flusher != nil {
		e.flusher.Flush()
	}
}
