// Package tokenizer provides fast token-count approximations and a deterministic
// pseudo-tokenizer used for prefix blocking.
//
// Waypoint deliberately avoids a real BPE tokenizer on the request path. See
// DESIGN.md D5 for the tradeoff: an exact tokenizer costs a model-specific vocab
// download and ~microseconds per KB of prompt, while routing and admission
// control only need a count that is *monotone and consistent*, not exact.
package tokenizer

import (
	"strings"
	"unicode"
)

// Estimator approximates how many tokens a piece of text occupies.
type Estimator interface {
	// Count returns an approximate token count for text.
	Count(text string) int
	// Name identifies the estimator in metrics and logs.
	Name() string
}

// DefaultBytesPerToken is the empirical average for English prose under
// cl100k-style BPE vocabularies.
const DefaultBytesPerToken = 4.0

// CharHeuristic estimates tokens as len(text)/BytesPerToken.
//
// This is O(1) — it reads only the string length — which is why it is the
// default on the request path.
type CharHeuristic struct {
	BytesPerToken float64
}

// NewCharHeuristic returns a CharHeuristic with the default ratio.
func NewCharHeuristic() CharHeuristic {
	return CharHeuristic{BytesPerToken: DefaultBytesPerToken}
}

// Count implements Estimator.
func (h CharHeuristic) Count(text string) int {
	ratio := h.BytesPerToken
	if ratio <= 0 {
		ratio = DefaultBytesPerToken
	}
	if len(text) == 0 {
		return 0
	}
	n := int(float64(len(text))/ratio + 0.5)
	if n < 1 {
		return 1
	}
	return n
}

// Name implements Estimator.
func (h CharHeuristic) Name() string { return "char_heuristic" }

// SplitEstimator counts tokens by actually running Split. It is slower but
// closer to a real tokenizer, and is used by the mock backend where simulation
// fidelity matters more than per-request cost.
type SplitEstimator struct{}

// Count implements Estimator.
func (SplitEstimator) Count(text string) int { return len(Split(text)) }

// Name implements Estimator.
func (SplitEstimator) Name() string { return "split" }

// maxWordPiece caps how long a single alphanumeric token may be before it is
// broken up, mimicking how BPE splits long or rare words into subwords.
const maxWordPiece = 6

// Split deterministically breaks text into pseudo-tokens.
//
// The rules approximate byte-pair encoding closely enough for prefix blocking:
//   - a run of letters/digits, together with any single leading space, is one
//     token (BPE vocabularies encode " the" as a single token);
//   - runs longer than maxWordPiece are chopped into maxWordPiece-sized pieces;
//   - every other rune (punctuation, newline, extra whitespace) is its own token.
//
// Determinism is the critical property: the same text must always yield the same
// token boundaries, or prefix-cache block hashes would not be stable.
func Split(text string) []string {
	if text == "" {
		return nil
	}

	tokens := make([]string, 0, len(text)/3+1)
	runes := []rune(text)

	for i := 0; i < len(runes); {
		r := runes[i]

		// A space followed by an alphanumeric run attaches to that run.
		if r == ' ' && i+1 < len(runes) && isWordRune(runes[i+1]) {
			start := i
			i++ // consume the space
			for i < len(runes) && isWordRune(runes[i]) {
				i++
			}
			tokens = appendWordPieces(tokens, string(runes[start:i]))
			continue
		}

		if isWordRune(r) {
			start := i
			for i < len(runes) && isWordRune(runes[i]) {
				i++
			}
			tokens = appendWordPieces(tokens, string(runes[start:i]))
			continue
		}

		tokens = append(tokens, string(r))
		i++
	}
	return tokens
}

// appendWordPieces splits an over-long word into subword pieces.
func appendWordPieces(dst []string, word string) []string {
	// Account for a leading space without letting it skew the piece length.
	prefix := ""
	if strings.HasPrefix(word, " ") {
		prefix, word = " ", word[1:]
	}

	if len([]rune(word)) <= maxWordPiece {
		return append(dst, prefix+word)
	}

	runes := []rune(word)
	for start := 0; start < len(runes); start += maxWordPiece {
		end := start + maxWordPiece
		if end > len(runes) {
			end = len(runes)
		}
		piece := string(runes[start:end])
		if start == 0 {
			piece = prefix + piece
		}
		dst = append(dst, piece)
	}
	return dst
}

func isWordRune(r rune) bool {
	return unicode.IsLetter(r) || unicode.IsDigit(r)
}
