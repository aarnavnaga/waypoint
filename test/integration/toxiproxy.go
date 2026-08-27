//go:build integration

package integration

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"time"
)

// ToxiproxyClient is a minimal client for Toxiproxy's control API.
//
// Toxiproxy injects *transport*-level faults — connection resets, added latency,
// bandwidth limits — which an application-level fault knob cannot express. A
// backend that returns HTTP 500 is a very different failure from one whose TCP
// connection is severed mid-stream, and resilience code has to handle both.
type ToxiproxyClient struct {
	baseURL string
	client  *http.Client
}

// NewToxiproxyClient returns a client for the control API at baseURL.
func NewToxiproxyClient(baseURL string) *ToxiproxyClient {
	return &ToxiproxyClient{
		baseURL: baseURL,
		client:  &http.Client{Timeout: 5 * time.Second},
	}
}

// Available reports whether the Toxiproxy control API is reachable.
func (c *ToxiproxyClient) Available() bool {
	resp, err := c.client.Get(c.baseURL + "/version")
	if err != nil {
		return false
	}
	defer func() { _ = resp.Body.Close() }()
	return resp.StatusCode == http.StatusOK
}

// Proxy describes a Toxiproxy proxy.
type Proxy struct {
	Name     string `json:"name"`
	Listen   string `json:"listen"`
	Upstream string `json:"upstream"`
	Enabled  bool   `json:"enabled"`
}

// CreateProxy registers a proxy forwarding listen -> upstream.
func (c *ToxiproxyClient) CreateProxy(name, listen, upstream string) (*Proxy, error) {
	body, err := json.Marshal(Proxy{
		Name: name, Listen: listen, Upstream: upstream, Enabled: true,
	})
	if err != nil {
		return nil, err
	}

	resp, err := c.client.Post(c.baseURL+"/proxies", "application/json", bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("create proxy: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusCreated && resp.StatusCode != http.StatusConflict {
		return nil, fmt.Errorf("create proxy %s: status %d", name, resp.StatusCode)
	}

	var created Proxy
	if err := json.NewDecoder(resp.Body).Decode(&created); err != nil {
		// A conflict response may not carry a body; the proxy still exists.
		created = Proxy{Name: name, Listen: listen, Upstream: upstream, Enabled: true}
	}
	return &created, nil
}

// DeleteProxy removes a proxy.
func (c *ToxiproxyClient) DeleteProxy(name string) error {
	req, err := http.NewRequest(http.MethodDelete, c.baseURL+"/proxies/"+name, nil)
	if err != nil {
		return err
	}
	resp, err := c.client.Do(req)
	if err != nil {
		return err
	}
	defer func() { _ = resp.Body.Close() }()
	return nil
}

// SetEnabled turns a proxy on or off. Disabling severs existing connections,
// which simulates a backend disappearing mid-flight.
func (c *ToxiproxyClient) SetEnabled(name string, enabled bool) error {
	body, err := json.Marshal(map[string]bool{"enabled": enabled})
	if err != nil {
		return err
	}

	req, err := http.NewRequest(http.MethodPost, c.baseURL+"/proxies/"+name, bytes.NewReader(body))
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := c.client.Do(req)
	if err != nil {
		return fmt.Errorf("set enabled: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("set enabled %s: status %d", name, resp.StatusCode)
	}
	return nil
}

// AddLatency injects a fixed delay (ms) with jitter on the downstream path.
func (c *ToxiproxyClient) AddLatency(proxy, toxicName string, latencyMS, jitterMS int) error {
	return c.addToxic(proxy, map[string]any{
		"name":   toxicName,
		"type":   "latency",
		"stream": "downstream",
		"attributes": map[string]int{
			"latency": latencyMS,
			"jitter":  jitterMS,
		},
	})
}

// AddTimeout stops all data after the given delay, simulating a stalled peer.
func (c *ToxiproxyClient) AddTimeout(proxy, toxicName string, timeoutMS int) error {
	return c.addToxic(proxy, map[string]any{
		"name":       toxicName,
		"type":       "timeout",
		"stream":     "downstream",
		"attributes": map[string]int{"timeout": timeoutMS},
	})
}

func (c *ToxiproxyClient) addToxic(proxy string, toxic map[string]any) error {
	body, err := json.Marshal(toxic)
	if err != nil {
		return err
	}

	resp, err := c.client.Post(c.baseURL+"/proxies/"+proxy+"/toxics",
		"application/json", bytes.NewReader(body))
	if err != nil {
		return fmt.Errorf("add toxic: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusOK && resp.StatusCode != http.StatusCreated {
		return fmt.Errorf("add toxic to %s: status %d", proxy, resp.StatusCode)
	}
	return nil
}

// RemoveToxic deletes a toxic from a proxy.
func (c *ToxiproxyClient) RemoveToxic(proxy, toxicName string) error {
	req, err := http.NewRequest(http.MethodDelete,
		c.baseURL+"/proxies/"+proxy+"/toxics/"+toxicName, nil)
	if err != nil {
		return err
	}
	resp, err := c.client.Do(req)
	if err != nil {
		return err
	}
	defer func() { _ = resp.Body.Close() }()
	return nil
}
