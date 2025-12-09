package github

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strconv"
	"sync"
	"time"
)

// Client is a GitHub API client with rate limiting and authentication
type Client struct {
	token       string
	httpClient  *http.Client
	rateLimiter *RateLimiter
}

// RateLimiter tracks and enforces GitHub API rate limits
type RateLimiter struct {
	remaining int
	reset     time.Time
	mu        sync.Mutex
	ticker    *time.Ticker
}

// NewClient creates a new GitHub API client
func NewClient(token string) *Client {
	return &Client{
		token:      token,
		httpClient: &http.Client{Timeout: 90 * time.Second},
		rateLimiter: &RateLimiter{
			remaining: 5000,
			reset:     time.Now().Add(time.Hour),
			ticker:    time.NewTicker(time.Second), // Min 1 second between requests
		},
	}
}

// Wait blocks until it's safe to make another request
func (rl *RateLimiter) Wait() {
	rl.mu.Lock()
	defer rl.mu.Unlock()

	// Wait for ticker (minimum delay between requests)
	<-rl.ticker.C

	// If we're low on rate limit, wait until reset
	if rl.remaining <= 10 {
		waitTime := time.Until(rl.reset)
		if waitTime > 0 {
			rl.mu.Unlock()
			time.Sleep(waitTime)
			rl.mu.Lock()
		}
	}
}

// Update updates rate limit info from response headers
func (rl *RateLimiter) Update(resp *http.Response) {
	rl.mu.Lock()
	defer rl.mu.Unlock()

	if remaining := resp.Header.Get("X-RateLimit-Remaining"); remaining != "" {
		rl.remaining, _ = strconv.Atoi(remaining)
	}

	if reset := resp.Header.Get("X-RateLimit-Reset"); reset != "" {
		resetUnix, _ := strconv.ParseInt(reset, 10, 64)
		rl.reset = time.Unix(resetUnix, 0)
	}
}

// GetRemaining returns the current rate limit remaining
func (rl *RateLimiter) GetRemaining() int {
	rl.mu.Lock()
	defer rl.mu.Unlock()
	return rl.remaining
}

// doRequest performs an authenticated HTTP request with rate limiting
func (c *Client) doRequest(url string) (*http.Response, error) {
	c.rateLimiter.Wait()

	req, err := http.NewRequestWithContext(context.Background(), "GET", url, nil)
	if err != nil {
		return nil, err
	}

	req.Header.Set("Accept", "application/vnd.github+json")
	req.Header.Set("Authorization", "Bearer "+c.token)
	req.Header.Set("X-GitHub-Api-Version", "2022-11-28")

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, err
	}

	c.rateLimiter.Update(resp)

	if resp.StatusCode != 200 {
		resp.Body.Close()
		return nil, fmt.Errorf("API error: %d for %s", resp.StatusCode, url)
	}

	return resp, nil
}

// doRequestWithRetry performs a request with automatic retries
func (c *Client) doRequestWithRetry(url string, maxRetries int) (*http.Response, error) {
	var resp *http.Response
	var err error

	for attempt := 1; attempt <= maxRetries; attempt++ {
		resp, err = c.doRequest(url)
		if err == nil {
			return resp, nil
		}

		if attempt < maxRetries {
			waitTime := time.Duration(attempt) * 5 * time.Second
			time.Sleep(waitTime)
		}
	}

	return nil, fmt.Errorf("failed after %d attempts: %w", maxRetries, err)
}

// decodeJSON decodes JSON from response body
func decodeJSON(resp *http.Response, v interface{}) error {
	defer resp.Body.Close()
	return json.NewDecoder(resp.Body).Decode(v)
}
