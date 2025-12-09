package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"net/url"
	"os"
	"time"
)

// Repo represents a GitHub repository from the search API
type Repo struct {
	FullName string `json:"full_name"`
	HTMLURL  string `json:"html_url"`
	Stars    int    `json:"stargazers_count"`
	License  struct {
		Key  string `json:"key"`
		Name string `json:"name"`
	} `json:"license"`
	PushedAt  string `json:"pushed_at"`
	Archived  bool   `json:"archived"`
	Fork      bool   `json:"fork"`
	Size      int    `json:"size"` // in KB
}

// SearchResponse represents GitHub's search API response
type SearchResponse struct {
	TotalCount int    `json:"total_count"`
	Items      []Repo `json:"items"`
}

func main() {
	token := os.Getenv("GITHUB_TOKEN")
	if token == "" {
		log.Fatal("GITHUB_TOKEN environment variable is required")
	}

	// Allowed open source licenses
	allowedLicenses := map[string]bool{
		"mit":           true,
		"apache-2.0":    true,
		"bsd-3-clause":  true,
		"bsd-2-clause":  true,
		"gpl-3.0":       true,
		"lgpl-3.0":      true,
		"mpl-2.0":       true,
	}

	var allRepos []string
	seen := make(map[string]bool)

	// GitHub search API allows max 1000 results (10 pages × 100 per page)
	// To get 5000 repos, we'll need multiple queries with different star ranges
	queries := []string{
		"language:Go stars:>=10000",                    // Top tier repos
		"language:Go stars:5000..9999",                 // High tier
		"language:Go stars:2000..4999",                 // Mid-high tier
		"language:Go stars:1000..1999",                 // Mid tier
		"language:Go stars:500..999",                   // Lower-mid tier
		"language:Go stars:100..499",                   // Entry tier
	}

	client := &http.Client{Timeout: 90 * time.Second} // Increased timeout for slow API responses

	for _, query := range queries {
		log.Printf("Searching: %s", query)

		// Fetch multiple pages for each query
		for page := 1; page <= 10; page++ {
			// Retry logic for transient failures
			var repos []Repo
			var err error
			maxRetries := 3
			for attempt := 1; attempt <= maxRetries; attempt++ {
				repos, err = searchRepos(client, token, query, page)
				if err == nil {
					break
				}
				if attempt < maxRetries {
					log.Printf("Attempt %d/%d failed, retrying in 5s: %v", attempt, maxRetries, err)
					time.Sleep(5 * time.Second)
				}
			}

			if err != nil {
				log.Printf("Error searching page %d after %d attempts: %v", page, maxRetries, err)
				break
			}

			if len(repos) == 0 {
				log.Printf("No more results for query")
				break
			}

			for _, repo := range repos {
				// Apply filters
				if repo.Archived {
					continue // Skip archived repos
				}
				if repo.Fork {
					continue // Skip forks
				}
				if repo.Size > 1024*1024 {
					continue // Skip repos > 1GB
				}
				if !allowedLicenses[repo.License.Key] {
					continue // Skip non-open source licenses
				}

				// Check if updated in last 2 years
				pushedAt, err := time.Parse(time.RFC3339, repo.PushedAt)
				if err != nil || time.Since(pushedAt) > 2*365*24*time.Hour {
					continue // Skip inactive repos
				}

				// Deduplicate
				if seen[repo.FullName] {
					continue
				}
				seen[repo.FullName] = true

				allRepos = append(allRepos, repo.FullName)
				log.Printf("Added: %s (stars: %d, license: %s)", repo.FullName, repo.Stars, repo.License.Key)

				// Stop if we've reached 5000
				if len(allRepos) >= 5000 {
					goto done
				}
			}

			// Rate limiting: wait 2 seconds between requests
			time.Sleep(2 * time.Second)
		}
	}

done:
	log.Printf("\nTotal repos found: %d", len(allRepos))

	// Initialize as empty array if no repos found
	if allRepos == nil {
		allRepos = []string{}
	}

	// Save to config/repos.json
	outputPath := "config/repos.json"
	f, err := os.Create(outputPath)
	if err != nil {
		log.Fatalf("Failed to create output file: %v", err)
	}
	defer f.Close()

	encoder := json.NewEncoder(f)
	encoder.SetIndent("", "  ")
	if err := encoder.Encode(allRepos); err != nil {
		log.Fatalf("Failed to write JSON: %v", err)
	}

	log.Printf("Successfully wrote %d repos to %s", len(allRepos), outputPath)

	// Print file size
	info, _ := os.Stat(outputPath)
	log.Printf("File size: %d KB", info.Size()/1024)
}

func searchRepos(client *http.Client, token, query string, page int) ([]Repo, error) {
	apiURL := fmt.Sprintf(
		"https://api.github.com/search/repositories?q=%s&per_page=100&page=%d&sort=stars&order=desc",
		url.QueryEscape(query), page,
	)

	req, err := http.NewRequestWithContext(context.Background(), "GET", apiURL, nil)
	if err != nil {
		return nil, err
	}

	req.Header.Set("Accept", "application/vnd.github+json")
	req.Header.Set("Authorization", "Bearer "+token)
	req.Header.Set("X-GitHub-Api-Version", "2022-11-28")

	resp, err := client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != 200 {
		return nil, fmt.Errorf("API error: %d", resp.StatusCode)
	}

	// Check rate limit
	remaining := resp.Header.Get("X-RateLimit-Remaining")
	log.Printf("  Page %d: Rate limit remaining: %s", page, remaining)

	var result SearchResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, err
	}

	return result.Items, nil
}
