package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"strings"
	"time"

	"go_coder_llm_pipeline/internal/github"
	"go_coder_llm_pipeline/internal/types"
)

func main() {
	// Command line flags
	maxPRs := flag.Int("max-prs", 50, "Maximum number of PRs to fetch per repository")
	testMode := flag.Bool("test", false, "Test mode: only fetch from first 5 repos")
	resume := flag.Bool("resume", true, "Resume from checkpoint if available")
	flag.Parse()

	// Check for GitHub token
	token := os.Getenv("GITHUB_TOKEN")
	if token == "" {
		log.Fatal("GITHUB_TOKEN environment variable is required")
	}

	// Read repository list from config file
	configPath := "config/repos.json"
	data, err := os.ReadFile(configPath)
	if err != nil {
		log.Fatalf("Failed to read %s: %v", configPath, err)
	}

	var repos []string
	if err := json.Unmarshal(data, &repos); err != nil {
		log.Fatalf("Failed to parse %s: %v", configPath, err)
	}

	if len(repos) == 0 {
		log.Fatal("No repositories found in config/repos.json. Run 'go run scripts/discover_repos.go' first.")
	}

	// Test mode: only first 5 repos
	if *testMode {
		if len(repos) > 5 {
			repos = repos[:5]
		}
		log.Printf("TEST MODE: Processing only first %d repositories", len(repos))
	}

	log.Printf("Found %d repositories to fetch", len(repos))
	log.Printf("Configuration: max-prs=%d, resume=%v", *maxPRs, *resume)

	// Create output directory
	outputDir := "data/raw"
	if err := os.MkdirAll(outputDir, 0755); err != nil {
		log.Fatalf("Failed to create output directory: %v", err)
	}

	// Load checkpoint
	var checkpoint *types.Checkpoint
	if *resume {
		checkpoint, err = github.LoadCheckpoint()
		if err != nil {
			log.Fatalf("Failed to load checkpoint: %v", err)
		}
		if len(checkpoint.ReposComplete) > 0 {
			log.Printf("Resuming from checkpoint: %d repos already complete", len(checkpoint.ReposComplete))
		}
	} else {
		checkpoint = &types.Checkpoint{
			ReposComplete: []string{},
			Timestamp:     time.Now(),
		}
	}

	// Create GitHub client
	client := github.NewClient(token)

	// Track overall progress
	startTime := time.Now()
	totalRepos := len(repos)
	processedRepos := 0
	skippedRepos := 0

	// Process each repository
	for i, repo := range repos {
		// Skip if already complete
		if github.IsRepoComplete(checkpoint, repo) {
			log.Printf("[%d/%d] Skipping %s (already complete)", i+1, totalRepos, repo)
			skippedRepos++
			continue
		}

		log.Printf("\n[%d/%d] Processing %s", i+1, totalRepos, repo)

		// Fetch PR data
		records, err := client.FetchRepoData(repo, *maxPRs)
		if err != nil {
			log.Printf("  ERROR: Failed to fetch %s: %v", repo, err)
			checkpoint.TotalErrors++
			if err := github.SaveCheckpoint(checkpoint); err != nil {
				log.Printf("  Warning: Failed to save checkpoint: %v", err)
			}
			continue
		}

		if len(records) == 0 {
			log.Printf("  No PRs found for %s", repo)
			github.MarkRepoComplete(checkpoint, repo)
			if err := github.SaveCheckpoint(checkpoint); err != nil {
				log.Printf("  Warning: Failed to save checkpoint: %v", err)
			}
			continue
		}

		// Save to JSONL file
		outputPath := filepath.Join(outputDir, sanitizeRepoName(repo)+".jsonl")
		if err := saveJSONL(records, outputPath); err != nil {
			log.Printf("  ERROR: Failed to save data: %v", err)
			checkpoint.TotalErrors++
			continue
		}

		// Update checkpoint
		processedRepos++
		checkpoint.TotalFetched += len(records)
		github.MarkRepoComplete(checkpoint, repo)
		if err := github.SaveCheckpoint(checkpoint); err != nil {
			log.Printf("  Warning: Failed to save checkpoint: %v", err)
		}

		// Print progress summary
		elapsed := time.Since(startTime)
		rate := float64(processedRepos) / elapsed.Seconds() * 60 // repos per minute
		remaining := totalRepos - processedRepos - skippedRepos
		eta := time.Duration(float64(remaining)/rate) * time.Minute

		log.Printf("  Saved %d PRs to %s", len(records), outputPath)
		log.Printf("  Progress: %d processed, %d skipped, %d remaining (%.1f repos/min, ETA: %v)",
			processedRepos, skippedRepos, remaining, rate, eta.Round(time.Minute))
	}

	// Final summary
	log.Printf("\n=== COMPLETED ===")
	log.Printf("Total repositories: %d", totalRepos)
	log.Printf("Processed: %d", processedRepos)
	log.Printf("Skipped (already done): %d", skippedRepos)
	log.Printf("Total PRs fetched: %d", checkpoint.TotalFetched)
	log.Printf("Errors: %d", checkpoint.TotalErrors)
	log.Printf("Total time: %v", time.Since(startTime).Round(time.Second))
	log.Printf("Output directory: %s", outputDir)
}

// sanitizeRepoName converts repo name to valid filename
func sanitizeRepoName(repo string) string {
	return strings.ReplaceAll(repo, "/", "_")
}

// saveJSONL saves records to a JSONL file
func saveJSONL(records []types.PRRecord, path string) error {
	f, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("failed to create file: %w", err)
	}
	defer f.Close()

	encoder := json.NewEncoder(f)
	for _, record := range records {
		if err := encoder.Encode(record); err != nil {
			return fmt.Errorf("failed to encode record: %w", err)
		}
	}

	return nil
}
