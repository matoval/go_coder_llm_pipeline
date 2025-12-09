package github

import (
	"fmt"
	"log"
	"strings"
	"time"

	"go_coder_llm_pipeline/internal/types"
)

// GitHubPR represents raw PR data from GitHub API
type GitHubPR struct {
	Number    int    `json:"number"`
	Title     string `json:"title"`
	Body      string `json:"body"`
	State     string `json:"state"`
	HTMLURL   string `json:"html_url"`
	User      struct {
		Login string `json:"login"`
	} `json:"user"`
	CreatedAt string  `json:"created_at"`
	MergedAt  *string `json:"merged_at"`
}

// GitHubComment represents raw comment data
type GitHubComment struct {
	ID   int    `json:"id"`
	Body string `json:"body"`
	User struct {
		Login string `json:"login"`
	} `json:"user"`
	CreatedAt string  `json:"created_at"`
	Path      *string `json:"path"`
	DiffHunk  *string `json:"diff_hunk"`
}

// GitHubFile represents raw file data
type GitHubFile struct {
	Filename  string `json:"filename"`
	Status    string `json:"status"`
	Additions int    `json:"additions"`
	Deletions int    `json:"deletions"`
	Patch     string `json:"patch"`
}

// FetchRepoData fetches all PR data for a repository
func (c *Client) FetchRepoData(repoFullName string, maxPRs int) ([]types.PRRecord, error) {
	log.Printf("Fetching PRs from %s...", repoFullName)

	// Fetch list of PRs
	prs, err := c.fetchPRList(repoFullName, maxPRs)
	if err != nil {
		return nil, fmt.Errorf("failed to fetch PR list: %w", err)
	}

	log.Printf("  Found %d PRs", len(prs))

	var records []types.PRRecord
	for i, pr := range prs {
		if i > 0 && i%10 == 0 {
			log.Printf("  Progress: %d/%d PRs processed (rate limit: %d)", i, len(prs), c.rateLimiter.GetRemaining())
		}

		record, err := c.fetchCompletePR(repoFullName, pr)
		if err != nil {
			log.Printf("  Warning: Failed to fetch PR #%d: %v", pr.Number, err)
			continue
		}

		records = append(records, record)
	}

	log.Printf("  Completed: %d PRs fetched successfully", len(records))
	return records, nil
}

// fetchPRList fetches the list of PRs for a repository
func (c *Client) fetchPRList(repo string, maxPRs int) ([]GitHubPR, error) {
	perPage := 100
	if maxPRs < perPage {
		perPage = maxPRs
	}

	url := fmt.Sprintf(
		"https://api.github.com/repos/%s/pulls?state=closed&per_page=%d&sort=updated&direction=desc",
		repo, perPage,
	)

	resp, err := c.doRequestWithRetry(url, 3)
	if err != nil {
		return nil, err
	}

	var prs []GitHubPR
	if err := decodeJSON(resp, &prs); err != nil {
		return nil, err
	}

	if len(prs) > maxPRs {
		prs = prs[:maxPRs]
	}

	return prs, nil
}

// fetchCompletePR fetches all data for a single PR
func (c *Client) fetchCompletePR(repo string, pr GitHubPR) (types.PRRecord, error) {
	record := types.PRRecord{
		Repo: types.RepoMeta{
			FullName:  repo,
			HTMLURL:   fmt.Sprintf("https://github.com/%s", repo),
			FetchedAt: time.Now(),
		},
		PullRequest: types.PullRequest{
			Number:  pr.Number,
			Title:   pr.Title,
			Body:    pr.Body,
			State:   pr.State,
			Author:  pr.User.Login,
			HTMLURL: pr.HTMLURL,
		},
	}

	// Parse timestamps
	if createdAt, err := time.Parse(time.RFC3339, pr.CreatedAt); err == nil {
		record.PullRequest.CreatedAt = createdAt
	}
	if pr.MergedAt != nil {
		if mergedAt, err := time.Parse(time.RFC3339, *pr.MergedAt); err == nil {
			record.PullRequest.MergedAt = &mergedAt
		}
	}

	// Fetch comments (including review comments)
	comments, err := c.fetchPRComments(repo, pr.Number)
	if err != nil {
		log.Printf("    Warning: Failed to fetch comments for PR #%d: %v", pr.Number, err)
	} else {
		record.Comments = comments
	}

	// Fetch file changes
	files, err := c.fetchPRFiles(repo, pr.Number)
	if err != nil {
		log.Printf("    Warning: Failed to fetch files for PR #%d: %v", pr.Number, err)
	} else {
		record.Files = files
	}

	// Categorize the PR
	record.Tags = categorizePR(record)

	return record, nil
}

// fetchPRComments fetches all comments for a PR (issue comments + review comments)
func (c *Client) fetchPRComments(repo string, prNum int) ([]types.Comment, error) {
	var allComments []types.Comment

	// Fetch issue comments
	issueURL := fmt.Sprintf("https://api.github.com/repos/%s/issues/%d/comments", repo, prNum)
	resp, err := c.doRequestWithRetry(issueURL, 3)
	if err != nil {
		return nil, err
	}

	var issueComments []GitHubComment
	if err := decodeJSON(resp, &issueComments); err != nil {
		return nil, err
	}

	for _, comment := range issueComments {
		createdAt, _ := time.Parse(time.RFC3339, comment.CreatedAt)
		allComments = append(allComments, types.Comment{
			ID:        comment.ID,
			User:      comment.User.Login,
			Body:      comment.Body,
			CreatedAt: createdAt,
		})
	}

	// Fetch review comments
	reviewURL := fmt.Sprintf("https://api.github.com/repos/%s/pulls/%d/comments", repo, prNum)
	resp, err = c.doRequestWithRetry(reviewURL, 3)
	if err != nil {
		return allComments, nil // Return issue comments even if review comments fail
	}

	var reviewComments []GitHubComment
	if err := decodeJSON(resp, &reviewComments); err != nil {
		return allComments, nil
	}

	for _, comment := range reviewComments {
		createdAt, _ := time.Parse(time.RFC3339, comment.CreatedAt)
		c := types.Comment{
			ID:        comment.ID,
			User:      comment.User.Login,
			Body:      comment.Body,
			CreatedAt: createdAt,
		}
		if comment.Path != nil {
			c.Path = *comment.Path
		}
		if comment.DiffHunk != nil {
			c.DiffHunk = *comment.DiffHunk
		}
		allComments = append(allComments, c)
	}

	return allComments, nil
}

// fetchPRFiles fetches file changes for a PR
func (c *Client) fetchPRFiles(repo string, prNum int) ([]types.PRFile, error) {
	url := fmt.Sprintf("https://api.github.com/repos/%s/pulls/%d/files", repo, prNum)
	resp, err := c.doRequestWithRetry(url, 3)
	if err != nil {
		return nil, err
	}

	var ghFiles []GitHubFile
	if err := decodeJSON(resp, &ghFiles); err != nil {
		return nil, err
	}

	var files []types.PRFile
	for _, f := range ghFiles {
		// Filter out non-Go files and excluded paths
		if !shouldIncludeFile(f.Filename) {
			continue
		}

		files = append(files, types.PRFile{
			Filename:  f.Filename,
			Status:    f.Status,
			Additions: f.Additions,
			Deletions: f.Deletions,
			Patch:     f.Patch,
		})
	}

	return files, nil
}

// shouldIncludeFile determines if a file should be included in the dataset
func shouldIncludeFile(path string) bool {
	// Exclude vendor, third-party, generated files
	excludedPrefixes := []string{
		"vendor/",
		"third_party/",
		"node_modules/",
		".git/",
		"testdata/",
	}

	for _, prefix := range excludedPrefixes {
		if strings.HasPrefix(path, prefix) {
			return false
		}
	}

	// Exclude generated files
	if strings.HasSuffix(path, ".pb.go") ||
		strings.HasSuffix(path, ".gen.go") ||
		strings.Contains(path, "generated") {
		return false
	}

	// Only include Go files
	return strings.HasSuffix(path, ".go")
}

// categorizePR categorizes a PR based on its content
func categorizePR(record types.PRRecord) []string {
	var tags []string

	pr := record.PullRequest

	// Natural language: PR has meaningful description
	if len(strings.Fields(pr.Body)) > 10 && !hasCodeBlocks(pr.Body) {
		tags = append(tags, "NAT")
	}

	// Mixed: Has code blocks in description or comments
	if hasCodeBlocks(pr.Body) {
		tags = append(tags, "MIX")
	}

	// Code: Has file changes
	if len(record.Files) > 0 {
		tags = append(tags, "CODE")
	}

	// Default to NAT if no tags
	if len(tags) == 0 {
		tags = append(tags, "NAT")
	}

	return tags
}

// hasCodeBlocks checks if text contains code blocks
func hasCodeBlocks(text string) bool {
	return strings.Contains(text, "```") || strings.Contains(text, "`")
}

// FetchPRData is deprecated, kept for backward compatibility
// Deprecated: Use Client.FetchRepoData instead
func FetchPRData(repo string) ([]types.PRComment, error) {
	// This is kept for backward compatibility but should not be used
	return nil, fmt.Errorf("FetchPRData is deprecated, use Client.FetchRepoData instead")
}
