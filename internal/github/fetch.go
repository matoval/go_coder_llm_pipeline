package github

import (
	"encoding/base64"
	"fmt"
	"go/parser"
	"go/token"
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
		Type  string `json:"type"` // User or Bot
	} `json:"user"`
	Base struct {
		SHA string `json:"sha"`
	} `json:"base"`
	Head struct {
		SHA string `json:"sha"`
	} `json:"head"`
	CreatedAt string  `json:"created_at"`
	MergedAt  *string `json:"merged_at"`
}

// IsMerged returns true if the PR was merged (derived from merged_at field)
func (pr *GitHubPR) IsMerged() bool {
	return pr.MergedAt != nil && *pr.MergedAt != ""
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
	Filename       string `json:"filename"`
	Status         string `json:"status"`
	Additions      int    `json:"additions"`
	Deletions      int    `json:"deletions"`
	Patch          string `json:"patch"`
	SHA            string `json:"sha"`
	ContentsURL    string `json:"contents_url"`
	PreviousFilename string `json:"previous_filename,omitempty"` // For renames
}

// GitHubContent represents file content from GitHub API
type GitHubContent struct {
	Content  string `json:"content"`  // Base64 encoded
	Encoding string `json:"encoding"` // Should be "base64"
	SHA      string `json:"sha"`
}

// GitHubCheckRun represents a check run from GitHub API
type GitHubCheckRun struct {
	Status     string `json:"status"`     // queued, in_progress, completed
	Conclusion string `json:"conclusion"` // success, failure, neutral, cancelled, skipped, timed_out, action_required
}

// QualityFilters defines filters for PR data quality
type QualityFilters struct {
	RequireMerged   bool  // Only include merged PRs
	ExcludeBots     bool  // Exclude bot-authored PRs
	MinFiles        int   // Minimum number of files changed
	MaxFiles        int   // Maximum number of files changed
	MinAdditions    int   // Minimum lines added
	MaxAdditions    int   // Maximum lines added
	MaxAgeYears     int   // Maximum PR age in years (0 = no limit)
}

// DefaultQualityFilters returns recommended quality filters
func DefaultQualityFilters() QualityFilters {
	return QualityFilters{
		RequireMerged:   true,
		ExcludeBots:     true,
		MinFiles:        1,
		MaxFiles:        20,
		MinAdditions:    10,
		MaxAdditions:    2000,
		MaxAgeYears:     2,
	}
}

// shouldIncludePR checks if a PR passes quality filters
func shouldIncludePR(pr GitHubPR, filters QualityFilters) (bool, string) {
	// Check merge status
	if filters.RequireMerged && !pr.IsMerged() {
		return false, "not merged"
	}

	// Check bot status
	if filters.ExcludeBots && isBot(pr.User.Login, pr.User.Type) {
		return false, "bot author"
	}

	// Check age
	if filters.MaxAgeYears > 0 {
		createdAt, err := time.Parse(time.RFC3339, pr.CreatedAt)
		if err == nil {
			age := time.Since(createdAt)
			maxAge := time.Duration(filters.MaxAgeYears) * 365 * 24 * time.Hour
			if age > maxAge {
				return false, fmt.Sprintf("too old (%d years)", int(age.Hours()/24/365))
			}
		}
	}

	return true, ""
}

// FetchRepoData fetches all PR data for a repository
func (c *Client) FetchRepoData(repoFullName string, maxPRs int) ([]types.PRRecord, error) {
	return c.FetchRepoDataWithFilters(repoFullName, maxPRs, DefaultQualityFilters())
}

// FetchRepoDataWithFilters fetches all PR data for a repository with quality filters
func (c *Client) FetchRepoDataWithFilters(repoFullName string, maxPRs int, filters QualityFilters) ([]types.PRRecord, error) {
	log.Printf("Fetching PRs from %s...", repoFullName)

	// Fetch list of PRs (fetch extra to account for filtering)
	fetchCount := maxPRs * 3 // Fetch 3x to account for ~66% filter rate
	if fetchCount > 300 {
		fetchCount = 300 // Cap at 300 to avoid excessive API calls
	}

	prs, err := c.fetchPRList(repoFullName, fetchCount)
	if err != nil {
		return nil, fmt.Errorf("failed to fetch PR list: %w", err)
	}

	log.Printf("  Found %d PRs, applying quality filters...", len(prs))

	var records []types.PRRecord
	filtered := 0
	filterReasons := make(map[string]int)

	for i, pr := range prs {
		// Stop if we've collected enough
		if len(records) >= maxPRs {
			break
		}

		if i > 0 && i%10 == 0 {
			log.Printf("  Progress: %d/%d PRs processed, %d accepted, %d filtered (rate limit: %d)",
				i, len(prs), len(records), filtered, c.rateLimiter.GetRemaining())
		}

		// Apply quality filters
		include, reason := shouldIncludePR(pr, filters)
		if !include {
			filtered++
			filterReasons[reason]++
			continue
		}

		record, err := c.fetchCompletePR(repoFullName, pr)
		if err != nil {
			log.Printf("  Warning: Failed to fetch PR #%d: %v", pr.Number, err)
			continue
		}

		// Additional filtering based on file counts (after fetching)
		fileCount := len(record.Files)
		if fileCount < filters.MinFiles || fileCount > filters.MaxFiles {
			filtered++
			filterReasons[fmt.Sprintf("file count (%d)", fileCount)]++
			continue
		}

		// Check total additions
		totalAdditions := 0
		for _, f := range record.Files {
			totalAdditions += f.Additions
		}
		if totalAdditions < filters.MinAdditions || totalAdditions > filters.MaxAdditions {
			filtered++
			filterReasons[fmt.Sprintf("additions (%d)", totalAdditions)]++
			continue
		}

		records = append(records, record)
	}

	// Log filter statistics
	log.Printf("  Completed: %d PRs fetched, %d filtered", len(records), filtered)
	if filtered > 0 {
		log.Printf("  Filter breakdown:")
		for reason, count := range filterReasons {
			log.Printf("    - %s: %d", reason, count)
		}
	}

	return records, nil
}

// fetchPRList fetches the list of PRs for a repository
func (c *Client) fetchPRList(repo string, maxPRs int) ([]GitHubPR, error) {
	var allPRs []GitHubPR
	perPage := 100
	page := 1

	for len(allPRs) < maxPRs {
		url := fmt.Sprintf(
			"https://api.github.com/repos/%s/pulls?state=closed&per_page=%d&page=%d&sort=updated&direction=desc",
			repo, perPage, page,
		)

		resp, err := c.doRequestWithRetry(url, 3)
		if err != nil {
			return nil, err
		}

		var prs []GitHubPR
		if err := decodeJSON(resp, &prs); err != nil {
			return nil, err
		}

		// If we got no results, we've reached the end
		if len(prs) == 0 {
			break
		}

		allPRs = append(allPRs, prs...)

		// If we got fewer than perPage results, this was the last page
		if len(prs) < perPage {
			break
		}

		page++
	}

	// Trim to maxPRs if we fetched more
	if len(allPRs) > maxPRs {
		allPRs = allPRs[:maxPRs]
	}

	return allPRs, nil
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
			Number:   pr.Number,
			Title:    pr.Title,
			Body:     pr.Body,
			State:    pr.State,
			Author:   pr.User.Login,
			HTMLURL:  pr.HTMLURL,
			Merged:   pr.IsMerged(),
			BaseSHA:  pr.Base.SHA,
			HeadSHA:  pr.Head.SHA,
			IsBot:    isBot(pr.User.Login, pr.User.Type),
			CIStatus: "unknown",
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

	// Fetch CI status if PR is merged
	if pr.IsMerged() && pr.Head.SHA != "" {
		record.PullRequest.CIStatus = c.fetchCIStatus(repo, pr.Head.SHA)
	}

	// Fetch comments (including review comments)
	comments, err := c.fetchPRComments(repo, pr.Number)
	if err != nil {
		log.Printf("    Warning: Failed to fetch comments for PR #%d: %v", pr.Number, err)
	} else {
		record.Comments = comments
	}

	// Fetch file changes with full content
	files, err := c.fetchPRFilesWithContent(repo, pr.Number, pr.Base.SHA, pr.Head.SHA)
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

// fetchPRFiles fetches file changes for a PR (legacy, without full content)
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

// fetchPRFilesWithContent fetches file changes with full before/after content
func (c *Client) fetchPRFilesWithContent(repo string, prNum int, baseSHA, headSHA string) ([]types.PRFile, error) {
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

		file := types.PRFile{
			Filename:  f.Filename,
			Status:    f.Status,
			Additions: f.Additions,
			Deletions: f.Deletions,
			Patch:     f.Patch,
			SHA:       f.SHA,
		}

		// Fetch content before (at base SHA) - skip for new files
		if f.Status != "added" {
			// Use previous filename if file was renamed
			beforePath := f.Filename
			if f.PreviousFilename != "" {
				beforePath = f.PreviousFilename
			}

			contentBefore, err := c.fetchFileContent(repo, beforePath, baseSHA)
			if err != nil {
				log.Printf("      Warning: Failed to fetch before content for %s: %v", f.Filename, err)
			} else {
				file.ContentBefore = contentBefore
			}
		}

		// Fetch content after (at head SHA) - skip for deleted files
		if f.Status != "removed" {
			contentAfter, err := c.fetchFileContent(repo, f.Filename, headSHA)
			if err != nil {
				log.Printf("      Warning: Failed to fetch after content for %s: %v", f.Filename, err)
			} else {
				file.ContentAfter = contentAfter
			}
		}

		// Validate syntax of the "after" code (what matters for training)
		if file.ContentAfter != "" {
			file.ValidSyntax = validateGoSyntax(file.ContentAfter)
		} else {
			// Deleted files don't need syntax validation
			file.ValidSyntax = true
		}

		files = append(files, file)
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

// fetchFileContent fetches the full content of a file at a specific ref (commit SHA or branch)
func (c *Client) fetchFileContent(repo, path, ref string) (string, error) {
	url := fmt.Sprintf("https://api.github.com/repos/%s/contents/%s?ref=%s", repo, path, ref)

	resp, err := c.doRequestWithRetry(url, 3)
	if err != nil {
		return "", err
	}

	var content GitHubContent
	if err := decodeJSON(resp, &content); err != nil {
		return "", err
	}

	// Decode base64 content
	if content.Encoding != "base64" {
		return "", fmt.Errorf("unexpected encoding: %s", content.Encoding)
	}

	decoded, err := base64.StdEncoding.DecodeString(content.Content)
	if err != nil {
		return "", fmt.Errorf("failed to decode base64 content: %w", err)
	}

	return string(decoded), nil
}

// fetchCIStatus fetches the CI/CD check status for a PR
func (c *Client) fetchCIStatus(repo string, ref string) string {
	url := fmt.Sprintf("https://api.github.com/repos/%s/commits/%s/check-runs", repo, ref)

	resp, err := c.doRequestWithRetry(url, 3)
	if err != nil {
		log.Printf("    Warning: Failed to fetch CI status: %v", err)
		return "unknown"
	}

	var response struct {
		CheckRuns []GitHubCheckRun `json:"check_runs"`
	}
	if err := decodeJSON(resp, &response); err != nil {
		log.Printf("    Warning: Failed to decode CI status: %v", err)
		return "unknown"
	}

	if len(response.CheckRuns) == 0 {
		return "none"
	}

	// Aggregate status: if any failed, status is failure; if all success, status is success
	allSuccess := true
	anyFailure := false
	for _, check := range response.CheckRuns {
		if check.Status == "completed" {
			if check.Conclusion == "failure" || check.Conclusion == "timed_out" {
				anyFailure = true
			} else if check.Conclusion != "success" {
				allSuccess = false
			}
		}
	}

	if anyFailure {
		return "failure"
	} else if allSuccess {
		return "success"
	}
	return "pending"
}

// validateGoSyntax checks if Go code is syntactically valid
func validateGoSyntax(code string) bool {
	fset := token.NewFileSet()
	_, err := parser.ParseFile(fset, "", code, parser.AllErrors)
	return err == nil
}

// isBot checks if a username or type indicates a bot account
func isBot(username, userType string) bool {
	// Check user type first
	if userType == "Bot" {
		return true
	}

	// Common bot name patterns
	botPatterns := []string{
		"bot", "Bot", "[bot]",
		"dependabot", "renovate",
		"greenkeeper", "snyk",
		"codecov", "coveralls",
		"github-actions", "travis",
	}

	usernameLower := strings.ToLower(username)
	for _, pattern := range botPatterns {
		if strings.Contains(usernameLower, strings.ToLower(pattern)) {
			return true
		}
	}

	return false
}

// FetchPRData is deprecated, kept for backward compatibility
// Deprecated: Use Client.FetchRepoData instead
func FetchPRData(repo string) ([]types.PRComment, error) {
	// This is kept for backward compatibility but should not be used
	return nil, fmt.Errorf("FetchPRData is deprecated, use Client.FetchRepoData instead")
}
