# Data Pipeline Guide

Complete guide for collecting and processing GitHub data for model training.

## Overview

The data pipeline fetches Go repositories, pull requests, issues, and code from GitHub, processes them into structured format, and prepares training corpus.

```
GitHub API → Fetch → Parse → Categorize → Serialize → Corpus
```

## Pipeline Stages

### Stage 1: Repository Selection

Select top Go repositories based on quality criteria.

**Criteria**:
- Language: Go
- License: MIT, Apache-2.0, GPL, BSD, MPL-2.0 (open source only)
- Stars: 100+ (quality filter)
- Active: Updated in last 2 years
- Size: Exclude massive monorepos (>1GB)

**GitHub Search Query**:
```
language:Go stars:>=100 license:mit OR license:apache-2.0 pushed:>2023-01-01
```

### Stage 2: Data Fetching

Fetch repository data using GitHub REST API.

**Data Sources**:
1. **Pull Requests**: Titles, descriptions, comments, reviews
2. **Issues**: Titles, descriptions, comments
3. **Code Files**: `.go` files with changes
4. **Diffs**: Unified diffs from PRs
5. **README**: Documentation
6. **File Tree**: Repository structure

## Implementation Guide

### GitHub API Authentication

```bash
# Set your GitHub token
export GITHUB_TOKEN=ghp_your_token_here
```

**Required Scopes**:
- `public_repo`: Access public repositories
- `read:org`: Read organization data (optional)

### Repository Fetching

**Current Implementation** (cmd/fetchdata/main.go:10):

```go
repos := []string{
    "golang/go",
    "spf13/cobra",
    "wailsapp/wails",
}

for _, repo := range repos {
    log.Printf("Fetching data from %s...", repo)
    data, err := github.FetchPRData(repo)
    if err != nil {
        log.Fatal(err)
    }
    tokenizer.SaveCorpus(data, "data/processed/"+repo+".txt")
}
```

### Enhanced Repository Selection

**Step 1: Discover Repositories**

Create `cmd/discover/main.go`:

```go
package main

import (
    "context"
    "encoding/json"
    "fmt"
    "net/http"
    "os"
)

type Repo struct {
    FullName   string `json:"full_name"`
    HTMLURL    string `json:"html_url"`
    Stars      int    `json:"stargazers_count"`
    License    struct {
        Key  string `json:"key"`
        Name string `json:"name"`
    } `json:"license"`
    PushedAt string `json:"pushed_at"`
}

func searchRepos(query string, perPage int) ([]Repo, error) {
    url := fmt.Sprintf(
        "https://api.github.com/search/repositories?q=%s&per_page=%d&sort=stars",
        query, perPage,
    )

    req, _ := http.NewRequestWithContext(context.Background(), "GET", url, nil)
    req.Header.Set("Accept", "application/vnd.github+json")
    req.Header.Set("Authorization", "Bearer "+os.Getenv("GITHUB_TOKEN"))

    resp, err := http.DefaultClient.Do(req)
    if err != nil {
        return nil, err
    }
    defer resp.Body.Close()

    var result struct {
        Items []Repo `json:"items"`
    }
    json.NewDecoder(resp.Body).Decode(&result)
    return result.Items, nil
}

func main() {
    repos, err := searchRepos("language:Go stars:>=100", 100)
    if err != nil {
        panic(err)
    }

    // Save to file
    f, _ := os.Create("data/repos.json")
    json.NewEncoder(f).Encode(repos)
}
```

**Step 2: Filter by License**

```go
func filterByLicense(repos []Repo) []Repo {
    allowed := map[string]bool{
        "mit": true,
        "apache-2.0": true,
        "gpl-3.0": true,
        "bsd-3-clause": true,
        "mpl-2.0": true,
    }

    var filtered []Repo
    for _, r := range repos {
        if allowed[r.License.Key] {
            filtered = append(filtered, r)
        }
    }
    return filtered
}
```

### Pull Request Data Extraction

**Enhanced Implementation** for internal/github/fetch.go:18:

```go
package github

import (
    "context"
    "encoding/json"
    "fmt"
    "net/http"
    "os"
    "time"
)

type Client struct {
    token      string
    httpClient *http.Client
    rateLimit  *time.Ticker
}

func NewClient(token string) *Client {
    return &Client{
        token:      token,
        httpClient: &http.Client{Timeout: 30 * time.Second},
        rateLimit:  time.NewTicker(time.Second), // 1 req/sec
    }
}

func (c *Client) FetchPRs(repo string, state string, limit int) ([]PullRequest, error) {
    <-c.rateLimit.C // Rate limiting

    url := fmt.Sprintf(
        "https://api.github.com/repos/%s/pulls?state=%s&per_page=%d",
        repo, state, limit,
    )

    req, _ := http.NewRequestWithContext(context.Background(), "GET", url, nil)
    req.Header.Set("Accept", "application/vnd.github+json")
    req.Header.Set("Authorization", "Bearer "+c.token)

    resp, err := c.httpClient.Do(req)
    if err != nil {
        return nil, err
    }
    defer resp.Body.Close()

    if resp.StatusCode != 200 {
        return nil, fmt.Errorf("API error: %d", resp.StatusCode)
    }

    var prs []PullRequest
    json.NewDecoder(resp.Body).Decode(&prs)
    return prs, nil
}

func (c *Client) FetchPRFiles(repo string, prNum int) ([]PRFile, error) {
    <-c.rateLimit.C

    url := fmt.Sprintf(
        "https://api.github.com/repos/%s/pulls/%d/files",
        repo, prNum,
    )

    req, _ := http.NewRequestWithContext(context.Background(), "GET", url, nil)
    req.Header.Set("Authorization", "Bearer "+c.token)

    resp, err := c.httpClient.Do(req)
    if err != nil {
        return nil, err
    }
    defer resp.Body.Close()

    var files []PRFile
    json.NewDecoder(resp.Body).Decode(&files)
    return files, nil
}

func (c *Client) FetchPRComments(repo string, prNum int) ([]Comment, error) {
    <-c.rateLimit.C

    url := fmt.Sprintf(
        "https://api.github.com/repos/%s/pulls/%d/comments",
        repo, prNum,
    )

    req, _ := http.NewRequestWithContext(context.Background(), "GET", url, nil)
    req.Header.Set("Authorization", "Bearer "+c.token)

    resp, err := c.httpClient.Do(req)
    if err != nil {
        return nil, err
    }
    defer resp.Body.Close()

    var comments []Comment
    json.NewDecoder(resp.Body).Decode(&comments)
    return comments, nil
}
```

### Data Structures

**Enhanced types** for internal/types/types.go:1:

```go
package types

type RepoMeta struct {
    FullName   string `json:"full_name"`
    HTMLURL    string `json:"html_url"`
    Stars      int    `json:"stars"`
    LicenseKey string `json:"license_key"`
    FetchedAt  string `json:"fetched_at"`
}

type PullRequest struct {
    Number    int      `json:"number"`
    Title     string   `json:"title"`
    Body      string   `json:"body"`
    State     string   `json:"state"`
    Author    string   `json:"user.login"`
    CreatedAt string   `json:"created_at"`
    MergedAt  string   `json:"merged_at,omitempty"`
    HTMLURL   string   `json:"html_url"`
}

type Comment struct {
    ID        int    `json:"id"`
    User      string `json:"user.login"`
    Body      string `json:"body"`
    CreatedAt string `json:"created_at"`
    Path      string `json:"path,omitempty"`      // For review comments
    DiffHunk  string `json:"diff_hunk,omitempty"` // Context
}

type PRFile struct {
    Filename    string `json:"filename"`
    Status      string `json:"status"` // added, removed, modified
    Additions   int    `json:"additions"`
    Deletions   int    `json:"deletions"`
    Patch       string `json:"patch"` // Unified diff
    RawURL      string `json:"raw_url"`
}

type PRRecord struct {
    Repo         RepoMeta    `json:"repo"`
    PullRequest  PullRequest `json:"pull_request"`
    Comments     []Comment   `json:"comments"`
    Files        []PRFile    `json:"files"`
    FileTree     []string    `json:"file_tree,omitempty"`
    Tags         []string    `json:"tags"` // NAT, MIX, CODE
}
```

## Data Processing

### Categorization

**Rules**:

```go
func categorize(pr PRRecord) []string {
    tags := []string{}

    // Natural language: mostly text
    if hasNaturalLanguage(pr.PullRequest.Body) && !hasCodeBlocks(pr.PullRequest.Body) {
        tags = append(tags, "NAT")
    }

    // Mixed: text with code
    if hasCodeBlocks(pr.PullRequest.Body) || hasInlineCode(pr.PullRequest.Body) {
        tags = append(tags, "MIX")
    }

    // Code: file changes
    if len(pr.Files) > 0 {
        tags = append(tags, "CODE")
    }

    return tags
}

func hasCodeBlocks(text string) bool {
    return strings.Contains(text, "```")
}

func hasInlineCode(text string) bool {
    return strings.Contains(text, "`") && !strings.Contains(text, "```")
}

func hasNaturalLanguage(text string) bool {
    words := strings.Fields(text)
    return len(words) > 10 // At least 10 words
}
```

### Text Serialization

**Format for training corpus** (internal/tokenizer/prepare.go:16):

```go
func SerializePR(pr PRRecord) string {
    var buf strings.Builder

    // Metadata
    buf.WriteString(fmt.Sprintf("<REPO> %s\n", pr.Repo.FullName))
    buf.WriteString(fmt.Sprintf("<PR_NUMBER> %d\n", pr.PullRequest.Number))
    buf.WriteString(fmt.Sprintf("<PR_TITLE> %s\n", pr.PullRequest.Title))

    // Description
    if pr.PullRequest.Body != "" {
        buf.WriteString("<PR_BODY>\n")
        buf.WriteString(pr.PullRequest.Body)
        buf.WriteString("\n</PR_BODY>\n")
    }

    // File changes
    if len(pr.Files) > 0 {
        buf.WriteString("<FILES>\n")
        for _, file := range pr.Files {
            buf.WriteString(fmt.Sprintf("<FILE path=\"%s\" status=\"%s\">\n", file.Filename, file.Status))
            if file.Patch != "" {
                buf.WriteString(file.Patch)
                buf.WriteString("\n")
            }
            buf.WriteString("</FILE>\n")
        }
        buf.WriteString("</FILES>\n")
    }

    // Comments
    if len(pr.Comments) > 0 {
        buf.WriteString("<COMMENTS>\n")
        for _, c := range pr.Comments {
            buf.WriteString(fmt.Sprintf("%s: %s\n", c.User, c.Body))
        }
        buf.WriteString("</COMMENTS>\n")
    }

    buf.WriteString("\n---\n\n")
    return buf.String()
}
```

## Rate Limiting & Optimization

### GitHub API Limits

- **Authenticated**: 5,000 requests/hour
- **Unauthenticated**: 60 requests/hour

### Strategies

1. **Token Rotation**: Use multiple tokens
2. **Caching**: Save responses, resume from checkpoint
3. **Conditional Requests**: Use ETags
4. **Pagination**: Fetch efficiently

### Rate Limiter Implementation

```go
type RateLimiter struct {
    remaining int
    reset     time.Time
    mu        sync.Mutex
}

func (rl *RateLimiter) Wait() {
    rl.mu.Lock()
    defer rl.mu.Unlock()

    if rl.remaining <= 10 {
        waitTime := time.Until(rl.reset)
        if waitTime > 0 {
            time.Sleep(waitTime)
        }
    }
}

func (rl *RateLimiter) Update(resp *http.Response) {
    rl.mu.Lock()
    defer rl.mu.Unlock()

    remaining := resp.Header.Get("X-RateLimit-Remaining")
    reset := resp.Header.Get("X-RateLimit-Reset")

    rl.remaining, _ = strconv.Atoi(remaining)
    resetUnix, _ := strconv.ParseInt(reset, 10, 64)
    rl.reset = time.Unix(resetUnix, 0)
}
```

## Data Quality

### Filtering

**Exclude**:
- Vendored code (`vendor/`, `third_party/`)
- Generated files (`*.pb.go`, `*.gen.go`)
- Non-Go files in Go repos
- Large binary files
- Test fixtures

```go
func shouldIncludeFile(path string) bool {
    excluded := []string{
        "vendor/",
        "third_party/",
        "node_modules/",
        ".git/",
        "testdata/",
    }

    for _, prefix := range excluded {
        if strings.HasPrefix(path, prefix) {
            return false
        }
    }

    // Check for generated files
    if strings.HasSuffix(path, ".pb.go") ||
       strings.HasSuffix(path, ".gen.go") ||
       strings.Contains(path, "generated") {
        return false
    }

    return strings.HasSuffix(path, ".go")
}
```

### Deduplication

```go
type SeenSet struct {
    repos map[string]bool
    prs   map[string]bool
}

func (s *SeenSet) AddRepo(fullName string) bool {
    if s.repos[fullName] {
        return false
    }
    s.repos[fullName] = true
    return true
}

func (s *SeenSet) AddPR(repo string, number int) bool {
    key := fmt.Sprintf("%s#%d", repo, number)
    if s.prs[key] {
        return false
    }
    s.prs[key] = true
    return true
}
```

## Persistence & Checkpointing

### JSONL Output

```go
func writeJSONL(records []PRRecord, path string) error {
    f, err := os.Create(path)
    if err != nil {
        return err
    }
    defer f.Close()

    encoder := json.NewEncoder(f)
    for _, record := range records {
        if err := encoder.Encode(record); err != nil {
            return err
        }
    }
    return nil
}
```

### Resume Support

```go
type Checkpoint struct {
    LastRepo   string    `json:"last_repo"`
    LastPR     int       `json:"last_pr"`
    Timestamp  time.Time `json:"timestamp"`
    TotalFetched int     `json:"total_fetched"`
}

func saveCheckpoint(cp Checkpoint, path string) error {
    data, _ := json.Marshal(cp)
    return os.WriteFile(path, data, 0644)
}

func loadCheckpoint(path string) (Checkpoint, error) {
    data, err := os.ReadFile(path)
    if err != nil {
        return Checkpoint{}, err
    }
    var cp Checkpoint
    json.Unmarshal(data, &cp)
    return cp, nil
}
```

## Running the Pipeline

### Basic Usage

```bash
# Build
go build -o bin/fetchdata ./cmd/fetchdata

# Run with environment variables
export GITHUB_TOKEN=your_token
export OUTPUT_DIR=data/processed
./bin/fetchdata
```

### Configuration File

Create `config.yaml`:

```yaml
github:
  token: ${GITHUB_TOKEN}
  repos:
    - golang/go
    - spf13/cobra
    - wailsapp/wails
  rate_limit: 1.0  # requests per second

output:
  dir: data/processed
  format: jsonl
  checkpoint: data/.checkpoint.json

filters:
  min_stars: 100
  licenses:
    - mit
    - apache-2.0
    - gpl-3.0
  exclude_patterns:
    - vendor/
    - third_party/
```

## Monitoring

### Progress Tracking

```go
type Progress struct {
    TotalRepos   int
    ProcessedRepos int
    TotalPRs     int
    Errors       int
}

func (p *Progress) Print() {
    fmt.Printf(
        "Progress: %d/%d repos, %d PRs, %d errors\n",
        p.ProcessedRepos, p.TotalRepos, p.TotalPRs, p.Errors,
    )
}
```

### Logging

```go
import "log/slog"

logger := slog.New(slog.NewJSONHandler(os.Stdout, nil))
logger.Info("fetching PR", "repo", repo, "number", prNum)
logger.Error("failed to fetch", "error", err)
```

## Best Practices

1. **Always authenticate** - Use GITHUB_TOKEN
2. **Respect rate limits** - Implement backoff
3. **Save checkpoints** - Resume on failure
4. **Validate licenses** - Only use allowed licenses
5. **Filter aggressively** - Remove vendor/generated
6. **Deduplicate** - Track seen repos/PRs
7. **Log everything** - Debug issues later
8. **Test with small set** - Validate before large run

## Troubleshooting

### API Rate Limit Exceeded

```bash
# Check current rate limit
curl -H "Authorization: Bearer $GITHUB_TOKEN" \
  https://api.github.com/rate_limit
```

**Solution**: Wait for reset or use multiple tokens

### Large Responses Timeout

**Solution**: Increase HTTP timeout, paginate

```go
client := &http.Client{Timeout: 60 * time.Second}
```

### Memory Issues

**Solution**: Process in batches, write to disk frequently

## Next Steps

After collecting data:
1. Review [Tokenizer Guide](TOKENIZER.md) to train tokenizer
2. Check [Training Guide](TRAINING.md) for model training
3. See [Architecture](ARCHITECTURE.md) for system overview

## Example Output

**data/processed/golang_go.jsonl**:

```json
{"repo":{"full_name":"golang/go","html_url":"https://github.com/golang/go","stars":120000,"license_key":"bsd-3-clause","fetched_at":"2025-11-06T12:00:00Z"},"pull_request":{"number":12345,"title":"runtime: fix nil pointer dereference","body":"This PR fixes...","state":"closed","author":"user1","created_at":"2024-01-01"},"comments":[{"user":"reviewer1","body":"LGTM"}],"files":[{"filename":"runtime/proc.go","status":"modified","patch":"@@ -100,6 +100,9 @@..."}],"tags":["MIX","CODE"]}
```

## Performance Tips

- Use goroutines for concurrent fetching
- Batch API requests where possible
- Use conditional requests (ETags)
- Compress stored data (gzip)
- Use SSD for intermediate storage
