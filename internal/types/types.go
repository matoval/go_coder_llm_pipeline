package types

import "time"

// RepoMeta contains repository metadata
type RepoMeta struct {
	FullName   string    `json:"full_name"`
	HTMLURL    string    `json:"html_url"`
	Stars      int       `json:"stars"`
	LicenseKey string    `json:"license_key"`
	FetchedAt  time.Time `json:"fetched_at"`
}

// PullRequest represents a GitHub pull request
type PullRequest struct {
	Number    int       `json:"number"`
	Title     string    `json:"title"`
	Body      string    `json:"body"`
	State     string    `json:"state"`
	Author    string    `json:"author"`
	CreatedAt time.Time `json:"created_at"`
	MergedAt  *time.Time `json:"merged_at,omitempty"`
	HTMLURL   string    `json:"html_url"`
}

// Comment represents a PR comment or review
type Comment struct {
	ID        int       `json:"id"`
	User      string    `json:"user"`
	Body      string    `json:"body"`
	CreatedAt time.Time `json:"created_at"`
	Path      string    `json:"path,omitempty"`      // For review comments
	DiffHunk  string    `json:"diff_hunk,omitempty"` // Code context
}

// PRFile represents a file changed in a PR
type PRFile struct {
	Filename  string `json:"filename"`
	Status    string `json:"status"` // added, removed, modified
	Additions int    `json:"additions"`
	Deletions int    `json:"deletions"`
	Patch     string `json:"patch"` // Unified diff
}

// PRRecord is the complete record for a pull request
type PRRecord struct {
	Repo        RepoMeta    `json:"repo"`
	PullRequest PullRequest `json:"pull_request"`
	Comments    []Comment   `json:"comments"`
	Files       []PRFile    `json:"files"`
	Tags        []string    `json:"tags"` // NAT, MIX, CODE
}

// Checkpoint stores progress for resumable fetching
type Checkpoint struct {
	LastRepo      string    `json:"last_repo"`
	LastPR        int       `json:"last_pr"`
	Timestamp     time.Time `json:"timestamp"`
	TotalFetched  int       `json:"total_fetched"`
	TotalErrors   int       `json:"total_errors"`
	ReposComplete []string  `json:"repos_complete"`
}

// PRComment represents a pull request comment with associated code changes
// Deprecated: Use Comment instead
type PRComment struct {
	Author  string `json:"author"`
	Body    string `json:"body"`
	File    string `json:"file"`
	Changes string `json:"changes"`
}
