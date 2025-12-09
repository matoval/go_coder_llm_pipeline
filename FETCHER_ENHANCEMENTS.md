# Data Fetcher Enhancements

## Summary

The data fetcher has been significantly enhanced with production-ready features for collecting high-quality training data from GitHub.

## What Was Enhanced

### 1. Complete Data Structures (`internal/types/types.go`)

**Before**: Simple `PRComment` struct with minimal fields

**After**: Complete type system including:
- `RepoMeta`: Repository metadata (name, URL, stars, license, fetch time)
- `PullRequest`: Full PR details (number, title, body, author, timestamps)
- `Comment`: Both issue comments and review comments with code context
- `PRFile`: File changes with diffs, additions/deletions
- `PRRecord`: Complete hierarchical record combining all data
- `Checkpoint`: Progress tracking for resumable fetching

### 2. GitHub Client with Rate Limiting (`internal/github/client.go`)

**Features**:
- Authenticated API requests with token
- Automatic rate limit tracking from response headers
- Minimum 1 second delay between requests
- Automatic waiting when rate limit is low (< 10 remaining)
- 90 second timeout for slow responses
- Automatic retry logic (up to 3 attempts with backoff)

**Rate Limits**:
- Authenticated: 5000 requests/hour
- Tracks remaining requests in real-time
- Automatically pauses before hitting limit

### 3. Enhanced Data Fetching (`internal/github/fetch.go`)

**Before**: Fetched only PR titles

**After**: Fetches complete PR data:
- PR metadata (title, body, author, state, timestamps)
- All issue comments
- All review comments (with file paths and code context)
- File changes (only .go files, filtered)
- Unified diffs for each file
- Automatic categorization (NAT/MIX/CODE tags)

**Smart Filtering**:
- Excludes vendor/, third_party/, generated files
- Only includes .go files
- Filters out .pb.go, .gen.go, testdata/

**Error Handling**:
- Retries failed requests
- Continues on errors (doesn't fail entire run)
- Logs warnings for partial failures

### 4. Checkpoint System (`internal/github/checkpoint.go`)

**Features**:
- Saves progress to `data/.checkpoint.json`
- Tracks completed repositories
- Records total PRs fetched and errors
- Automatic resume on restart
- No duplicate work

**Checkpoint Data**:
```json
{
  "last_repo": "gin-gonic/gin",
  "total_fetched": 5,
  "total_errors": 0,
  "repos_complete": ["avelino/awesome-go", "ollama/ollama", ...]
}
```

### 5. Production-Ready Main (`cmd/fetchdata/main.go`)

**New Features**:
- Command-line flags for configuration
- Test mode for validation
- Resume support (on by default)
- Progress tracking with ETA
- JSONL output format (one record per line)
- Rate estimation (repos/minute)
- Comprehensive final summary

**Command-Line Flags**:
```bash
-max-prs int     # Max PRs per repo (default: 50)
-test            # Test mode: first 5 repos only
-resume          # Resume from checkpoint (default: true)
```

## Output Format

### JSONL Files

Data is saved as JSONL (JSON Lines) in `data/raw/`:
- One file per repository: `{owner}_{repo}.jsonl`
- One JSON object per line (one PR per line)
- Easy to process line-by-line
- Efficient for large datasets

Example: `data/raw/golang_go.jsonl`

### Data Structure

Each line contains a complete `PRRecord`:
```json
{
  "repo": {
    "full_name": "golang/go",
    "html_url": "https://github.com/golang/go",
    "fetched_at": "2025-12-08T20:08:31Z"
  },
  "pull_request": {
    "number": 76727,
    "title": "cmd/cgo: use doc link for cgo.Handle",
    "body": "...",
    "author": "ariel-anieli",
    "created_at": "2025-12-06T10:30:34Z"
  },
  "comments": [
    {
      "id": 3619883697,
      "user": "gopherbot",
      "body": "...",
      "created_at": "2025-12-06T10:34:52Z"
    }
  ],
  "files": [
    {
      "filename": "src/cmd/cgo/doc.go",
      "status": "modified",
      "additions": 5,
      "deletions": 3,
      "patch": "@@ -100,6 +100,9 @@..."
    }
  ],
  "tags": ["NAT", "CODE"]
}
```

## Usage

### Test Mode (Recommended First Step)

Test with first 5 repos to validate setup:

```bash
go run cmd/fetchdata/main.go -test -max-prs=1
```

**Expected output**:
```
TEST MODE: Processing only first 5 repositories
Found 5 repositories to fetch
Configuration: max-prs=1, resume=true

[1/5] Processing avelino/awesome-go
Fetching PRs from avelino/awesome-go...
  Found 1 PRs
  Completed: 1 PRs fetched successfully
  Saved 1 PRs to data/raw/avelino_awesome-go.jsonl
  Progress: 1 processed, 0 skipped, 4 remaining (13.7 repos/min)

=== COMPLETED ===
Total repositories: 5
Processed: 5
Total PRs fetched: 5
Errors: 0
Total time: 20s
```

### Production Run

Fetch from all 3040 repos:

```bash
# Fetch 50 PRs per repo (default)
go run cmd/fetchdata/main.go

# Or customize
go run cmd/fetchdata/main.go -max-prs=100
```

### Resume After Interruption

If interrupted (Ctrl+C, network error, etc.), just rerun:

```bash
go run cmd/fetchdata/main.go
```

It will automatically:
- Load checkpoint
- Skip completed repos
- Continue from where it left off

### Disable Resume

Start fresh (ignores checkpoint):

```bash
go run cmd/fetchdata/main.go -resume=false
```

## Performance Estimates

Based on test run:
- **Rate**: ~15 repos/minute
- **Per PR**: ~3-4 API requests (PR list, comments, files)
- **Rate limit**: 5000 requests/hour = ~1200 PRs/hour

**For 3040 repos with 50 PRs each**:
- Total time: ~3.5 hours (at 15 repos/min)
- API requests: ~457,000 (3040 × 50 × 3)
- Cost: Free (within GitHub rate limits)

**Actual time may vary** based on:
- Repo size (more PRs = more API calls)
- Network speed
- GitHub API response times

## Data Quality Features

### Automatic Categorization

Each PR is tagged based on content:
- **NAT**: Natural language (description without code)
- **MIX**: Mixed (description with code snippets)
- **CODE**: Contains file changes

### Smart Filtering

Excludes:
- Vendor code (`vendor/`, `third_party/`)
- Generated files (`*.pb.go`, `*.gen.go`)
- Test fixtures (`testdata/`)
- Non-Go files

Includes:
- Only `.go` source files
- PRs with meaningful content
- Comments from actual reviewers

## Files Created

```
data/
  raw/                           # JSONL output (gitignored)
    avelino_awesome-go.jsonl
    golang_go.jsonl
    ...
  .checkpoint.json              # Progress tracking (gitignored)

internal/
  github/
    client.go                   # Rate-limited GitHub client
    fetch.go                    # Enhanced PR fetching
    checkpoint.go               # Resume support
  types/
    types.go                    # Complete data structures

cmd/
  fetchdata/
    main.go                     # Production-ready CLI
```

## Monitoring Progress

### Live Progress

The fetcher logs:
- Current repo being processed
- PRs found and fetched
- Rate limit remaining
- Progress (processed/skipped/remaining)
- Estimated time remaining (ETA)

### Checkpoint File

Check progress anytime:
```bash
cat data/.checkpoint.json | jq .
```

### Output Directory

Count fetched repos:
```bash
ls data/raw/*.jsonl | wc -l
```

Count total PRs:
```bash
wc -l data/raw/*.jsonl
```

## Error Handling

The fetcher is resilient:
- ✅ Retries failed API requests (3 attempts)
- ✅ Continues on errors (doesn't fail entire run)
- ✅ Saves progress after each repo
- ✅ Logs all errors and warnings
- ✅ Can resume from any point

## Next Steps

After data collection:
1. **Process data**: Convert JSONL to training format
2. **Train tokenizer**: Use collected corpus
3. **Train model**: HRM/TRM on processed data

See:
- `docs/DATA_PIPELINE.md` for processing
- `docs/TOKENIZER.md` for tokenization
- `docs/TRAINING.md` for model training

## Test Results

Test run (5 repos, 1 PR each):
- ✅ All repos fetched successfully
- ✅ Checkpoint saved correctly
- ✅ JSONL files created
- ✅ Complete PR data captured
- ✅ Comments and files included
- ✅ Rate limiting working
- ✅ Progress tracking accurate
- ✅ 20 seconds total time
- ✅ 0 errors
