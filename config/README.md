# Configuration Files

This directory contains configuration files for the data collection pipeline.

## Files

### repos.json
List of GitHub repositories to fetch data from. This file is populated by the discovery script.

**Format**: Simple JSON array of repository names in `owner/repo` format.

```json
[
  "golang/go",
  "kubernetes/kubernetes",
  "spf13/cobra"
]
```

**Usage**:
- Run `scripts/discover_repos.go` to populate this file with filtered repos
- The `cmd/fetchdata` tool reads from this file

**Size**: Expected to be 200 KB - 2 MB for 5000 repos (well within Git limits)

## Generating the Repository List

To populate `repos.json` with the top 5000 Go repositories:

```bash
# Set your GitHub token
export GITHUB_TOKEN=ghp_your_token_here

# Run the discovery script
go run scripts/discover_repos.go

# This will:
# 1. Search GitHub for Go repos with 100+ stars
# 2. Filter by open source licenses (MIT, Apache, BSD, GPL, MPL)
# 3. Filter by activity (updated in last 2 years)
# 4. Save filtered list to config/repos.json
```

## Filters Applied

The discovery script applies these filters:
- **Language**: Go
- **Stars**: >= 100 (quality threshold)
- **License**: MIT, Apache-2.0, BSD-3-Clause, GPL-3.0, MPL-2.0 (open source only)
- **Activity**: Updated within last 2 years
- **Excludes**: Forks, archived repos, massive monorepos (>1GB)

## Version Control

This file **IS committed to Git** because:
- Small size (1-2 MB for 5000 repos)
- Acts as pipeline configuration
- Allows reproducibility of data collection
- Easy to review changes via diffs
