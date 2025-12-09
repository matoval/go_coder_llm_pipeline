package github

import (
	"encoding/json"
	"os"
	"time"

	"go_coder_llm_pipeline/internal/types"
)

const checkpointFile = "data/.checkpoint.json"

// LoadCheckpoint loads the checkpoint from disk
func LoadCheckpoint() (*types.Checkpoint, error) {
	data, err := os.ReadFile(checkpointFile)
	if err != nil {
		if os.IsNotExist(err) {
			// No checkpoint exists, return empty checkpoint
			return &types.Checkpoint{
				ReposComplete: []string{},
				Timestamp:     time.Now(),
			}, nil
		}
		return nil, err
	}

	var cp types.Checkpoint
	if err := json.Unmarshal(data, &cp); err != nil {
		return nil, err
	}

	return &cp, nil
}

// SaveCheckpoint saves the checkpoint to disk
func SaveCheckpoint(cp *types.Checkpoint) error {
	cp.Timestamp = time.Now()

	data, err := json.MarshalIndent(cp, "", "  ")
	if err != nil {
		return err
	}

	return os.WriteFile(checkpointFile, data, 0644)
}

// IsRepoComplete checks if a repo has been completely fetched
func IsRepoComplete(cp *types.Checkpoint, repo string) bool {
	for _, r := range cp.ReposComplete {
		if r == repo {
			return true
		}
	}
	return false
}

// MarkRepoComplete marks a repo as completely fetched
func MarkRepoComplete(cp *types.Checkpoint, repo string) {
	if !IsRepoComplete(cp, repo) {
		cp.ReposComplete = append(cp.ReposComplete, repo)
	}
	cp.LastRepo = repo
}
