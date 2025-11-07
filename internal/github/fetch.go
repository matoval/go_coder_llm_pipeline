package github

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"

	"go_coder_llm_pipeline/internal/types"
)

func FetchPRData(repo string) ([]types.PRComment, error) {
	url := fmt.Sprintf("https://api.github.com/repos/%s/pulls?state=closed&per_page=50", repo)
	req, _ := http.NewRequestWithContext(context.Background(), "GET", url, nil)
	req.Header.Set("Accept", "application/vnd.github+json")

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	var prs []map[string]interface{}
	if err := json.NewDecoder(resp.Body).Decode(&prs); err != nil {
		return nil, err
	}

	var comments []types.PRComment
	for _, pr := range prs {
		comments = append(comments, types.PRComment{
			Author:  "unknown",
			Body:    fmt.Sprintf("%v", pr["title"]),
			File:    "",
			Changes: "",
		})
	}
	return comments, nil
}
