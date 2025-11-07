package tokenizer

import (
	"os"
	"strings"

	"go_coder_llm_pipeline/internal/types"
)

func SaveCorpus(comments []types.PRComment, path string) error {
	var builder strings.Builder
	for _, c := range comments {
		builder.WriteString("// Author: " + c.Author + "\n")
		builder.WriteString(c.Changes + "\n")
		builder.WriteString("// Comment: " + c.Body + "\n\n")
	}
	return os.WriteFile(path, []byte(builder.String()), 0644)
}
