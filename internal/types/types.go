package types

// PRComment represents a pull request comment with associated code changes
type PRComment struct {
	Author  string `json:"author"`
	Body    string `json:"body"`
	File    string `json:"file"`
	Changes string `json:"changes"`
}
