package main

import (
	"log"
	"go_coder_llm_pipeline/internal/github"
	"go_coder_llm_pipeline/internal/tokenizer"
)

func main() {
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

	log.Println("All repos processed.")
}
