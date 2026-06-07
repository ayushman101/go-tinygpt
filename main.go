package main

import (
	"fmt"
	"os"
	"go-gpt/tinyst"
)

const (
	inputFilePath string = "input.txt"
	weightFilePath string = "weights1.json"
)

func main () {
	_, err := os.ReadFile (inputFilePath)
	if err != nil {
		fmt.Printf ("failed to read input file : %v", err)
		panic (err)
	}

	cfg := tinyst.Config {
		VocabSize : 70,
		DModel : 64,
		MaxSeqLen : 64,
		NumHeads : 4,
		NumLayers : 3,
		FFNHidden : 256,
	}

	m, err := tinyst.NewModel (cfg)

	if err != nil {
		fmt.Printf ("Failed to create new model %v", err)
		panic (err)
	}

	err = m.Init ("")
	if err != nil {
		fmt.Printf ("Failed to init model %v", err)
		panic (err)
	}

	err = m.Save (weightFilePath)
	if err != nil {
		fmt.Printf ("Failed to save the model %v", err)
		panic (err)
	}
}
