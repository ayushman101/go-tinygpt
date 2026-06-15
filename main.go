package main

import (
	"fmt"
	"os"
	"go-gpt/tinyst"
	"go-gpt/tokenizer"
)

const (
	inputFilePath string = "./artifacts/input.txt"
	tokenizerFilePath string = "./artifacts/tokenizer_vocab.json"
	weightFilePath string = "./artifacts/weights1.json"
	VOCAB_SIZE int = 256
)

func main () {
	bpe := tokenizer.NewBPE ()

	fmt.Println ("New bpe created")
	fmt.Println ("Bpe regex : ", bpe.GetRegex () )

	bpe.Load (tokenizerFilePath)

	data, err := os.ReadFile (inputFilePath)
	if err != nil {
		fmt.Printf ("failed to read input file : %v", err)
		panic (err)
	}

	text := string (data)

	cfg := tinyst.Config {
		VocabSize : 256,
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

	err = m.Init (weightFilePath)
	if err != nil {
		fmt.Printf ("Failed to init model %v", err)
		panic (err)
	}

	fmt.Println (" model initialized successfully")

	// err = m.Save (weightFilePath)
	// if err != nil {
	// 	fmt.Printf ("Failed to save the model %v", err)
	// 	panic (err)
	// }

	fmt.Println (" input text:", text[:500])

	encoding := bpe.Encode (text[:500])

	fmt.Println (" encoding :", encoding, " \nlength : ", len (encoding))
	embed, err := m.Forward (encoding)

	if err != nil {
		fmt.Println ("failed forward pass", err)
		os.Exit (1)
	}

	fmt.Println ("length of input embedding :", len (embed))
}
