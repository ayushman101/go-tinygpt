package main

import (
	"fmt"
	"os"
	// "go-gpt/tinyst"
	"go-gpt/tokenizer"
)

const (
	inputFilePath string = "input.txt"
	// weightFilePath string = "weights1.json"
)

func main () {
	bpe := tokenizer.NewBPE ()

	fmt.Println ("New bpe created")
	fmt.Println ("Bpe regex : ", bpe.GetRegex () )

	data, err := os.ReadFile (inputFilePath)
	if err != nil {
		fmt.Printf ("failed to read input file : %v", err)
		panic (err)
	}

	text := string (data)

	bpe.Train (text, 256)

	fmt.Println ("first 100 characters", text[:100])

	encoding := bpe.Encode (text[:100])

	fmt.Println ("encoding" , encoding)

	decoding := bpe.Decode (encoding)

	fmt.Println ("decoding", decoding)
	// cfg := tinyst.Config {
	// 	VocabSize : 70,
	// 	DModel : 64,
	// 	MaxSeqLen : 64,
	// 	NumHeads : 4,
	// 	NumLayers : 3,
	// 	FFNHidden : 256,
	// }
	//
	// m, err := tinyst.NewModel (cfg)
	//
	// if err != nil {
	// 	fmt.Printf ("Failed to create new model %v", err)
	// 	panic (err)
	// }
	//
	// err = m.Init (weightFilePath)
	// if err != nil {
	// 	fmt.Printf ("Failed to init model %v", err)
	// 	panic (err)
	// }
	//
	// fmt.Println (" model initialized successfully")
	//
	// err = m.Save (weightFilePath)
	// if err != nil {
	// 	fmt.Printf ("Failed to save the model %v", err)
	// 	panic (err)
	// }
}
