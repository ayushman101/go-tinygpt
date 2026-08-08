package main

import (
	"flag"
	"fmt"
	"os"
	"go-gpt/tinyst"
	"go-gpt/tokenizer"
)

const (
	inputFilePath     string = "./artifacts/input.txt"
	tokenizerFilePath string = "./artifacts/tokenizer_vocab.json"
	weightFilePath    string = "./artifacts/weights1.json"
	VOCAB_SIZE        int    = 256
)

func main() {
	trainFlag := flag.Bool("train", false, "train the model")
	epochsFlag := flag.Int("epochs", 10, "number of training epochs")
	lrFlag := flag.Float64("lr", 0.001, "learning rate")
	logEveryFlag := flag.Int("log-every", 100, "log loss every N steps")
	flag.Parse()

	bpe := tokenizer.NewBPE()
	bpe.Load(tokenizerFilePath)

	data, err := os.ReadFile(inputFilePath)
	if err != nil {
		fmt.Printf("failed to read input file : %v", err)
		panic(err)
	}
	text := string(data)

	cfg := tinyst.Config{
		VocabSize: VOCAB_SIZE,
		DModel:    64,
		MaxSeqLen: 64,
		NumHeads:  4,
		NumLayers: 3,
		FFNHidden: 256,
	}

	m, err := tinyst.NewModel(cfg)
	if err != nil {
		fmt.Printf("Failed to create new model %v", err)
		panic(err)
	}

	err = m.Init(weightFilePath)
	if err != nil {
		fmt.Printf("No weights file found, using random init: %v\n", err)
		err = m.Init("")
		if err != nil {
			fmt.Printf("Failed to init model %v", err)
			panic(err)
		}
	}

	fmt.Println("model initialized successfully")

	if *trainFlag {
		encoding := bpe.Encode(text)
		fmt.Printf("encoded text to %d tokens\n", len(encoding))
		if err := m.Train(encoding, *epochsFlag, *lrFlag, *logEveryFlag); err != nil {
			fmt.Printf("training failed: %v", err)
			os.Exit(1)
		}
		if err := m.Save(weightFilePath); err != nil {
			fmt.Printf("Failed to save the model %v", err)
			os.Exit(1)
		}
		fmt.Println("training complete, weights saved to", weightFilePath)
		return
	}

	encoding := bpe.Encode(text[:500])
	fmt.Println("encoding length:", len(encoding))

	trainInput := encoding[:cfg.MaxSeqLen]
	trainTargets := encoding[1 : cfg.MaxSeqLen+1]
	cache, _, err := m.Forward(trainInput)
	if err != nil {
		fmt.Println("failed forward for training", err)
		os.Exit(1)
	}
	loss, grads := m.Backward(cache, trainTargets)
	m.ApplyGradients(grads, 0.001)

	fmt.Printf("loss on test batch: %.4f\n", loss)
	fmt.Println("gradients computed: TokenEmbed", len(grads.TokenEmbed), "x", len(grads.TokenEmbed[0]))
}
