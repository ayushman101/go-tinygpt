package main

import (
	"fmt"
	"os"
)

const (
	inputFilePath string = "input.txt"
)

func main () {
	// TODO: read the input file and store in a string
	data, err := os.ReadFile (inputFilePath)
	if err != nil {
		fmt.Printf ("failed to read input file : %v", err)
		panic (err)
	}

	fmt.Println (string (data[:1000]))
}
