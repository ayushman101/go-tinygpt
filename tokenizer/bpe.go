package tokenizer

import (
	"fmt"
	"regexp"
)


type Pair struct {
	A, B byte
}

type BPETokenizer struct {
	vocab     map[string]int
	idToToken map[int]string
	merges    [][2]int
	regex     *regexp.Regexp // pre tokenization regular expression
}

func NewBPE () *BPETokenizer {
	pattern := regexp.MustCompile (`(?i:'s|'t|'re|'ve|'m|'ll|'d)| ?\pL+| ?\pN+| ?[^\s\pL\pN]+|\s+`)
	bpe := &BPETokenizer {
		vocab:     make(map[string]int),
		idToToken: make (map[int]string),
		merges:    [][2]int{},
		regex:     pattern,
	}
	return bpe
}

func (bpe *BPETokenizer) GetRegex () *regexp.Regexp {
	return bpe.regex
}

func (bpe *BPETokenizer) VocabSize () int {
	return len (bpe.vocab)
}

func (bpe *BPETokenizer) Train (data string) error {
	words := bpe.regex.FindAllString (data, -1)

	chunks := make ([][]byte, len(words))

	for i, word := range words {
		chunks[i] =  []byte(word)
	}

	// build a map of unique bytes or tokens
	seen := make (map[byte]bool)
	for _, chunk := range chunks {
		for _, b := range chunk {
			seen[b] = true
		}
	}

	// build initial vocab and idToToken
	nextId := 0
	for b := range seen {
		hex := fmt.Sprintf ("%02x", b)
		bpe.vocab[hex] = nextId
		bpe.idToToken [nextId] = hex
		nextId++
	}

	// start merging
	merges := make (map[Pair]int)

	for _, chunk := range chunks {
		if len (chunk) > 1 {
			for i := 0; i < (len (chunk) - 1); i++ {
				merges [Pair {chunk [i], chunk[i+1]}]++
			}
		}
	}

	fmt.Println ("BPE Merges : ", merges)

	return nil
}

func (bpe *BPETokenizer) Encode (input string) []int {
	return nil
}

func (bpe *BPETokenizer) Decode ([]int) string {
	return "" 
}

func (bpe *BPETokenizer) Save (path string) error {
	return nil
}

func (bpe *BPETokenizer) Load (path string) error {
	return nil
}

