package tokenizer

import (
	"fmt"
	"regexp"
)


type Pair struct {
	A, B int
}

// Byte Pair encoding tokenizer
type BPETokenizer struct {
	vocab     map[string]int
	idToToken map[int]string
	merges    map[Pair]int
	regex     *regexp.Regexp // pre tokenization regular expression
}

func NewBPE () *BPETokenizer {
	pattern := regexp.MustCompile (`(?i:'s|'t|'re|'ve|'m|'ll|'d)| ?\pL+| ?\pN+| ?[^\s\pL\pN]+|\s+`)
	bpe := &BPETokenizer {
		vocab:     make (map[string]int),
		idToToken: make (map[int]string),
		merges:    make (map[Pair]int),
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

func (bpe *BPETokenizer) Train (data string, trainingSize int) error {
	words := bpe.regex.FindAllString (data, -1)

	// build a map of unique bytes or tokens
	seen := make (map[byte]bool)
	for _, word := range words {
		for _, b := range []byte (word) {
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

	chunks := make ([][]int, len (words))

	for i, word := range words {
		var chunk []int
		for _, b := range []byte (word) {
			hex := fmt.Sprintf ("%02x", b)
			id := bpe.vocab[hex]
			chunk = append (chunk, id)
		}
		chunks[i] = chunk
	}

	fmt.Println ("Length of initial chunks", len (chunks))

	for nextId < trainingSize {
		// start merging
		merges := make (map[Pair]int)

		var bestPair Pair
		bestFreq := 0

		for _, chunk := range chunks {
			if len (chunk) > 1 {
				for i := 0; i < (len (chunk) - 1); i++ {
					pair := Pair {chunk [i], chunk[i+1]}
					merges [pair]++
					if merges [pair] > bestFreq {
						bestPair = pair
						bestFreq = merges[pair]
					}
				}
			}
		}

		if bestFreq <= 0 {
			break
		}

		lefthex := bpe.idToToken [bestPair.A]
		rightHex := bpe.idToToken [bestPair.B]
		mergedHex := lefthex + rightHex

		bpe.vocab[mergedHex] = nextId
		bpe.idToToken[nextId] = mergedHex
		bpe.merges[bestPair] = nextId

		// Apply the merge to all chunks
		for ci, chunk := range chunks {
			var newChunk []int
			for i := 0; i < len(chunk); i++ {
				if i+1 < len(chunk) && chunk[i] == bestPair.A && chunk[i+1] == bestPair.B {
					newChunk = append(newChunk, nextId)
					i++ // skip the second element of the pair
				} else {
					newChunk = append(newChunk, chunk[i])
				}
			}
			chunks[ci] = newChunk
		}

		nextId++
	}

	fmt.Println ("BPE vocab", bpe.vocab)
	fmt.Println ("Length of final chunks", len (chunks))
	fmt.Println ("Lenght of final Vocab", len (bpe.vocab))

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

