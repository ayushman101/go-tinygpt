package tokenizer

import (
	"fmt"
	"regexp"
	"encoding/hex"
	"os"
	"encoding/json"
)


type Pair struct {
	A int `json:"left"`
	B int `json:"right"`
}

// so that json marshal can encode this to json
func (p Pair) MarshalText() ([]byte, error) {
    return []byte(fmt.Sprintf("%d,%d", p.A, p.B)), nil
}

func (p *Pair) UnmarshalText (text []byte) error {
	_, err:= fmt.Sscanf (string (text), "%d,%d", &p.A, &p.B)
	return err
}

// Byte Pair encoding tokenizer
type BPETokenizer struct {
	Vocab     map[string]int `json:"vocab"`
	IdToToken map[int]string `json:"idToToken"`
	Merges    map[Pair]int   `json:"merges"`
	regex     *regexp.Regexp `json:"-"`// pre tokenization regular expression . Skipped by json
}

func NewBPE () *BPETokenizer {
	pattern := regexp.MustCompile (`(?i:'s|'t|'re|'ve|'m|'ll|'d)| ?\pL+| ?\pN+| ?[^\s\pL\pN]+|\s+`)
	bpe := &BPETokenizer {
		Vocab:     make (map[string]int),
		IdToToken: make (map[int]string),
		Merges:    make (map[Pair]int),
		regex:     pattern,
	}
	return bpe
}

func (bpe *BPETokenizer) GetRegex () *regexp.Regexp {
	return bpe.regex
}

func (bpe *BPETokenizer) VocabSize () int {
	return len (bpe.Vocab)
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

	// build initial Vocab and IdToToken
	nextId := 0
	for b := range seen {
		hex := fmt.Sprintf ("%02x", b)
		bpe.Vocab[hex] = nextId
		bpe.IdToToken [nextId] = hex
		nextId++
	}

	chunks := make ([][]int, len (words))

	for i, word := range words {
		var chunk []int
		for _, b := range []byte (word) {
			hex := fmt.Sprintf ("%02x", b)
			id := bpe.Vocab[hex]
			chunk = append (chunk, id)
		}
		chunks[i] = chunk
	}

	fmt.Println ("Length of initial chunks", len (chunks))

	for nextId < trainingSize {
		// start merging
		Merges := make (map[Pair]int)

		var bestPair Pair
		bestFreq := 0

		for _, chunk := range chunks {
			if len (chunk) > 1 {
				for i := 0; i < (len (chunk) - 1); i++ {
					pair := Pair {chunk [i], chunk[i+1]}
					Merges [pair]++
					if Merges [pair] > bestFreq {
						bestPair = pair
						bestFreq = Merges[pair]
					}
				}
			}
		}

		if bestFreq <= 0 {
			break
		}

		lefthex := bpe.IdToToken [bestPair.A]
		rightHex := bpe.IdToToken [bestPair.B]
		mergedHex := lefthex + rightHex

		bpe.Vocab[mergedHex] = nextId
		bpe.IdToToken[nextId] = mergedHex
		bpe.Merges[bestPair] = nextId

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

	fmt.Println ("BPE Vocab", bpe.Vocab)
	fmt.Println ("Length of final chunks", len (chunks))
	fmt.Println ("Lenght of final Vocab", len (bpe.Vocab))

	return nil
}

func (bpe *BPETokenizer) Encode (input string) []int {
	words := bpe.regex.FindAllString (input, -1)

	chunks := make ([][]int, len (words))

	for i, word := range words {
		var chunk []int
		for _, b := range []byte (word) {
			hex := fmt.Sprintf ("%02x", b)
			id := bpe.Vocab[hex]
			chunk = append (chunk, id)
		}
		chunks[i] = chunk
	}

	for pair, id := range bpe.Merges {
		for ci, chunk := range chunks {
			var newChunk []int
			for i := 0; i< len(chunk) ; i++ {
				if i < (len(chunk) - 1) && chunk[i] == pair.A && chunk[i+1] == pair.B {
					newChunk = append (newChunk, id)
					i++
				} else {
					newChunk = append (newChunk, chunk [i])
				}
			}

			chunks[ci] = newChunk
		}
	}

	var result []int
	for _, chunk := range chunks {
		result = append (result, chunk...)
	}
	return result
}

func (bpe *BPETokenizer) Decode (input []int) string {
	var hexString string

	for _, id := range input {
		h := bpe.IdToToken [id]
		hexString += h
	}

	bytes, err := hex.DecodeString (hexString)
	if err != nil {
		fmt.Println (" failed to convert hexString to bytes", err)
		return ""
	}

	return string (bytes)
}

func (bpe *BPETokenizer) Load (path string) error {
	data, err := os.ReadFile (path);
	if err != nil {
		return err;
	}
	return json.Unmarshal (data, bpe)
}

func (bpe *BPETokenizer) Save (path string) error {
	if path == "" {
		return fmt.Errorf ("must give path of save file for saving weights")
	}
	data, err := json.MarshalIndent (bpe, "", " ")
	if err != nil {
		return err
	}

	return os.WriteFile (path, data, 0644)
}
