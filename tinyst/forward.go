package tinyst

import "fmt"

// TODO: need to process input in max seq len batches
func (m *Model) Forward (input []int) ([][]float64, error) { // input is Id vector , returns logits
	// 1. create our token embedding matrix
	low := 0
	high := 0
	seq_len := len (input)

	var input_embed [][]float64

	for low < seq_len {
		if (seq_len - low) > m.MaxSeqLen {
			high = m.MaxSeqLen
		} else {
			high = seq_len - low
		}

		window := input[low:low + high]

		input_embed = make ([][]float64, len (window))

		// get the token vector from Id
		for i:=0 ; i<len (window); i++ {
			input_embed [i] = m.TokenEmbed [window[i]]
		}

		// 2. Add pos embed
		for i := range input_embed {
			for j := range input_embed[i] {
				input_embed[i][j] += m.PosEmbed[i][j]
			}
		}

		// process Tblocks (transformer layers)
		for _, t := range m.TBlocks {
			// transformer attention heads
			for _, ah:= range t.Attention.Heads {
				// Query matrix
				Q, err := Mult (input_embed, ah.W_Q)
				if err != nil {
					return nil, err
				}

				fmt.Println ("Query matrix dimensions ", len (Q), " ", len (Q[0]))

				// Key matrix
				K, err := Mult (input_embed, ah.W_K)
				if err != nil {
					return nil, err
				}

				fmt.Println ("Key matrix dimensions ", len (K), " ", len (K[0]))

				// Value Matrix
				V, err := Mult (input_embed, ah.W_V)
				if err != nil {
					return nil, err
				}

				fmt.Println ("Value matrix dimensions ", len (V), " ", len (V[0]))

				// transpose the keys matrix
				Kt := transpose (K)

				// multiply query with key to get Weights
				W, err := Mult (Q, Kt)
				if err != nil {
					return nil, err
				}


				fmt.Println ("Weights matrix dimensions ", len (W), " ", len (W[0]))
			}
		}

		low += high 
	}

	return input_embed, nil
}
