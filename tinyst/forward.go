package tinyst

import (
	"fmt"
	"math"
)

func (m *Model) Forward (input []int) ([][]float64, error) { // input is Id vector , returns logits
	// 1. create our token embedding matrix
	low := 0
	high := 0
	seq_len := len (input)

	var input_embed [][]float64

	// final logits to be returned
	var allLogits [][]float64

	for low < seq_len {
		if (seq_len - low) > m.MaxSeqLen {
			high = m.MaxSeqLen
		} else {
			high = seq_len - low
		}

		window := input[low:low + high]

		input_embed = make ([][]float64, len (window))

		// get the token vector from Id
		for i, id := range window {
			row := make([]float64, m.DModel)
			copy(row, m.TokenEmbed[id])
			input_embed[i] = row
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
			var concat [][]float64

			// keep a copy of original input matrix
			x := CopyMat (input_embed)

			// first layer normalization
			input_embed = applyLayerNorm (input_embed, t.LN1)
			fmt.Println ("embed dimensions after first normalization : ", len (input_embed), " ", len (input_embed[0]))

			for index, ah:= range t.Attention.Heads {
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

				// masking the upper triangle
				// A token can only affect tokens coming before it
				for i:=0; i< len (W); i++ {
					for j:=i+1 ; j< len (W[i]) ; j++ {
						W[i][j] = -1e9
					}
				}

				// apply softmax
				for i := range W {
					SoftMax (W[i])
				}

				headOut, err := Mult (W, V)
				if err != nil {
					return nil, err
				}

				if index == 0 {
					concat = headOut
				} else {
					concat, err = Concat (concat, headOut)
					if err != nil {
						return nil, err
					}
				}
			}

			attentionOut, err := Mult (concat, t.Attention.W_O)  // W_O read-only
			if err != nil {
				return nil, err
			}

			// add the attention output to original input embedding
			Add (x, attentionOut)
			fmt.Println ("input embed after adding attention output", len (x), " ", len (x[0]))

			// new snapshot
			input_embed = CopyMat (x)

			// Next is Layer normalization
			normal := applyLayerNorm (input_embed, t.LN2)
			fmt.Println ("embed after normalization dimensions : ", len (normal), " ", len (normal[0]))
	
			// Feed Forward
			// multiply with Weight matrix 1
			ffn1, err := Mult (normal, t.FFN.W1)
			if err != nil {
				return nil, err
			}

			// add bias 1
			for i := range ffn1 {
				for j:= range ffn1[i] {
					ffn1[i][j] += t.FFN.B1[j]
				}
			}

			// apply relu
			ReLU (ffn1)
	
			// multiply weight matrix 2
			ffn2, err := Mult (ffn1, t.FFN.W2)
			if err != nil {
				return nil, err
			}

			// add bias 1
			for i := range ffn2 {
				for j:= range ffn2[i] {
					ffn2[i][j] += t.FFN.B2[j]
				}
			}

			// add the ffn2 to input embedding
			Add (input_embed, ffn2)
			fmt.Println ("input embed after adding feed forward 2 layer output", len (input_embed), " ", len (input_embed[0]))
		}

		// get the logits
		logits, err:= Mult(input_embed, m.Unembed)

		if err != nil {
			return nil, err
		}

		allLogits = append(allLogits, logits...)

		low += high
	}

	return allLogits, nil
}


func applyLayerNorm(x [][]float64, ln LayerNormal) [][]float64 {
    result := CopyMat(x)
    d := len(result[0])
    for i := range result {
        var mean float64
        for _, v := range result[i] {
            mean += v
        }
        mean /= float64(d)

        var variance float64
        for _, v := range result[i] {
            diff := v - mean
            variance += diff * diff
        }
        variance /= float64(d)

        for j := range result[i] {
            result[i][j] = (result[i][j] - mean) / math.Sqrt(variance+1e-5)
            result[i][j] = result[i][j]*ln.Gamma[j] + ln.Beta[j]
        }
    }
    return result
}
