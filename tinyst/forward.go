package tinyst

import (
	"math"
)

func (m *Model) Forward(input []int) (*TrainCache, [][]float64, error) {
	cache := &TrainCache{
		InputIDs: input,
		Blocks:   make([]BlockCache, len(m.TBlocks)),
	}

	low := 0
	high := 0
	seqLen := len(input)
	var allLogits [][]float64

	for low < seqLen {
		if seqLen-low > m.MaxSeqLen {
			high = m.MaxSeqLen
		} else {
			high = seqLen - low
		}

		window := input[low : low+high]

		x := make([][]float64, len(window))
		for i, id := range window {
			row := make([]float64, m.DModel)
			copy(row, m.TokenEmbed[id])
			x[i] = row
		}

		for i := range x {
			for j := range x[i] {
				x[i][j] += m.PosEmbed[i][j]
			}
		}

		for bi, block := range m.TBlocks {
			bc := &cache.Blocks[bi]

			// === Attention sublayer ===
			bc.LN1In = CopyMat(x)
			normed := applyLayerNorm(x, block.LN1)
			bc.LN1Out = normed

			bc.Heads = make([]HeadCache, len(block.Attention.Heads))
			var concat [][]float64

			for hi, ah := range block.Attention.Heads {
				hc := &bc.Heads[hi]

				Q, err := Mult(normed, ah.W_Q)
				if err != nil {
					return nil, nil, err
				}
				K, err := Mult(normed, ah.W_K)
				if err != nil {
					return nil, nil, err
				}
				V, err := Mult(normed, ah.W_V)
				if err != nil {
					return nil, nil, err
				}

				Kt := transpose(K)
				W, err := Mult(Q, Kt)
				if err != nil {
					return nil, nil, err
				}

				dHead := m.DModel / m.NumHeads
				scale := 1.0 / math.Sqrt(float64(dHead))
				for i := range W {
					for j := range W[i] {
						W[i][j] *= scale
					}
				}

				for i := 0; i < len(W); i++ {
					for j := i + 1; j < len(W[i]); j++ {
						W[i][j] = -1e9
					}
				}

				for i := range W {
					SoftMax(W[i])
				}

				hc.Q = Q
				hc.K = K
				hc.V = V
				hc.W = CopyMat(W)

				headOut, err := Mult(W, V)
				if err != nil {
					return nil, nil, err
				}

				if hi == 0 {
					concat = headOut
				} else {
					concat, err = Concat(concat, headOut)
					if err != nil {
						return nil, nil, err
					}
				}
			}

			bc.Concat = CopyMat(concat)
			attentionOut, err := Mult(concat, block.Attention.W_O)
			if err != nil {
				return nil, nil, err
			}

			for i := range x {
				for j := range x[i] {
					x[i][j] += attentionOut[i][j]
				}
			}

			// === FFN sublayer ===
			bc.LN2In = CopyMat(x)

			normed = applyLayerNorm(x, block.LN2)
			bc.LN2Out = normed

			ffn1, err := Mult(normed, block.FFN.W1)
			if err != nil {
				return nil, nil, err
			}
			for i := range ffn1 {
				for j := range ffn1[i] {
					ffn1[i][j] += block.FFN.B1[j]
				}
			}
			bc.FFN1 = CopyMat(ffn1)

			ReLU(ffn1)
			bc.FFN1R = CopyMat(ffn1)

			ffn2, err := Mult(ffn1, block.FFN.W2)
			if err != nil {
				return nil, nil, err
			}
			for i := range ffn2 {
				for j := range ffn2[i] {
					ffn2[i][j] += block.FFN.B2[j]
				}
			}

			for i := range x {
				for j := range x[i] {
					x[i][j] += ffn2[i][j]
				}
			}
		}

		cache.FinalEmb = CopyMat(x)
		logits, err := Mult(x, m.Unembed)
		if err != nil {
			return nil, nil, err
		}

		allLogits = append(allLogits, logits...)
		low += high
	}

	cache.Logits = allLogits
	return cache, allLogits, nil
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
