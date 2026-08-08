package tinyst

import "math"

type Gradients struct {
	TokenEmbed [][]float64
	PosEmbed   [][]float64
	Blocks     []BlockGradients
	Unembed    [][]float64
}

type BlockGradients struct {
	Heads    []HeadGradients
	W_O      [][]float64
	W1       [][]float64
	B1       []float64
	W2       [][]float64
	B2       []float64
	LN1Gamma []float64
	LN1Beta  []float64
	LN2Gamma []float64
	LN2Beta  []float64
}

type HeadGradients struct {
	W_Q [][]float64
	W_K [][]float64
	W_V [][]float64
}

func newGradients(m *Model) *Gradients {
	g := &Gradients{
		TokenEmbed: make2D(m.VocabSize, m.DModel),
		PosEmbed:   make2D(m.MaxSeqLen, m.DModel),
		Unembed:    make2D(m.DModel, m.VocabSize),
		Blocks:     make([]BlockGradients, m.NumLayers),
	}
	dHead := m.DModel / m.NumHeads
	for i := range g.Blocks {
		bg := &g.Blocks[i]
		bg.Heads = make([]HeadGradients, m.NumHeads)
		for h := range bg.Heads {
			bg.Heads[h].W_Q = make2D(m.DModel, dHead)
			bg.Heads[h].W_K = make2D(m.DModel, dHead)
			bg.Heads[h].W_V = make2D(m.DModel, dHead)
		}
		bg.W_O = make2D(m.DModel, m.DModel)
		bg.W1 = make2D(m.DModel, m.FFNHidden)
		bg.B1 = make([]float64, m.FFNHidden)
		bg.W2 = make2D(m.FFNHidden, m.DModel)
		bg.B2 = make([]float64, m.DModel)
		bg.LN1Gamma = make([]float64, m.DModel)
		bg.LN1Beta = make([]float64, m.DModel)
		bg.LN2Gamma = make([]float64, m.DModel)
		bg.LN2Beta = make([]float64, m.DModel)
	}
	return g
}

func softmaxBackwardRow(W []float64, dW []float64) []float64 {
	sum := 0.0
	for k := range W {
		sum += W[k] * dW[k]
	}
	dScores := make([]float64, len(W))
	for j := range W {
		dScores[j] = W[j] * (dW[j] - sum)
	}
	return dScores
}

func layerNormBackward(dOut [][]float64, input [][]float64, ln LayerNormal) ([][]float64, []float64, []float64) {
	batchSize := len(dOut)
	d := len(dOut[0])

	dInput := make([][]float64, batchSize)
	dGamma := make([]float64, d)
	dBeta := make([]float64, d)

	for i := range dOut {
		var mean float64
		for _, v := range input[i] {
			mean += v
		}
		mean /= float64(d)

		var variance float64
		for _, v := range input[i] {
			diff := v - mean
			variance += diff * diff
		}
		variance /= float64(d)
		std := math.Sqrt(variance + 1e-5)

		xNorm := make([]float64, d)
		for j := range input[i] {
			xNorm[j] = (input[i][j] - mean) / std
		}

		for j := range dOut[i] {
			dBeta[j] += dOut[i][j]
			dGamma[j] += dOut[i][j] * xNorm[j]
		}

		var meanDy, meanDyXNorm float64
		for j := range dOut[i] {
			meanDy += dOut[i][j]
			meanDyXNorm += dOut[i][j] * xNorm[j]
		}
		meanDy /= float64(d)
		meanDyXNorm /= float64(d)

		dInput[i] = make([]float64, d)
		for j := range dOut[i] {
			dInput[i][j] = ln.Gamma[j] * (dOut[i][j] - meanDy - xNorm[j]*meanDyXNorm) / std
		}
	}

	return dInput, dGamma, dBeta
}

func (m *Model) Backward(cache *TrainCache, targets []int) (float64, *Gradients) {
	grads := newGradients(m)
	batchSize := len(cache.Logits)
	vocabSize := len(cache.Logits[0])
	dHead := m.DModel / m.NumHeads

	// ============== Step 1: Loss gradient (combined CE + softmax) ==============
	dLogits := make([][]float64, batchSize)
	var totalLoss float64
	for i := range cache.Logits {
		dLogits[i] = make([]float64, vocabSize)
		copy(dLogits[i], cache.Logits[i])
		SoftMax(dLogits[i])
		totalLoss -= math.Log(dLogits[i][targets[i]])
		dLogits[i][targets[i]] -= 1.0
	}

	// ============== Step 2: Unembed ==============
	unembedT := transpose(m.Unembed)
	dX, _ := Mult(dLogits, unembedT)

	finalEmbT := transpose(cache.FinalEmb)
	dUnembed, _ := Mult(finalEmbT, dLogits)
	grads.Unembed = dUnembed

	// ============== Step 3: Blocks (reverse order) ==============
	for bi := m.NumLayers - 1; bi >= 0; bi-- {
		bc := &cache.Blocks[bi]
		block := &m.TBlocks[bi]
		bg := &grads.Blocks[bi]

		// ============ FFN sublayer ============
		dFFN2 := CopyMat(dX)

		// W2 gradient
		ffn1rT := transpose(bc.FFN1R)
		dW2, _ := Mult(ffn1rT, dFFN2)
		bg.W2 = dW2

		// B2 gradient
		bg.B2 = SumRows(dFFN2)

		// ReLU backward: zero gradients where input was ≤ 0
		dReLUOut, _ := Mult(dFFN2, transpose(block.FFN.W2))
		for i := range dReLUOut {
			for j := range dReLUOut[i] {
				if bc.FFN1[i][j] <= 0 {
					dReLUOut[i][j] = 0
				}
			}
		}

		// W1 gradient
		ln2OutT := transpose(bc.LN2Out)
		dW1, _ := Mult(ln2OutT, dReLUOut)
		bg.W1 = dW1

		// B1 gradient
		bg.B1 = SumRows(dReLUOut)

		// Gradient for LN2 output (input to FFN1)
		dLN2Out, _ := Mult(dReLUOut, transpose(block.FFN.W1))

		// LayerNorm 2 backward
		dLN2In, dLN2Gamma, dLN2Beta := layerNormBackward(dLN2Out, bc.LN2In, block.LN2)
		bg.LN2Gamma = dLN2Gamma
		bg.LN2Beta = dLN2Beta

		// FFN residual: dX (unused) + dLN2In (through LN2+FFN path)
		for i := range dLN2In {
			for j := range dLN2In[i] {
				dLN2In[i][j] += dX[i][j]
			}
		}

		// ============ Attention sublayer ============
		dAttnOut := dLN2In

		// W_O gradient
		concatT := transpose(bc.Concat)
		dWO, _ := Mult(concatT, dAttnOut)
		bg.W_O = dWO

		// Concat backward: d_concat = dAttnOut · W_Oᵀ
		dConcat, _ := Mult(dAttnOut, transpose(block.Attention.W_O))

		// Per-head backward
		dLN1Out := make([][]float64, batchSize)
		for i := range dLN1Out {
			dLN1Out[i] = make([]float64, m.DModel)
		}

		for hi := len(block.Attention.Heads) - 1; hi >= 0; hi-- {
			hc := &bc.Heads[hi]
			ah := &block.Attention.Heads[hi]
			hg := &bg.Heads[hi]

			// Extract this head's gradient from dConcat
			start := hi * dHead
			end := (hi + 1) * dHead
			dHeadMat := make([][]float64, batchSize)
			for i := range dConcat {
				dHeadMat[i] = make([]float64, dHead)
				copy(dHeadMat[i], dConcat[i][start:end])
			}

			// dW = dHead · Vᵀ  (gradient for the attention weights after softmax)
			vT := transpose(hc.V)
			dW, _ := Mult(dHeadMat, vT)

			// dV = Wᵀ · dHead
			wT := transpose(hc.W)
			dV, _ := Mult(wT, dHeadMat)

			// Softmax backward for each row
			dScores := make([][]float64, batchSize)
			for i := range hc.W {
				dScores[i] = softmaxBackwardRow(hc.W[i], dW[i])
			}

			// Undo 1/√dHead scaling
			scale := 1.0 / math.Sqrt(float64(dHead))
			for i := range dScores {
				for j := range dScores[i] {
					dScores[i][j] *= scale
				}
			}

			// dQ = dScores · K
			dQ, _ := Mult(dScores, hc.K)

			// dK = dScoresᵀ · Q
			dScoresT := transpose(dScores)
			dK, _ := Mult(dScoresT, hc.Q)

			// W_Q, W_K, W_V gradients
			ln1OutT := transpose(bc.LN1Out)
			dW_Q, _ := Mult(ln1OutT, dQ)
			dW_K, _ := Mult(ln1OutT, dK)
			dW_V, _ := Mult(ln1OutT, dV)

			hg.W_Q = dW_Q
			hg.W_K = dW_K
			hg.W_V = dW_V

			// Accumulate gradient for LN1 output (backward through Q, K, V projections)
			dQx, _ := Mult(dQ, transpose(ah.W_Q))
			dKx, _ := Mult(dK, transpose(ah.W_K))
			dVx, _ := Mult(dV, transpose(ah.W_V))

			for i := range dLN1Out {
				for j := range dLN1Out[i] {
					dLN1Out[i][j] += dQx[i][j] + dKx[i][j] + dVx[i][j]
				}
			}
		}

		// LayerNorm 1 backward
		dLN1In, dLN1Gamma, dLN1Beta := layerNormBackward(dLN1Out, bc.LN1In, block.LN1)
		bg.LN1Gamma = dLN1Gamma
		bg.LN1Beta = dLN1Beta

		// Attention residual: dLN1In + dAttnOut (dAttnOut already includes FFN residual)
		for i := range dLN1In {
			for j := range dLN1In[i] {
				dLN1In[i][j] += dAttnOut[i][j]
			}
		}
		dX = dLN1In
	}

	// ============== Step 4: Embeddings ==============
	for i, id := range cache.InputIDs {
		for j := 0; j < m.DModel; j++ {
			grads.TokenEmbed[id][j] += dX[i][j]
		}
	}
	for i := 0; i < len(cache.InputIDs); i++ {
		for j := 0; j < m.DModel; j++ {
			grads.PosEmbed[i][j] += dX[i][j]
		}
	}

	return totalLoss/float64(batchSize), grads
}
