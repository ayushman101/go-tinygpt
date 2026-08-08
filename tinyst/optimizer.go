package tinyst

func sgdStep2D(params, grads [][]float64, lr float64) {
	for i := range params {
		for j := range params[i] {
			params[i][j] -= lr * grads[i][j]
		}
	}
}

func sgdStep1D(params, grads []float64, lr float64) {
	for i := range params {
		params[i] -= lr * grads[i]
	}
}

func (m *Model) ApplyGradients(grads *Gradients, lr float64) {
	sgdStep2D(m.TokenEmbed, grads.TokenEmbed, lr)
	sgdStep2D(m.PosEmbed, grads.PosEmbed, lr)
	sgdStep2D(m.Unembed, grads.Unembed, lr)

	for i := range m.TBlocks {
		block := &m.TBlocks[i]
		bg := &grads.Blocks[i]

		for h := range block.Attention.Heads {
			head := &block.Attention.Heads[h]
			hg := &bg.Heads[h]
			sgdStep2D(head.W_Q, hg.W_Q, lr)
			sgdStep2D(head.W_K, hg.W_K, lr)
			sgdStep2D(head.W_V, hg.W_V, lr)
		}

		sgdStep2D(block.Attention.W_O, bg.W_O, lr)
		sgdStep2D(block.FFN.W1, bg.W1, lr)
		sgdStep2D(block.FFN.W2, bg.W2, lr)
		sgdStep1D(block.FFN.B1, bg.B1, lr)
		sgdStep1D(block.FFN.B2, bg.B2, lr)
		sgdStep1D(block.LN1.Gamma, bg.LN1Gamma, lr)
		sgdStep1D(block.LN1.Beta, bg.LN1Beta, lr)
		sgdStep1D(block.LN2.Gamma, bg.LN2Gamma, lr)
		sgdStep1D(block.LN2.Beta, bg.LN2Beta, lr)
	}
}