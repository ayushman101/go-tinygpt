package tinyst

// TODO: need to process input in max seq len batches
func (m *Model) Forward (input []int) ([][]float64, error) { // input is Id vector , returns logits
	// 1. create our token embedding matrix
	seq_len := len (input)
	input_embed := make ([][]float64, seq_len)

	for i:=0 ; i<seq_len; i++ {
		input_embed [i] = m.TokenEmbed [input[i]]
	}

	// 2. Add pos embed
	for i := range input_embed {
		for j := range input_embed[i] {
			input_embed[i][j] += m.PosEmbed[i][j]
		}
	}

	return input_embed, nil
}
