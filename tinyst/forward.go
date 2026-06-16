package tinyst

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

		low += high 
	}

	return input_embed, nil
}
