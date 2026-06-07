package tinyst

type Config struct {
	VocabSize int
	MaxSeqLen int
	DModel    int
	NumHeads  int
	NumLayers int
	FFNHidden int
}

type AttentionHead struct {
	W_Q [][]float64 // d_model*d_heads
	W_K [][]float64 // d_model*d_heads
	W_V [][]float64 // d_model*d_heads
}

type MultiHeadAttention struct {
	Heads []AttentionHead // NumHeads
	W_O [][]float64       // d_model * d_model
}

type FeedForward struct {
	W1 [][]float64 // d_model*FFNHidden
	B1 []float64   // FFNHidden
	W2 [][]float64 // FFNHidden*d_model
	B2 []float64   // d_model
}

type LayerNormal struct {
	Gamma []float64 // d_model
	Beta  []float64 // d_model
}

type Transformer struct {
	Attention MultiHeadAttention
	LN1 LayerNormal
	FFN FeedForward
	LN2 LayerNormal
}

type Model struct {
	Config
	TokenEmbed [][]float64 // VocabSize * d_model
	PosEmbed   [][]float64 // MaxSeqLen * d_model
	TBlocks    []Transformer // length = NumLayers
	Unembed    [][]float64 // d_model * VocabSize
}

func NewModel (cfg Config) (*Model, error) {
	model := &Model {
		Config : cfg,
	}

	// allocate TokenEmbed
	model.TokenEmbed = make([][]float64, cfg.VocabSize)
	for i := range model.TokenEmbed {
		model.TokenEmbed[i] = make ([]float64, cfg.DModel)
	}

	// allocate PosEmbed
	model.PosEmbed = make([][]float64, cfg.MaxSeqLen)
	for i:= range model.PosEmbed {
		model.PosEmbed[i] = make([]float64, cfg.DModel)
	}

	// allocate Unembed
	model.Unembed = make([][]float64, cfg.DModel)
	for i:= range model.Unembed {
		model.Unembed[i] = make([]float64, cfg.VocabSize)
	}

	// allocate transformer
	model.TBlocks = make([]Transformer, cfg.NumLayers)
	dhead := cfg.DModel / cfg.NumHeads
	for i:= range model.TBlocks {
		block := &model.TBlocks[i]

		block.Attention.Heads = make([]AttentionHead, cfg.NumHeads)
		for l := range block.Attention.Heads {
			Head := block.Attention.Heads[l]
			Head.W_Q = make2D (cfg.DModel, dhead)
			Head.W_K = make2D (cfg.DModel, dhead)
			Head.W_V = make2D (cfg.DModel, dhead)
		}
		block.Attention.W_O = make2D (cfg.DModel, cfg.DModel)

		// allocate block layer normailization
		block.LN1.Gamma = make ([]float64, cfg.DModel)
		block.LN1.Beta = make ([]float64, cfg.DModel)

		// allocate FeedForward
		block.FFN.W1 =  make2D(cfg.DModel,cfg.FFNHidden)
		block.FFN.B1 =  make ([]float64, cfg.FFNHidden)
		block.FFN.W2 =  make2D(cfg.FFNHidden, cfg.DModel)
		block.FFN.B2 =  make ([]float64, cfg.DModel)

		// allocate block layer normailization 2
		block.LN2.Gamma = make ([]float64, cfg.DModel)
		block.LN2.Beta = make ([]float64, cfg.DModel)
	}
	return model, nil
}

// Helper to allocate a rows×cols 2D slice
func make2D (rows, cols int) [][]float64 {
    mat := make([][]float64, rows)
    for i := range mat {
        mat[i] = make([]float64, cols)
    }
    return mat
}
