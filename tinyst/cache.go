package tinyst

type TrainCache struct {
	InputIDs []int
	FinalEmb [][]float64
	Logits   [][]float64
	Blocks   []BlockCache
}

type BlockCache struct {
	LN1In  [][]float64
	LN1Out [][]float64
	Heads  []HeadCache
	Concat [][]float64
	LN2In  [][]float64
	LN2Out [][]float64
	FFN1   [][]float64
	FFN1R  [][]float64
}

type HeadCache struct {
	Q [][]float64
	K [][]float64
	V [][]float64
	W [][]float64
}
