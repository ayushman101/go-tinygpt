package tinyst

import (
	"fmt"
	"math"
	"math/rand"
)

type example struct {
	input  []int
	target []int
}

func prepareExamples(tokens []int, seqLen, stride int) []example {
	var examples []example
	for i := 0; i+seqLen+1 <= len(tokens); i += stride {
		examples = append(examples, example{
			input:  tokens[i : i+seqLen],
			target: tokens[i+1 : i+seqLen+1],
		})
	}
	return examples
}

func (m *Model) Train(tokens []int, epochs int, lr float64, logEvery int) error {
	seqLen := m.MaxSeqLen
	stride := seqLen / 2

	examples := prepareExamples(tokens, seqLen, stride)
	if len(examples) == 0 {
		return fmt.Errorf("not enough tokens (%d) for training with seqLen %d", len(tokens), seqLen)
	}

	fmt.Printf("Training on %d examples (seqLen=%d, stride=%d), lr=%.4f\n",
		len(examples), seqLen, stride, lr)

	for epoch := 0; epoch < epochs; epoch++ {
		rand.Shuffle(len(examples), func(i, j int) {
			examples[i], examples[j] = examples[j], examples[i]
		})

		var totalLoss float64
		for step, ex := range examples {
			cache, _, err := m.Forward(ex.input)
			if err != nil {
				return err
			}
			loss, grads := m.Backward(cache, ex.target)
			m.ApplyGradients(grads, lr)

			if math.IsNaN(loss) {
				return fmt.Errorf("loss is NaN at epoch %d step %d (consider lowering lr)", epoch, step)
			}

			totalLoss += loss

			if (step+1)%logEvery == 0 {
				fmt.Printf("epoch %d/%d step %d/%d loss %.4f\n",
					epoch+1, epochs, step+1, len(examples), totalLoss/float64(step+1))
			}
		}

		avgLoss := totalLoss / float64(len(examples))
		fmt.Printf("epoch %d/%d complete, avg loss %.4f\n", epoch+1, epochs, avgLoss)

		if err := m.Save(fmt.Sprintf("./artifacts/weights1.json")); err != nil {
			return err
		}
	}

	return nil
}