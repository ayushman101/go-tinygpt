package tinyst

import (
	"math"
	"fmt"
)

func Mult (a, b [][]float64) ([][]float64, error) {
	a_rows := len (a)
	a_cols := len (a[0])

	b_rows := len (b)
	b_cols := len (b[0])

	if a_cols != b_rows {
		return nil, fmt.Errorf ("Cannot multiply incompatible matrices")
	}

	r := make ([][]float64, a_rows)
	for i:= range r {
		r[i] = make ([]float64, b_cols)
		for j := range a[i] {
			aij := a[i][j]
			if aij == 0 {
				continue
			}

			for k := range b_cols {
				r[i][k] += aij * b[j][k]
			}
		}
	}
	return r, nil
}

func Add (a, b [][]float64) error {
	if len (a) != len (b) || len (a[0]) != len (b[0]) {
		return fmt.Errorf ("cannot add incompatible matrices")
	}

	for i := range a {
		for j:= range a[i] {
			a[i][j] += b[i][j]
		}
	}

	return nil
}

func Scale (a [][]float64, s int) {
	for i:= range a {
		for j:= range a[i] {
			a[i][j] *= float64 (s)
		}
	}
}

// transpose returns Aᵀ. A is m×n, result is n×m.
func transpose(A [][]float64) [][]float64 {
    m, n := len(A), len(A[0])
    T := make([][]float64, n)
    for i := range T {
        T[i] = make([]float64, m)
    }
    for i := 0; i < m; i++ {
        for j := 0; j < n; j++ {
            T[j][i] = A[i][j]
        }
    }
    return T
}

// we subtract the max as it is possible that 
// one of the elements is very large and that would result in overflow on exp
func SoftMax (a []float64) {
	max := a[0]
	for _, v := range a[1:] {
		if v > max {
			max = v
		}
	}

	var sum float64 = 0.0
	for i := range a {
		a[i] = math.Exp (a[i] - max)
		sum += a[i]
	}

	for i := range a {
		a[i] = a[i] / sum
	}
}

func ReLU (a [][]float64) {
	for i := range a {
		for k := range a[i] {
			 if a[i][k] < 0 {
				 a[i][k] = 0
			 }
		}
	}
}

func AddBias (a, b []float64) ([]float64, error) {
	if len (a) != len (b) {
		return nil, fmt.Errorf ("incompatible vectors")
	}

	var c []float64

	for i := range a {
		c[i] = a[i] + b[i]
	}

	return c, nil
}

func CopyMat (a [][]float64) [][]float64 {
	b := make ([][]float64, len (a))

	for i:= range a {
		b[i] = make ([]float64, len (a[i]))
		for k := range a[i] {
			b[i][k] = a[i][k]
		}
	}

	return b
}

// probablity for each token in our vocab and expected targets
func CrossEntropy (probs []float64, targets []int) float64 {
	for i := range probs {
		if targets[i] == 1 {
			return -math.Log (probs[i])
		}
	}

	panic ("no target class found")
}

func Concat (a, b [][]float64) ([][]float64, error){
	if len (a) != len (b) {
		return nil, fmt.Errorf ("incompatible matrices. Their rows should be same")
	}

	result := make ([][]float64, len (a))

	for i := range result {
		result[i] = make ([]float64, len (a[i]) + len (b[i]))
		k := 0

		// push a matrix values
		for j:=0;j<len (a[i]); j++ {
			result [i][k] = a[i][j]
			k++
		}

		// push b matrix values
		for j:=0;j<len (b[i]); j++ {
			result [i][k] = b[i][j]
			k++
		}
	}

	return result, nil
}
