package main

import (
	tokenizers "chroma-default-ef/toke"
	"errors"
	"fmt"
	ort "github.com/yalue/onnxruntime_go"
	"math"
)

/*
#cgo LDFLAGS: -L./libs
*/
import "C"

func ReshapeFlattenedTensor(flatTensor []float32, shape []int) (interface{}, error) {
	// Check if the shape is valid (2D or 3D)
	if len(shape) != 2 && len(shape) != 3 {
		return nil, errors.New("shape must be 2D or 3D")
	}

	// Calculate total elements based on shape
	totalElements := 1
	for _, dim := range shape {
		totalElements *= dim
	}

	// Check if the input slice has the correct number of elements
	if len(flatTensor) != totalElements {
		return nil, errors.New("input slice length does not match the specified shape")
	}

	if len(shape) == 2 {
		// Handle 2D case
		tensor := make([][]float32, shape[0])
		for i := range tensor {
			tensor[i] = make([]float32, shape[1])
		}

		index := 0
		for i := 0; i < shape[0]; i++ {
			for j := 0; j < shape[1]; j++ {
				tensor[i][j] = flatTensor[index]
				index++
			}
		}
		return tensor, nil
	} else {
		// Handle 3D case
		tensor := make([][][]float32, shape[0])
		for i := range tensor {
			tensor[i] = make([][]float32, shape[1])
			for j := range tensor[i] {
				tensor[i][j] = make([]float32, shape[2])
			}
		}

		index := 0
		for i := 0; i < shape[0]; i++ {
			for j := 0; j < shape[1]; j++ {
				for k := 0; k < shape[2]; k++ {
					tensor[i][j][k] = flatTensor[index]
					index++
				}
			}
		}
		return tensor, nil
	}
}

// GOOD
// ExpandDims adds a dimension of size 1 to the end of a 2D slice
func ExpandDims1(input [][]int64) [][][]int64 {
	x := len(input)
	y := len(input[0])

	output := make([][][]int64, x)
	for i := range output {
		output[i] = make([][]int64, y)
		for j := range output[i] {
			output[i][j] = []int64{input[i][j]}
		}
	}

	return output
}

type Tensor3Di = [][][]int64

// BroadcastTo simulates np.broadcast_to for any 3D tensor
func BroadcastTo(input Tensor3Di, targetShape [3]int) Tensor3Di {
	result := make(Tensor3Di, targetShape[0])
	for i := range result {
		result[i] = make([][]int64, targetShape[1])
		for j := range result[i] {
			result[i][j] = make([]int64, targetShape[2])
			for k := range result[i][j] {
				// Use modulo to wrap around input dimensions
				i_in := i % len(input)
				j_in := j % len(input[i_in])
				k_in := k % len(input[i_in][j_in])
				result[i][j][k] = input[i_in][j_in][k_in]
			}
		}
	}
	return result
}

// Number is a constraint that permits any number type
type Number interface {
	~int | ~int8 | ~int16 | ~int32 | ~int64 | ~uint | ~uint8 | ~uint16 | ~uint32 | ~uint64 | ~float32 | ~float64
}

// Tensor3D is a generic 3D tensor
type Tensor3D[T Number] [][][]T

// ConvertTensor3D converts a Tensor3D of one numeric type to another
func ConvertTensor3D[S Number, D Number](src Tensor3D[S]) Tensor3D[D] {
	dst := make(Tensor3D[D], len(src))
	for i := range src {
		dst[i] = make([][]D, len(src[i]))
		for j := range src[i] {
			dst[i][j] = make([]D, len(src[i][j]))
			for k := range src[i][j] {
				dst[i][j][k] = D(src[i][j][k])
			}
		}
	}
	return dst
}

type Tensor2D[T Number] [][]T

// ConvertTensor2D converts a Tensor2D of one numeric type to another
func ConvertTensor2D[S Number, D Number](src Tensor2D[S]) Tensor2D[D] {
	dst := make(Tensor2D[D], len(src))
	for i := range src {
		dst[i] = make([]D, len(src[i]))
		for j := range src[i] {
			dst[i][j] = D(src[i][j])
		}
	}
	return dst
}

// Sum calculates the sum along a specified axis
func (t Tensor3D[T]) Sum(axis int) ([][]T, error) {
	if len(t) == 0 || len(t[0]) == 0 || len(t[0][0]) == 0 {
		return nil, errors.New("empty tensor")
	}

	shape := []int{len(t), len(t[0]), len(t[0][0])}

	switch axis {
	case 0:
		result := make([][]T, shape[1])
		for i := range result {
			result[i] = make([]T, shape[2])
		}
		for i := 0; i < shape[1]; i++ {
			for j := 0; j < shape[2]; j++ {
				var sum T
				for k := 0; k < shape[0]; k++ {
					sum += t[k][i][j]
				}
				result[i][j] = sum
			}
		}
		return result, nil
	case 1:
		result := make([][]T, shape[0])
		for i := range result {
			result[i] = make([]T, shape[2])
		}
		for i := 0; i < shape[0]; i++ {
			for j := 0; j < shape[2]; j++ {
				var sum T
				for k := 0; k < shape[1]; k++ {
					sum += t[i][k][j]
				}
				result[i][j] = sum
			}
		}
		return result, nil
	case 2:
		result := make([][]T, shape[0])
		for i := range result {
			result[i] = make([]T, shape[1])
		}
		for i := 0; i < shape[0]; i++ {
			for j := 0; j < shape[1]; j++ {
				var sum T
				for k := 0; k < shape[2]; k++ {
					sum += t[i][j][k]
				}
				result[i][j] = sum
			}
		}
		return result, nil
	default:
		return nil, fmt.Errorf("invalid axis: %d", axis)
	}
}

func multiplyTensors(a [][][]float32, b [][][]int64) ([][][]float32, error) {
	// Convert b to float32
	bFloat := make([][][]float32, len(b))
	for i := range b {
		bFloat[i] = make([][]float32, len(b[i]))
		for j := range b[i] {
			bFloat[i][j] = make([]float32, len(b[i][j]))
			for k := range b[i][j] {
				bFloat[i][j][k] = float32(b[i][j][k])
			}
		}
	}

	// Check dimensions
	if len(a) != len(bFloat) || len(a[0]) != len(bFloat[0]) || len(a[0][0]) != len(bFloat[0][0]) {
		return nil, errors.New("tensor dimensions are not compatible for element-wise multiplication")
	}

	// Perform multiplication
	result := make([][][]float32, len(a))
	for i := range a {
		result[i] = make([][]float32, len(a[i]))
		for j := range a[i] {
			result[i][j] = make([]float32, len(a[i][j]))
			for k := range a[i][j] {
				result[i][j][k] = a[i][j][k] * bFloat[i][j][k]
			}
		}
	}

	return result, nil
}

// clip applies the clip operation to a Tensor2D
func clip[T Number](input Tensor2D[T], min, max T) Tensor2D[T] {
	rows := len(input)
	if rows == 0 {
		return input
	}
	cols := len(input[0])

	result := make(Tensor2D[T], rows)
	for i := range result {
		result[i] = make([]T, cols)
	}

	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			result[i][j] = clipValue(input[i][j], min, max)
		}
	}

	return result
}

// clipValue clips a single value between min and max
func clipValue[T Number](x, min, max T) T {
	if x < min {
		return min
	}
	if x > max {
		return max
	}
	return x
}

// divide performs element-wise division of tensor a by tensor b
// It supports broadcasting and handles division by zero similar to NumPy
func divide[T Number](a, b Tensor2D[T]) Tensor2D[T] {
	rowsA, colsA := len(a), len(a[0])
	rowsB, colsB := len(b), len(b[0])

	// Determine output shape based on broadcasting rules
	rowsOut, colsOut := max(rowsA, rowsB), max(colsA, colsB)

	result := make(Tensor2D[T], rowsOut)
	for i := range result {
		result[i] = make([]T, colsOut)
	}

	for i := 0; i < rowsOut; i++ {
		for j := 0; j < colsOut; j++ {
			aVal := a[i%rowsA][j%colsA]
			bVal := b[i%rowsB][j%colsB]
			result[i][j] = divideValues(aVal, bVal)
		}
	}

	return result
}

// divideValues performs division for a single pair of values
func divideValues[T Number](a, b T) T {
	if b == 0 {
		if a > 0 {
			return T(math.Inf(1))
		} else if a < 0 {
			return T(math.Inf(-1))
		} else {
			return T(math.NaN())
		}
	}
	return T(float64(a) / float64(b))
}

func main() {
	err := tokenizers.LoadLibrary("libs/libtokenizers.dylib")
	if err != nil {
		panic(err)
	}
	tk, err := tokenizers.FromFile("/Users/tazarov/.cache/chroma/onnx_models/all-MiniLM-L6-v2/onnx/tokenizer.json")
	if err != nil {
		panic(err)
	}
	ort.SetSharedLibraryPath("./libs/libonnxruntime.1.18.0.dylib")
	err = ort.InitializeEnvironment()
	if err != nil {
		panic(err)
	}
	defer func() {
		err := ort.DestroyEnvironment()
		if err != nil {
			panic(err)
		}
	}()
	// release native resources
	defer func(tk *tokenizers.Tokenizer) {
		err := tk.Close()
		if err != nil {

		}
	}(tk)
	var res1, _ = tk.EncodeWithOptions("Mellow world", true, tokenizers.WithReturnAttentionMask(), tokenizers.WithReturnTypeIDs())
	var res, _ = tk.EncodeWithOptions("Hello, my name is John. I am a Data Scientist.", true, tokenizers.WithReturnAttentionMask(), tokenizers.WithReturnTypeIDs())
	inputIDs := make([]int64, len(res.IDs)+len(res1.IDs))
	for i, v := range res.IDs {
		inputIDs[i] = int64(v)
	}
	for i, v := range res1.IDs {
		inputIDs[i+len(res.IDs)] = int64(v)
	}

	attnMask := make([]int64, len(res.AttentionMask)+len(res1.AttentionMask))
	for i, v := range res.AttentionMask {
		attnMask[i] = int64(v)
	}
	for i, v := range res1.AttentionMask {
		attnMask[i+len(res.AttentionMask)] = int64(v)
	}
	typeIDs := make([]int64, len(res.TypeIDs)+len(res1.TypeIDs))
	for i, v := range res.TypeIDs {
		typeIDs[i] = int64(v)
	}

	for i, v := range res1.TypeIDs {
		typeIDs[i+len(res.TypeIDs)] = int64(v)
	}

	inputShape := ort.NewShape(2, int64(len(res.TypeIDs)))
	inputTensor, err := ort.NewTensor(inputShape, inputIDs)
	attentionTensor, err := ort.NewTensor(inputShape, attnMask)
	typeTensor, err := ort.NewTensor(inputShape, typeIDs)
	defer inputTensor.Destroy()
	defer attentionTensor.Destroy()
	defer typeTensor.Destroy()
	outputShape := ort.NewShape(2, 256, 384)
	outputTensor, err := ort.NewEmptyTensor[float32](outputShape)
	defer outputTensor.Destroy()

	fmt.Println(inputTensor)
	//fmt.Println(attentionTensor)
	//fmt.Println(typeTensor)

	session, err := ort.NewAdvancedSession("/Users/tazarov/.cache/chroma/onnx_models/all-MiniLM-L6-v2/onnx/model.onnx",
		[]string{"input_ids", "attention_mask", "token_type_ids"}, []string{"last_hidden_state"},
		[]ort.Value{inputTensor, attentionTensor, typeTensor}, []ort.Value{outputTensor}, nil)
	defer session.Destroy()

	// Check for errors
	if err != nil {
		panic(err)
	}

	// Calling Run() will run the network, reading the current contents of the
	// input tensors and modifying the contents of the output tensors.
	err = session.Run()

	if err != nil {
		panic(err)
	}

	// Get a slice view of the output tensor's data.
	outputData := outputTensor.GetData()

	for _, v := range outputData {
		if v == 0.03936 {
			fmt.Println("=======")
		}
	}
	t, err := ReshapeFlattenedTensor(outputData, []int{2, 256, 384})
	if err != nil {
		fmt.Println("Error1:", err)
		return
	}
	//fmt.Println(inpt.([][][]float32)[0][0])
	fmt.Println("=-======================")

	attnMask1 := make([]int64, len(res.AttentionMask))
	for i, v := range res.AttentionMask {
		attnMask1[i] = int64(v)
	}
	attnMask2 := make([]int64, len(res1.AttentionMask))
	for i, v := range res1.AttentionMask {
		attnMask2[i] = int64(v)
	}

	expandedMask := BroadcastTo(ExpandDims1([][]int64{attnMask1, attnMask2}), [3]int{2, 256, 384})
	mtpl, err := multiplyTensors(t.([][][]float32), expandedMask)
	if err != nil {
		fmt.Println("Error2:", err)
		return
	}

	summed, err := Tensor3D[float32](mtpl).Sum(1)
	if err != nil {
		fmt.Println("Error3:", err)
		return
	}
	summedExpandedMask, err := Tensor3D[int64](expandedMask).Sum(1)
	if err != nil {
		fmt.Println("Error4:", err)
		return
	}
	summedExpandedMaskF32 := ConvertTensor2D[int64, float32](summedExpandedMask)
	clippedSummed := clip(summedExpandedMaskF32, 1e-9, math.MaxFloat32)
	fmt.Println(clippedSummed)
	fmt.Println(summed)
	embeddings := divide(summed, clippedSummed)
	if err != nil {
		fmt.Println("Error5:", err)
		return
	}
	fmt.Println("embeddings")
	fmt.Println(normalize(embeddings)[1])

}

// normalize function for a generic Tensor2D type.
func normalize[T Number](v Tensor2D[T]) Tensor2D[float64] {
	rows := len(v)
	cols := len(v[0])
	norm := make([]float64, rows)

	// Step 1: Compute the L2 norm of each row
	for i := 0; i < rows; i++ {
		sum := 0.0
		for j := 0; j < cols; j++ {
			sum += float64(v[i][j]) * float64(v[i][j])
		}
		norm[i] = math.Sqrt(sum)
	}

	// Step 2: Handle zero norms
	for i := 0; i < rows; i++ {
		if norm[i] == 0 {
			norm[i] = 1e-12
		}
	}

	// Step 3: Normalize each row
	normalized := make(Tensor2D[float64], rows)
	for i := 0; i < rows; i++ {
		normalized[i] = make([]float64, cols)
		for j := 0; j < cols; j++ {
			normalized[i][j] = float64(v[i][j]) / norm[i]
		}
	}

	return normalized
}
