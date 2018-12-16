package pia

import (
	ttp "github.com/tensortask/ttp/gen"
)

// Run executes a TensorFlow model using the TTP standard. Feeds are
// included in the TTP transport's "tensors" field. The target specifices
// which operation's to run and which feeds to return. Targets are registered
// with the RegisterTarget function. Registered Targets automatically validate
// input tensor shapes/data types.
func (m *Model) Run(transport ttp.Transport) {
	// batchTensor, err := tf.NewTensor(batch)
	// if err != nil {
	// 	panic(err)
	// }
	// feeds := map[tf.Output]*tf.Tensor{m.input: batchTensor}
	// fetches := []tf.Output{m.output}
	// results, err := m.sess.Run(feeds, fetches, nil)
	// if err != nil {
	// 	panic(err)
	// }
	// fetched := results[0].Value().([][][]float32)
	// fmt.Println("Predictions:")
	// for i := range batch {
	// 	fmt.Printf("\tx = %v, predicted y = %v\n", batch[i], fetched[i])
	// }
}
