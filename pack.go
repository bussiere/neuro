package neuro

import (
	tf "github.com/tensorflow/tensorflow/tensorflow/go"
	ttp "github.com/tensortask/ttp/gen"
)

// PackTransport takes a target and a map of aliases and native types
// and first converts them to a tensorflow tensor and then packs that into
// a TTP transport. Useful for working directly with golang native types.
// NOTE: the type must be serializable by the tensorflow package (numerical).
func PackTransport(target string, tensors map[string]interface{}) (ttp.Transport, error) {
	var aliases []string
	var tfTensors []*tf.Tensor
	for key, value := range tensors {
		tfTensor, err := tf.NewTensor(value)
		if err != nil {
			return ttp.Transport{}, err
		}
		aliases = append(aliases, key)
		tfTensors = append(tfTensors, tfTensor)
	}
	transport, err := convertTFToTransport(aliases, tfTensors)
	if err != nil {
		return ttp.Transport{}, err
	}
	transport.Target = target
	return transport, nil
}
