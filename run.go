package neuro

import (
	tf "github.com/tensorflow/tensorflow/tensorflow/go"
	ttp "github.com/tensortask/ttp/gen"
)

// Run executes a TensorFlow model using the TTP standard. Feeds are
// included in the TTP transport's "tensors" field. The target specifices
// which operation's to run and which feeds to return. Targets are registered
// with the RegisterTarget function. Registered Targets automatically validate
// input tensor aliases/dimensions/types.
func (s *Session) Run(transport ttp.Transport) (ttp.Transport, error) {
	feeds, fetches, operations, err := s.model.convertTransportToTF(transport)
	if err != nil {
		return ttp.Transport{}, err
	}
	// unwrap the fetches
	var tfFetches []tf.Output
	var fetchOrder []string
	for k, v := range fetches {
		tfFetches = append(tfFetches, v.graphOutput)
		fetchOrder = append(fetchOrder, k)
	}
	results, err := s.sess.Run(feeds, tfFetches, operations)
	if err != nil {
		return ttp.Transport{}, err
	}
	returnTransport, err := convertTFToTransport(fetchOrder, results)
	if err != nil {
		return ttp.Transport{}, err
	}
	return returnTransport, nil
}
