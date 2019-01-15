package neuro

import (
	"fmt"

	tf "github.com/tensorflow/tensorflow/tensorflow/go"
	ttp "github.com/tensortask/ttp/go"
)

// Run executes a TensorFlow model using the TTP standard. Feeds are
// included in the TTP transport's "tensors" field. The target specifics
// which operation's to run and which feeds to return. Targets are registered
// with the RegisterTarget function. Registered Targets automatically validate
// input tensor aliases/dimensions/types.
func (s *Session) Run(input ttp.Transport) (ttp.Transport, error) {
	feeds, fetches, operations, err := s.model.convertTransportToTF(input)
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

// PartialRun ...
type PartialRun struct {
	partialRun *tf.PartialRun
	model      Model
}

// NewPartialRun creates a partial run structure which contains a tensorflow
// partial run pointer and a snapshot of the model being used.
func (s *Session) NewPartialRun(targetNames ...string) (PartialRun, error) {
	feeds, fetches, operations, err := s.consolidateTargets(targetNames...)
	if err != nil {
		return PartialRun{}, err
	}
	partialRun, err := s.sess.NewPartialRun(feeds, fetches, operations)
	if err != nil {
		return PartialRun{}, err
	}

	return PartialRun{
		partialRun: partialRun,
		model:      s.model,
	}, nil
}

// Run executes a graph AND saves the graph's state. This is useful for closed
// feedback loops or forward spikes.
func (pr *PartialRun) Run(input ttp.Transport) (ttp.Transport, error) {
	feeds, fetches, operations, err := pr.model.convertTransportToTF(input)
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
	results, err := pr.partialRun.Run(feeds, tfFetches, operations)
	if err != nil {
		return ttp.Transport{}, err
	}
	returnTransport, err := convertTFToTransport(fetchOrder, results)
	if err != nil {
		return ttp.Transport{}, err
	}
	return returnTransport, nil
}

// consolidateTargets converts a maps of outputs and operations to an array
// of feeds, fetches, and operations based on the input targets specified.
// Useful for clustering feeds, fetches, and operations prior to initializing
// a partial run.
func (s *Session) consolidateTargets(targetNames ...string) ([]tf.Output, []tf.Output, []*tf.Operation, error) {
	var feeds []tf.Output
	var fetches []tf.Output
	var operations []*tf.Operation

	for _, targetName := range targetNames {
		target, ok := s.model.targets[targetName]
		if !ok {
			return nil, nil, nil, fmt.Errorf("target name %s not registered", targetName)
		}
		for _, output := range target.feeds {
			feeds = append(feeds, output.graphOutput)
		}
		for _, output := range target.fetches {
			fetches = append(fetches, output.graphOutput)
		}
		operations = append(operations, target.operations...)
	}
	return feeds, fetches, operations, nil
}
