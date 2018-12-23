package main

import (
	"fmt"

	"github.com/tensortask/neuro"
)

func main() {
	// NewModel loads a model using the supplied graph definition.
	// Pass options into new model to change default parameters.
	model, err := neuro.NewModel("graph.pb")
	if err != nil {
		panic(err)
	}

	// Target represents a target computation.
	// In this case we are creating a target called predict
	// which feeds data into the input node and fetches data
	// from the output node.
	target := neuro.Target{
		Name:    "predict",
		Feeds:   []string{"input"},
		Fetches: []string{"output"},
	}

	// Before using a computational target, it must be registered with the model.
	// RegisterTargets() takes any number of input targets and registers it with
	// the model. RegisterTargets() automatically performs op/output referencing.
	model.RegisterTargets(target)

	// Prior to running a computation, a session must be created. Sessions are
	// thread-safe.
	sess, err := model.NewSession()
	if err != nil {
		panic(err)
	}

	// Any variables in the graph must be loaded or initialized.
	// The Start() method looks for checkpoints and loads them.
	// If there are no checkpoints, Start() initializes the variables.
	err = sess.Start()
	if err != nil {
		panic(err)
	}

	// Note: neuro runs exclusively using the tensor transport protocol (TTP).
	// TTP is especially useful when it comes to exchanging tensors between languages
	// or sending them over the wire.

	// In this example we will convert native go types to a TTP transport.
	tensors := map[string]interface{}{
		"input": [][][]float32{{{1}}, {{2}}, {{3}}},
	}

	// PackTransport takes a target and a map of tensor aliases and interfaces.
	transport, err := neuro.PackTransport("predict", tensors)
	if err != nil {
		panic(err)
	}

	// Run will process the transport. Remember, the transport includes the name
	// of the computational target (predict) that we registered above and a set of
	// tensors. Run automatically handles alias, type, and size validation.
	result, err := sess.Run(transport)
	if err != nil {
		panic(err)
	}

	// Do stuff with the results
	fmt.Println(result)
}
