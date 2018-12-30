package neuro

import (
	"io/ioutil"

	tf "github.com/tensorflow/tensorflow/tensorflow/go"
)

// Model holds the tensorflow graph, session, variable initalization op,
// checkpoint details, restore op, and registered targets.
type Model struct {
	graphPath             string
	graph                 *tf.Graph
	initOp                *tf.Operation
	checkpointDirectory   string
	checkpointPrefix      string
	checkpointPlaceholder tf.Output
	checkpointOp          *tf.Operation
	restoreOp             *tf.Operation
	targets               map[string]target
}

// NewModel loads a saved TF graph definition (graph.pb)
// and initializes a new TensorFlow session.
func NewModel(graphDefFilePath string, options ...func(*Model)) (*Model, error) {
	graphDef, err := ioutil.ReadFile(graphDefFilePath)
	if err != nil {
		return nil, err
	}
	graph := tf.NewGraph()
	if err = graph.Import(graphDef, ""); err != nil {
		return nil, err
	}
	// register targets here
	model := &Model{
		graphPath:             graphDefFilePath,
		graph:                 graph,
		initOp:                graph.Operation(DefaultInitOpName),
		checkpointOp:          graph.Operation(DefaultSaveOpName),
		restoreOp:             graph.Operation(DefaultRestoreOpName),
		checkpointDirectory:   DefaultCheckpointDirectory,
		checkpointPrefix:      DefaultCheckpointFilePrefix,
		checkpointPlaceholder: graph.Operation(DefaultCheckpointPlaceholderName).Output(0),
		targets:               make(map[string]target),
	}
	for _, option := range options {
		option(model)
	}
	return model, nil
}

///////////////////////////////////////////////////////////////////////////////
// NewModel Override Functions
///////////////////////////////////////////////////////////////////////////////

// CheckpointPath overrides the default model checkpoint directory
// and prefix. Include this in a new model call.
func CheckpointPath(directory string, prefix string) func(*Model) {
	return func(m *Model) {
		m.checkpointDirectory = directory
		m.checkpointPrefix = prefix
	}
}
