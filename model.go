package pia

import (
	"io/ioutil"

	tf "github.com/tensorflow/tensorflow/tensorflow/go"
)

// Model holds the tensorflow graph, session, variable initalization op,
// checkpoint details, restore op, and registered targets.
type Model struct {
	graph                 *tf.Graph
	sess                  *tf.Session
	initOp                *tf.Operation
	checkpointDirectory   string
	checkpointPrefix      string
	checkpointPlaceholder tf.Output
	checkpointOp          *tf.Operation
	restoreOp             *tf.Operation
	targets               map[string]Target
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
	sess, err := tf.NewSession(graph, nil)
	if err != nil {
		return nil, err
	}
	model := &Model{
		graph:                 graph,
		sess:                  sess,
		initOp:                graph.Operation(DefaultInitOpName),
		checkpointOp:          graph.Operation(DefaultSaveOpName),
		restoreOp:             graph.Operation(DefaultRestoreOpName),
		checkpointDirectory:   DefaultCheckpointDirectory,
		checkpointPrefix:      DefaultCheckpointFilePrefix,
		checkpointPlaceholder: graph.Operation(DefaultCheckpointPlaceholderName).Output(0),
	}
	for _, option := range options {
		option(model)
	}
	return model, nil
}

// Init initializes a model's variables by calling the variable init op.
func (m *Model) Init() error {
	if _, err := m.sess.Run(nil, nil, []*tf.Operation{m.initOp}); err != nil {
		return err
	}
	return nil
}

// CheckpointPath overrides the default model checkpoint directory
// and prefix. Include this in a new model call.
func CheckpointPath(directory string, prefix string) func(*Model) {
	return func(m *Model) {
		m.checkpointDirectory = directory
		m.checkpointPrefix = prefix
	}
}
