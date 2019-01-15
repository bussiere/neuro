package neuro

import (
	"io/ioutil"

	tf "github.com/tensorflow/tensorflow/tensorflow/go"
)

// Model holds the tensorflow graph, session, variable initialization op,
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
func NewModel(graphDefFilePath string, options ...func(*Model) error) (*Model, error) {
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
		err := option(model)
		if err != nil {
			return nil, err
		}
	}
	return model, nil
}

///////////////////////////////////////////////////////////////////////////////
// NewModel Override Functions
///////////////////////////////////////////////////////////////////////////////

// InitOp overrides the default variable initialization operation name "init".
// Pass an operation name into the function.
func InitOp(name string) func(*Model) error {
	return func(m *Model) error {
		m.initOp = m.graph.Operation(name)
		return nil
	}
}

// CheckpointOp overrides the default variable checkpoint operation.
// The default is: "save/control_dependency". Pass an operation name
// into the function.
func CheckpointOp(name string) func(*Model) error {
	return func(m *Model) error {
		m.checkpointOp = m.graph.Operation(name)
		return nil
	}
}

// RestoreOp overrides the default restore operation. The default
// is: "save/restore_all". Pass an operation name into the function.
func RestoreOp(name string) func(*Model) error {
	return func(m *Model) error {
		m.restoreOp = m.graph.Operation(name)
		return nil
	}
}

// CheckpointPlaceholder overrides the default checkpoint
// prefix placeholder. Default is: "save/Const".
func CheckpointPlaceholder(name string) func(*Model) error {
	return func(m *Model) error {
		opName, outputNumber, err := getOutputFromName(name)
		if err != nil {
			return err
		}
		m.checkpointPlaceholder = m.graph.Operation(opName).Output(outputNumber)
		return nil
	}
}

// CheckpointPath overrides the default model checkpoint directory
// and prefix.
func CheckpointPath(directory string, prefix string) func(*Model) error {
	return func(m *Model) error {
		m.checkpointDirectory = directory
		m.checkpointPrefix = prefix
		return nil
	}
}
