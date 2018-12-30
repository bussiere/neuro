// Package neuro simplfies the official TensorFlow golang package.
// Neuro runs exclusively using the Tensor Transport Protocol (TTP)
// which greatly simplifies exchanging tensors between languages and
// over the wire. Neuro process transports based on registered targets.
// Registered targets enable type, dim, and alias validation.
package neuro

const (
	// DefaultInitOpName is the default operation name to initialize
	// graph variables.
	DefaultInitOpName = "init"
	// DefaultSaveOpName is the default operation name to save
	// the graph's state.
	DefaultSaveOpName = "save/control_dependency"
	// DefaultRestoreOpName is the default operation name to restore
	// saved graph state from a checkpoint.
	DefaultRestoreOpName = "save/restore_all"
	// DefaultCheckpointPlaceholderName is the default placeholder
	// for feeding in a checkpoint variable.
	DefaultCheckpointPlaceholderName = "save/Const"
	// DefaultCheckpointDirectory is the default folder for saving
	// checkpoint files. E.g. <CHECKPOINTDIR>/<PREFIX>-1.data-00000-of-00001.
	DefaultCheckpointDirectory = "checkpoints"
	// DefaultCheckpointFilePrefix is the default checkpoint file
	// prefix. E.g. <PREFIX>-1.data-00000-of-00001.
	DefaultCheckpointFilePrefix = "checkpoint"
)
