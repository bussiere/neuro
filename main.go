package pia

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

// func (m *Model) RunTrainStep(inputBatch, targetBatch [][][]float32) {
// 	inputTensor, err := tf.NewTensor(inputBatch)
// 	if err != nil {
// 		panic(err)
// 	}
// 	targetTensor, err := tf.NewTensor(targetBatch)
// 	if err != nil {
// 		panic(err)
// 	}
// 	feeds := map[tf.Output]*tf.Tensor{
// 		m.input:  inputTensor,
// 		m.target: targetTensor,
// 	}
// 	if _, err = m.sess.Run(feeds, nil, []*tf.Operation{m.trainOp}); err != nil {
// 		panic(err)
// 	}
// }

// func main() {
// 	var (
// 		graphDef            = "graph.pb"
// 		checkpointPrefix, _ = filepath.Abs(filepath.Join("checkpoints", "checkpoint"))
// 		restore             = directoryExists("checkpoints")
// 	)

// 	log.Print("Loading graph")
// 	model := NewModel(graphDef)

// 	if restore {
// 		log.Print("Restoring variables from checkpoint")
// 		model.Restore(checkpointPrefix)
// 	} else {
// 		log.Print("Initializing variables")
// 		model.Init()
// 	}

// 	testdata := [][][]float32{{{1}}, {{2}}, {{3}}}
// 	log.Print("Generating initial predictions")
// 	model.Predict(testdata)

// 	log.Print("Training for a few steps")
// 	for i := 0; i < 200; i++ {
// 		model.RunTrainStep(nextBatchForTraining())
// 	}

// 	log.Print("Updated predictions")
// 	model.Predict(testdata)

// 	log.Print("Saving checkpoint")
// 	model.Checkpoint(checkpointPrefix)
// }

// func directoryExists(dir string) bool {
// 	_, err := os.Stat(dir)
// 	return !os.IsNotExist(err)
// }

// func nextBatchForTraining() (inputs, targets [][][]float32) {
// 	const BATCH_SIZE = 10
// 	for i := 0; i < BATCH_SIZE; i++ {
// 		v := rand.Float32()
// 		inputs = append(inputs, [][]float32{{v}})
// 		targets = append(targets, [][]float32{{3*v + 2}})
// 	}
// 	return
// }
