package pia

import (
	"fmt"
	"os"
	"path/filepath"

	tf "github.com/tensorflow/tensorflow/tensorflow/go"
)

// Checkpoint saves a model's state to checkpoint file(s). This operation
// automatically increments the checkpoint index.
func (m *Model) Checkpoint() error {
	checkpointPath, err := filepath.Abs(filepath.Join(m.checkpointDirectory, m.checkpointPrefix))
	if err != nil {
		return err
	}
	checkpointTensor, err := tf.NewTensor(checkpointPath)
	if err != nil {
		return err
	}
	feeds := map[tf.Output]*tf.Tensor{
		m.checkpointPlaceholder: checkpointTensor,
	}
	if _, err := m.sess.Run(feeds, nil, []*tf.Operation{m.checkpointOp}); err != nil {
		return err
	}
	return nil
}

// Restore loads a models state from the latest checkpoint.
func (m *Model) Restore() error {
	_, err := os.Stat(m.checkpointDirectory)
	if os.IsNotExist(err) {
		return fmt.Errorf("checkpoint directory '%s' does not exist", m.checkpointDirectory)
	}
	checkpointPath, err := filepath.Abs(filepath.Join(m.checkpointDirectory, m.checkpointPrefix))
	if err != nil {
		return err
	}
	checkpointTensor, err := tf.NewTensor(checkpointPath)
	if err != nil {
		return err
	}
	feeds := map[tf.Output]*tf.Tensor{
		m.checkpointPlaceholder: checkpointTensor,
	}
	if _, err := m.sess.Run(feeds, nil, []*tf.Operation{m.restoreOp}); err != nil {
		return err
	}
	return nil
}
