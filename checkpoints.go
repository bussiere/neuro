package pia

import (
	"fmt"
	"os"
	"path/filepath"

	tf "github.com/tensorflow/tensorflow/tensorflow/go"
)

// Checkpoint saves a model's state to checkpoint file(s). This operation
// automatically increments the checkpoint index.
func (s *Session) Checkpoint() error {
	checkpointPath, err := filepath.Abs(filepath.Join(s.model.checkpointDirectory, s.model.checkpointPrefix))
	if err != nil {
		return err
	}
	checkpointTensor, err := tf.NewTensor(checkpointPath)
	if err != nil {
		return err
	}
	feeds := map[tf.Output]*tf.Tensor{
		s.model.checkpointPlaceholder: checkpointTensor,
	}
	if _, err := s.sess.Run(feeds, nil, []*tf.Operation{s.model.checkpointOp}); err != nil {
		return err
	}
	return nil
}

// Restore loads a models state from the latest checkpoint.
func (s *Session) Restore() error {
	_, err := os.Stat(s.model.checkpointDirectory)
	if os.IsNotExist(err) {
		return fmt.Errorf("checkpoint directory '%s' does not exist", s.model.checkpointDirectory)
	}
	checkpointPath, err := filepath.Abs(filepath.Join(s.model.checkpointDirectory, s.model.checkpointPrefix))
	if err != nil {
		return err
	}
	checkpointTensor, err := tf.NewTensor(checkpointPath)
	if err != nil {
		return err
	}
	feeds := map[tf.Output]*tf.Tensor{
		s.model.checkpointPlaceholder: checkpointTensor,
	}
	if _, err := s.sess.Run(feeds, nil, []*tf.Operation{s.model.restoreOp}); err != nil {
		return err
	}
	return nil
}

// Start is a convenience function that either restores a model
// from the latest checkpoint if the checkpoint directory exists
// or initializes the model's variables using the init method.
func (s *Session) Start() error {
	_, err := os.Stat(s.model.checkpointDirectory)
	if os.IsNotExist(err) {
		return s.Init()
	}
	return s.Restore()
}
