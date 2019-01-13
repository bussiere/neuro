package neuro

import (
	"fmt"
	"os"
	"path/filepath"

	tf "github.com/tensorflow/tensorflow/tensorflow/go"
)

// Checkpoint saves the model's state to a checkpoint file. This operation
// automatically increments the checkpoint index. The checkpoint diretory and
// file prefix must be set at model initalization.
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

// Restore loads a models state from the latest checkpoint. Returns an error if
// the checkpoint directory/prefix does not exist. The restore op is configured at
// model initalization.
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

// Init initializes a model's variables by calling the variable init op within the graph.
func (s *Session) Init() error {
	if _, err := s.sess.Run(nil, nil, []*tf.Operation{s.model.initOp}); err != nil {
		return err
	}
	return nil
}

///////////////////////////////////////////////////////////////////////////////
// PARTIAL RUN CHECKPOINT HELPERS
///////////////////////////////////////////////////////////////////////////////

// Checkpoint saves the model's state to a checkpoint file. This operation
// automatically increments the checkpoint index. The checkpoint diretory and
// file prefix must be set at model initalization.
func (pr *PartialRun) Checkpoint() error {
	checkpointPath, err := filepath.Abs(filepath.Join(pr.model.checkpointDirectory, pr.model.checkpointPrefix))
	if err != nil {
		return err
	}
	checkpointTensor, err := tf.NewTensor(checkpointPath)
	if err != nil {
		return err
	}
	feeds := map[tf.Output]*tf.Tensor{
		pr.model.checkpointPlaceholder: checkpointTensor,
	}
	if _, err := pr.partialRun.Run(feeds, nil, []*tf.Operation{pr.model.checkpointOp}); err != nil {
		return err
	}
	return nil
}

// Restore loads a models state from the latest checkpoint. Returns an error if
// the checkpoint directory/prefix does not exist. The restore op is configured at
// model initalization.
func (pr *PartialRun) Restore() error {
	_, err := os.Stat(pr.model.checkpointDirectory)
	if os.IsNotExist(err) {
		return fmt.Errorf("checkpoint directory '%s' does not exist", pr.model.checkpointDirectory)
	}
	checkpointPath, err := filepath.Abs(filepath.Join(pr.model.checkpointDirectory, pr.model.checkpointPrefix))
	if err != nil {
		return err
	}
	checkpointTensor, err := tf.NewTensor(checkpointPath)
	if err != nil {
		return err
	}
	feeds := map[tf.Output]*tf.Tensor{
		pr.model.checkpointPlaceholder: checkpointTensor,
	}
	if _, err := pr.partialRun.Run(feeds, nil, []*tf.Operation{pr.model.restoreOp}); err != nil {
		return err
	}
	return nil
}

// Start is a convenience function that either restores a model
// from the latest checkpoint if the checkpoint directory exists
// or initializes the model's variables using the init method.
func (pr *PartialRun) Start() error {
	_, err := os.Stat(pr.model.checkpointDirectory)
	if os.IsNotExist(err) {
		return pr.Init()
	}
	return pr.Restore()
}

// Init initializes a model's variables by calling the variable init op within the graph.
func (pr *PartialRun) Init() error {
	if _, err := pr.partialRun.Run(nil, nil, []*tf.Operation{pr.model.initOp}); err != nil {
		return err
	}
	return nil
}
