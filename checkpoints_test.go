package neuro

import (
	"testing"
)

func TestSessionInit(t *testing.T) {
	model, err := NewModel("testing/graph.pb")
	if err != nil {
		t.Error(err)
	}
	target := Target{
		Name:    "predict",
		Feeds:   []string{"input"},
		Fetches: []string{"output"},
	}
	model.RegisterTargets(target)
	sess, err := model.NewSession()
	if err != nil {
		t.Error(err)
	}
	err = sess.Init()
	if err != nil {
		t.Error(err)
	}
}

func TestSessionCheckpoint(t *testing.T) {
	model, err := NewModel("testing/graph.pb", CheckpointPath("testing", "checkpoint"))
	if err != nil {
		t.Error(err)
	}
	target := Target{
		Name:    "predict",
		Feeds:   []string{"input"},
		Fetches: []string{"output"},
	}
	model.RegisterTargets(target)
	sess, err := model.NewSession()
	if err != nil {
		t.Error(err)
	}
	err = sess.Init()
	if err != nil {
		t.Error(err)
	}
	err = sess.Checkpoint()
	if err != nil {
		t.Error(err)
	}
}

func TestSessionRestore(t *testing.T) {
	model, err := NewModel("testing/graph.pb", CheckpointPath("testing", "checkpoint"))
	if err != nil {
		t.Error(err)
	}
	target := Target{
		Name:    "predict",
		Feeds:   []string{"input"},
		Fetches: []string{"output"},
	}
	model.RegisterTargets(target)
	sess, err := model.NewSession()
	if err != nil {
		t.Error(err)
	}
	err = sess.Init()
	if err != nil {
		t.Error(err)
	}
	err = sess.Checkpoint()
	if err != nil {
		t.Error(err)
	}
	err = sess.Restore()
	if err != nil {
		t.Error(err)
	}
}
