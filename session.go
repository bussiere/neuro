package neuro

import (
	tf "github.com/tensorflow/tensorflow/tensorflow/go"
)

// Session contains a snapshot of the model being used and
// an underlying tensorflow session from which to run the model.
type Session struct {
	model Model
	sess  *tf.Session
}

// NewSession creates a new TensorFlow session and copies the model's structure.
// NewSession must be called before a model is run.
func (m *Model) NewSession() (*Session, error) {
	sess, err := tf.NewSession(m.graph, &tf.SessionOptions{})
	if err != nil {
		return nil, err
	}
	return &Session{
		sess:  sess,
		model: *m,
	}, nil
}

// NewSessionWithOptions is the same as NewSession but allows feeding in custom
// TensorFlow Session options into the underlying tf session.
func (m *Model) NewSessionWithOptions(options *tf.SessionOptions) (*Session, error) {
	sess, err := tf.NewSession(m.graph, options)
	if err != nil {
		return nil, err
	}
	return &Session{
		sess:  sess,
		model: *m,
	}, nil
}
