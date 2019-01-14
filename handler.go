package neuro

import ttp "github.com/tensortask/ttp/go"

// RunInterface is an interface that abstracts various methods of
// running a graph.
type RunInterface interface {
	Run(input ttp.Transport) (ttp.Transport, error)
}

type runHandler struct{}

func (rh *runHandler) Run(input ttp.Transport) (ttp.Transport, error) { return ttp.Transport{}, nil }

type PerRunHandler struct {
	*runHandler
	model *Model
}

func (rh *PerRunHandler) Run(input ttp.Transport) (ttp.Transport, error) {
	sess, err := rh.model.NewSession()
	if err != nil {
		return ttp.Transport{}, err
	}
	err = sess.Start()
	if err != nil {
		return ttp.Transport{}, err
	}
	sess.Run(input)
	return ttp.Transport{}, nil
}

type PoolRunHandler struct {
	*runHandler
}

type PartialRunHandler struct {
	*runHandler
}
