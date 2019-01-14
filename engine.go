package neuro

import (
	"fmt"

	ttp "github.com/tensortask/ttp/go"
)

// Engine consolidates all models into a single execution
// engine which routes input transports to the corresponding
// model and task.
type Engine struct {
	overwritable bool
	targets      map[string]*Model
}

// NewEngine creates an execution engine that consolidates
// targets across model(s). Engines route input transports
// to the corresponding model, target, and run handler.
func NewEngine(models ...*Model) (*Engine, error) {
	engine := Engine{
		overwritable: false,
		targets:      make(map[string]*Model),
	}
	if len(models) != 0 {
		err := engine.RegisterModels(models...)
		if err != nil {
			return nil, err
		}
	}
	return &engine, nil
}

// RegisterModels adds new models and their targets to
// the execution engine. Adds every target included in the
// model. Use RegisterTarget to add or overwrite a single target.
func (e *Engine) RegisterModels(models ...*Model) error {
	for _, model := range models {
		for name := range model.targets {
			_, targetPresent := model.targets[name]
			if targetPresent && !e.overwritable {
				return fmt.Errorf("target ")
			}
			e.targets[name] = model
		}
	}
	return nil
}

// RegisterTarget adds a single target (from a single model) to an
// execution engine.
func (e *Engine) RegisterTarget(model *Model, name string) error {
	_, targetPresent := model.targets[name]
	if targetPresent && !e.overwritable {
		return fmt.Errorf("target ")
	}
	e.targets[name] = model
	return nil
}

// ToggleOverwrite switches an engines private overwritable
// engine. Default is to disallow overwrites. Must call this
// to switch from default.
func (e *Engine) ToggleOverwrite(overwritable bool) {
	e.overwritable = overwritable
}

func (e *Engine) Run(input ttp.Transport) (ttp.Transport, error) {
	e.targets[input.Target].targets[input.Target].runHandler.Run(input)

}
