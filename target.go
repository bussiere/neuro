package neuro

import (
	"fmt"
	"strconv"
	"strings"

	"github.com/gogo/protobuf/proto"
	tf "github.com/tensorflow/tensorflow/tensorflow/go"
	"github.com/tensortask/tfprotos/core/framework"
)

// Target ...
type Target struct {
	Name       string
	Feeds      []string
	Fetches    []string
	Operations []string
}

type target struct {
	name       string
	feeds      map[string]*Output
	fetches    map[string]*Output
	operations map[string]*tf.Operation
}

// Output ...
type Output struct {
	graphOutput tf.Output
	dataType    string
	dim         []int64
}

// RegisterTargets registers a target computation for
// the given TensorFlow model.
func (m *Model) RegisterTargets(targets ...Target) error {
	return nil
}

// graph.Operation("input").Output(0)
// graph.Operation("save/restore_all")

// sanitize output names <NAME>:0 -> must become just <NAME>
func parseGraphDef(nodeNames []string, graphDef *[]byte) (map[string]*framework.NodeDef, error) {
	graphProto := &framework.GraphDef{}

	err := proto.Unmarshal(*graphDef, graphProto)
	if err != nil {
		return nil, err
	}
	nodes := graphProto.GetNode()
	var nodeMap map[string]*framework.NodeDef

	for _, node := range nodes {
		if nameIsPresent(node.Name, nodeNames) {
			nodeMap[node.Name] = node
		}
	}
	return nodeMap, nil
}

func (m *Model) makeTFOperations(operationNames []string, nodeMap map[string]*framework.NodeDef) (map[string]*tf.Operation, error) {
	var operationMap map[string]*tf.Operation
	for _, operationName := range operationNames {
		if nodeMap[operationName] != nil {
			operationMap[operationName] = m.graph.Operation(operationName)
		} else {
			return nil, fmt.Errorf("operation '%s' is not present in graph '%s'", operationName, m.graphPath)
		}
	}
	return operationMap, nil
}

func (m *Model) makeTFOutputs(outputNames []string, nodeMap map[string]*framework.NodeDef) (map[string]*Output, error) {
	var outputMap map[string]*Output
	for _, outputName := range outputNames {
		node := nodeMap[outputName]
		if node != nil {
			output := Output{}
			operationName, outputIndex, err := getOutputFromName(outputName)
			if err != nil {
				return nil, err
			}
			output.graphOutput = m.graph.Operation(operationName).Output(outputIndex)
			output.dataType = node.Attr["dtype"].GetType().String()
			for _, size := range node.Attr["shape"].GetShape().GetDim() {
				output.dim = append(output.dim, size.GetSize())
			}
			outputMap[outputName] = &output
		} else {
			return nil, fmt.Errorf("ouput '%s' is not present in graph '%s'", outputName, m.graphPath)
		}
	}
	// validation and stuff
	return outputMap, nil
}

func nameIsPresent(nodeName string, nameList []string) bool {
	for _, nameFromList := range nameList {
		if nodeName == nameFromList {
			return true
		}
	}
	return false
}

func getOutputFromName(outputName string) (string, int, error) {
	s := strings.Split(outputName, ":")
	if len(s) == 1 {
		return s[0], 0, nil
	}
	if len(s) == 2 {
		index, err := strconv.Atoi(s[1])
		if err != nil {
			return "", 0, err
		}
		return s[0], index, nil
	}
	return "", 0, fmt.Errorf("output name %s should include a maximum of one colon", outputName)
}
