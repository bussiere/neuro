package neuro

import (
	"fmt"
	"io/ioutil"
	"strconv"
	"strings"

	"github.com/gogo/protobuf/proto"
	tf "github.com/tensorflow/tensorflow/tensorflow/go"
	"github.com/tensortask/tfprotos/core/framework"
)

// Target contains the lists of names required to parse out
// TensorFlow outputs and operations from the graph definition.
// Target must have a name.
type Target struct {
	Name       string
	Feeds      []string
	Fetches    []string
	Operations []string
}

type target struct {
	name       string
	feeds      map[string]output
	fetches    map[string]output
	operations []*tf.Operation
}

type output struct {
	graphOutput tf.Output
	dataType    string
	dim         []int64
}

// RegisterTargets registers a target computation for
// the given TensorFlow model.
func (m *Model) RegisterTargets(targets ...Target) error {
	nodeList := consolidateTargetNodeNames(targets...)
	graphDef, err := ioutil.ReadFile(m.graphPath)
	if err != nil {
		return err
	}
	nodeMap, err := parseGraphDef(nodeList, &graphDef)
	if err != nil {
		return err
	}

	for _, inputTarget := range targets {
		t := target{
			name:    inputTarget.Name,
			feeds:   make(map[string]output),
			fetches: make(map[string]output),
		}
		ops, err := m.makeTFOperations(inputTarget.Operations, nodeMap)
		if err != nil {
			return err
		}
		t.operations = ops

		feeds, err := m.makeTFOutputs(inputTarget.Feeds, nodeMap)
		if err != nil {
			return err
		}
		t.feeds = feeds

		fetches, err := m.makeTFOutputs(inputTarget.Fetches, nodeMap)
		if err != nil {
			return err
		}
		t.fetches = fetches
		m.targets[inputTarget.Name] = t

	}
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
	nodeMap := make(map[string]*framework.NodeDef)

	for _, node := range nodes {
		if nameIsPresent(node.Name, nodeNames) {
			nodeMap[node.Name] = node
		}
	}
	return nodeMap, nil
}

func (m *Model) makeTFOperations(operationNames []string, nodeMap map[string]*framework.NodeDef) ([]*tf.Operation, error) {
	var operationSlice []*tf.Operation
	for _, operationName := range operationNames {
		if nodeMap[operationName] != nil {
			operationSlice = append(operationSlice, m.graph.Operation(operationName))
		} else {
			return nil, fmt.Errorf("operation '%s' is not present in graph '%s'", operationName, m.graphPath)
		}
	}
	return operationSlice, nil
}

func (m *Model) makeTFOutputs(outputNames []string, nodeMap map[string]*framework.NodeDef) (map[string]output, error) {
	outputMap := make(map[string]output)
	for _, outputName := range outputNames {
		node := nodeMap[outputName]
		if node != nil {
			output := output{}
			operationName, outputIndex, err := getOutputFromName(outputName)
			if err != nil {
				return nil, err
			}
			output.graphOutput = m.graph.Operation(operationName).Output(outputIndex)
			output.dataType = node.Attr["dtype"].GetType().String()
			for _, size := range node.Attr["shape"].GetShape().GetDim() {
				output.dim = append(output.dim, size.GetSize())
			}
			outputMap[outputName] = output
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

func consolidateTargetNodeNames(targets ...Target) []string {
	var nodeList []string
	for _, target := range targets {
		nodeList = append(nodeList, target.Feeds...)
		nodeList = append(nodeList, target.Fetches...)
		nodeList = append(nodeList, target.Operations...)
	}
	return removeDuplicatesUnordered(nodeList)
}

func removeDuplicatesUnordered(elements []string) []string {
	encountered := map[string]bool{}

	// Create a map of all unique elements.
	for v := range elements {
		encountered[elements[v]] = true
	}

	// Place all keys from the map into a slice.
	result := []string{}
	for key := range encountered {
		result = append(result, key)
	}
	return result
}
