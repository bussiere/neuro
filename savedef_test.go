package pia_test

import (
	"fmt"
	"io/ioutil"
	"testing"

	"github.com/golang/protobuf/proto"
	"github.com/tensortask/tfprotos/core/framework"
)

func TestLoadSaveDef(t *testing.T) {
	graphDef, err := ioutil.ReadFile("graph.pb")
	if err != nil {
		t.Error(err)
	}
	graph := &framework.GraphDef{}

	err = proto.Unmarshal(graphDef, graph)
	if err != nil {
		t.Error(err)
	}
	nodes := graph.GetNode()
	for _, val := range nodes {
		if val.Name == "input_batch" {
			fmt.Println(val)
		}
	}
}
