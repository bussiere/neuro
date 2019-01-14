package neuro

import (
	"encoding/hex"
	"testing"

	"github.com/gogo/protobuf/proto"

	tf "github.com/tensorflow/tensorflow/tensorflow/go"
)

func TestConvertTFToTransport(t *testing.T) {
	testTensor, err := tf.NewTensor([][][]float32{{{1}}, {{2}}, {{3}}})
	if err != nil {
		t.Error(err)
	}
	order := []string{"test"}
	tensors := []*tf.Tensor{testTensor}
	transport, err := convertTFToTransport(order, tensors)
	if err != nil {
		t.Error(err)
	}
	data, err := proto.Marshal(&transport)
	if err != nil {
		t.Error(err)
	}
	hexData := hex.EncodeToString(data)
	if hexData != "121d0a04746573741215080112030301011a0c0000803f0000004000004040" {
		t.Error("the generated TTP transport did not match target for float array")
	}
}

// func TestConvertTransportToTF(t *testing.T) {
// 	model, err := NewModel("testing/graph.pb")
// 	if err != nil {
// 		t.Error(err)
// 	}
// 	target := Target{
// 		Name:    "predict",
// 		Feeds:   []string{"input"},
// 		Fetches: []string{"output"},
// 	}
// 	model.RegisterTargets(target)
// 	sess, err := model.NewSession()
// 	if err != nil {
// 		t.Error(err)
// 	}
// 	err = sess.Init()
// 	if err != nil {
// 		t.Error(err)
// 	}
// 	// In this example we will convert native go types to a TTP transport.
// 	tensors := map[string]interface{}{
// 		"input": [][][]float32{{{1}}, {{2}}, {{3}}},
// 	}

// 	// PackTransport takes a target and a map of tensor aliases and interfaces.
// 	transport, err := PackTransport("predict", tensors)
// 	if err != nil {
// 		panic(err)
// 	}

// 	feeds, fetches, operations, err := model.convertTransportToTF(transport)
// }
