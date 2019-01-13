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
