package neuro

import (
	"bytes"
	"errors"
	"fmt"

	tf "github.com/tensorflow/tensorflow/tensorflow/go"
	ttp "github.com/tensortask/ttp/gen"
)

// convertTransportToTF takes a TTP transport and returns maps containing
// the input data read into tf.Tensors. The fetches are in the format of
// a [string]Output map to help with repacking the transports upon processing.
func (m Model) convertTransportToTF(transport ttp.Transport) (map[tf.Output]*tf.Tensor, map[string]output, []*tf.Operation, error) {
	target := m.targets[transport.Target]
	tensors := transport.Tensors
	feeds := make(map[tf.Output]*tf.Tensor)
	for alias, tensor := range tensors {
		output, ok := target.feeds[alias]
		if !ok {
			return nil, nil, nil, fmt.Errorf("alias %s not found in target %s", alias, target.name)
		}
		tfDataType, err := ttpTypeToTf(tensor.GetType())
		if err != nil {
			return nil, nil, nil, err
		}
		contents := bytes.NewReader(tensor.Contents)
		readTensor, err := tf.ReadTensor(tfDataType, tensor.Dim, contents)
		feeds[output.graphOutput] = readTensor
	}

	fetches := target.fetches
	ops := target.operations
	return feeds, fetches, ops, nil
}

// convertTFToTransport converts an array of aliases and a corresponding tensors
// to a packed TTP transport.
func convertTFToTransport(order []string, tensors []*tf.Tensor) (ttp.Transport, error) {
	if len(order) != len(tensors) {
		return ttp.Transport{}, fmt.Errorf("length of order does not match the supplied tensors to convert to TTP")
	}
	transport := ttp.Transport{
		Tensors: make(map[string]*ttp.Tensor),
	}

	for index, alias := range order {
		ttpType, err := tfTypeToTtp(tensors[index].DataType())
		if err != nil {
			return ttp.Transport{}, err
		}
		var contents bytes.Buffer
		_, err = tensors[index].WriteContentsTo(&contents)
		if err != nil {
			return ttp.Transport{}, err
		}

		ttpTensor := ttp.Tensor{
			Type:     ttpType,
			Dim:      tensors[index].Shape(),
			Contents: contents.Bytes(),
		}
		transport.Tensors[alias] = &ttpTensor
	}
	return transport, nil
}

// tfTypeToTtp is a helper function that converts a tf data type
// to a TTP type. Driven by a simple switch statement.
func tfTypeToTtp(dataType tf.DataType) (ttp.Type, error) {
	switch dataType {
	case tf.Float:
		return ttp.Type_FLOAT, nil
	case tf.Double:
		return ttp.Type_DOUBLE, nil
	case tf.Int32:
		return ttp.Type_INT32, nil
	case tf.Uint8:
		return ttp.Type_UINT8, nil
	case tf.Int16:
		return ttp.Type_INT16, nil
	case tf.Int8:
		return ttp.Type_INT8, nil
	// case tf.String:
	// 	return ttp.Type_STRING, nil
	case tf.Complex64:
		return ttp.Type_COMPLEX64, nil
	case tf.Int64:
		return ttp.Type_INT64, nil
	case tf.Bool:
		return ttp.Type_BOOL, nil
	case tf.Qint8:
		return ttp.Type_QINT8, nil
	case tf.Quint8:
		return ttp.Type_QUINT8, nil
	case tf.Qint32:
		return ttp.Type_QINT32, nil
	case tf.Bfloat16:
		return ttp.Type_BFLOAT16, nil
	case tf.Qint16:
		return ttp.Type_QINT16, nil
	case tf.Quint16:
		return ttp.Type_QUINT16, nil
	case tf.Uint16:
		return ttp.Type_UINT16, nil
	case tf.Complex128:
		return ttp.Type_COMPLEX128, nil
	case tf.Half:
		return ttp.Type_HALF, nil
	case tf.Uint32:
		return ttp.Type_UINT32, nil
	case tf.Uint64:
		return ttp.Type_UINT64, nil
	default:
		return 0, errors.New("invalid data type")
	}
}

// ttpTypeToTf is a helper function that converts a TTP data type
// to a TensorFlow type.
func ttpTypeToTf(dataType ttp.Type) (tf.DataType, error) {
	switch dataType.String() {
	case "FLOAT":
		return tf.Float, nil
	case "DOUBLE":
		return tf.Double, nil
	case "INT32":
		return tf.Int32, nil
	case "UINT8":
		return tf.Uint8, nil
	case "INT16":
		return tf.Int16, nil
	case "INT8":
		return tf.Int8, nil
	// case "STRING":
	// 	return tf.String, nil
	case "COMPLEX64":
		return tf.Complex64, nil
	case "INT64":
		return tf.Int64, nil
	case "BOOL":
		return tf.Bool, nil
	case "QINT8":
		return tf.Qint8, nil
	case "QUINT8":
		return tf.Quint8, nil
	case "QINT32":
		return tf.Qint32, nil
	case "BFLOAT16":
		return tf.Bfloat16, nil
	case "QINT16":
		return tf.Qint16, nil
	case "QUINT16":
		return tf.Quint16, nil
	case "UINT16":
		return tf.Uint16, nil
	case "COMPLEX128":
		return tf.Complex128, nil
	case "HALF":
		return tf.Half, nil
	case "UINT32":
		return tf.Uint32, nil
	case "UINT64":
		return tf.Uint64, nil
	default:
		return 0, fmt.Errorf("invalid data type %v", dataType.String())
	}
}
