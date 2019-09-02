import onnx
from onnx import helper
from onnx import AttributeProto, TensorProto, GraphProto
import numpy as np
import cv2
import onnxruntime.backend as backend


# 进行convlution操作
def run_relu_op(x):

    # Create inputs (ValueInfoProto)
    X = helper.make_tensor_value_info('X', TensorProto.FLOAT, list(x.shape))

    # Create output (ValueInfoProto)
    # output形状应该和input相同，
    output_shape = list(x.shape)
    Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, output_shape)

    # create node
    node = onnx.helper.make_node(
        op_type = 'Relu',
        inputs=['X'],
        outputs=['Y'],
    )

    # create graph
    graph_def = helper.make_graph(
        nodes = [node],
        name = 'test-model',
        inputs = [X],
        outputs = [Y],
    )

    # create model
    model_def = helper.make_model(graph_def, producer_name='sun')
    onnx.checker.check_model(model_def)

    # model = onnx.load(onnx_file)
    session = backend.prepare(model_def)
    output = session.run(x)
    output = np.array(output[0])
    print(output)
    print(output.shape)

    #save onnx model
    onnx.save(model_def, 'relu.onnx')
    print("save file relu.onnx")
    output = np.array(output)
    np.save('relu', output)
    print("save output relu.npy")

    return output


if __name__ == "__main__":

    # 指定input
    X = np.load("batchNormalization.npy")

    # 进行单步推理
    run_relu_op(X)

