import pytest

from edge_ai_bench.artifacts import compare_outputs, inspect_artifact, validate_outputs
from edge_ai_bench.onnx_runtime import OnnxRuntimeConfig, OnnxRuntimeSession


def _write_identity_model(path, onnx, tensor_proto, helper):
    input_info = helper.make_tensor_value_info("signal", tensor_proto.FLOAT, [None, 2])
    output_info = helper.make_tensor_value_info("prediction", tensor_proto.FLOAT, [None, 2])
    graph = helper.make_graph(
        [helper.make_node("Identity", ["signal"], ["prediction"])],
        "identity",
        [input_info],
        [output_info],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)], ir_version=9)
    onnx.save(model, path)


@pytest.mark.integration
def test_onnx_runtime_session_runs_identity_model(tmp_path):
    np = pytest.importorskip("numpy")
    onnx = pytest.importorskip("onnx")
    tensor_proto = onnx.TensorProto
    helper = onnx.helper
    pytest.importorskip("onnxruntime")
    model_path = tmp_path / "identity.onnx"
    _write_identity_model(model_path, onnx, tensor_proto, helper)
    session = OnnxRuntimeSession(model_path, OnnxRuntimeConfig(threads=1, optimization="basic"))
    values = np.array([[1.0, -2.0]], dtype=np.float32)

    outputs = session.run(values)

    assert session.input_name == "signal"
    assert session.output_names == ["prediction"]
    assert session.artifact.size_bytes > 0
    assert len(session.artifact.sha256) == 64
    np.testing.assert_allclose(outputs[0], values)

    comparison = compare_outputs(values, outputs[0])
    assert comparison.allclose
    assert comparison.max_absolute_error == 0.0


def test_runtime_config_rejects_invalid_settings():
    with pytest.raises(ValueError, match="threads"):
        OnnxRuntimeConfig(threads=0)
    with pytest.raises(ValueError, match="optimization"):
        OnnxRuntimeConfig(optimization="fast")


def test_artifact_inspection_and_output_tolerance(tmp_path):
    artifact_path = tmp_path / "model.onnx"
    artifact_path.write_bytes(b"model")

    metadata = inspect_artifact(artifact_path)
    close = compare_outputs([1.0, 2.0], [1.0, 2.00001], atol=1e-4)
    different = compare_outputs([1.0, 2.0], [1.0, 2.1], atol=1e-4)

    assert metadata.size_bytes == 5
    assert metadata.size_kib > 0
    assert close.allclose
    assert not different.allclose


def test_validate_outputs_requires_matching_output_count():
    with pytest.raises(ValueError, match="output counts"):
        validate_outputs([[1.0]], [])
