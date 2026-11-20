"""Critical DFL tensor layout and model import contracts without large models."""

import pickle

import numpy as np
import pytest
import torch

from facework.model import depth_to_space, describe_model
from facework.weights import TensorWeights, read_numpy_pickle


def dump_pickle(path, value):
    path.write_bytes(pickle.dumps(value, protocol=4))
    return path


def numpy_dfl_depth_to_space(source, block):
    # Explicit coordinate mapping, independent of the implementation's reshape.
    batch, channels, height, width = source.shape
    output_channels = channels // block**2
    result = np.empty((batch, output_channels, height * block, width * block), source.dtype)
    for n in range(batch):
        for c in range(output_channels):
            for y in range(height * block):
                for x in range(width * block):
                    channel = ((y % block) * block + x % block) * output_channels + c
                    result[n, c, y, x] = source[n, channel, y // block, x // block]
    return result


@pytest.mark.parametrize("shape,block", [((2, 12, 3, 5), 2), ((1, 18, 2, 3), 3)])
def test_dfl_depth_to_space_channel_order_matches_coordinate_reference(shape, block):
    source = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    actual = depth_to_space(torch.from_numpy(source), block).numpy()
    np.testing.assert_array_equal(actual, numpy_dfl_depth_to_space(source, block))


def test_depth_to_space_rejects_incompatible_channels():
    with pytest.raises(ValueError, match="divisible"):
        depth_to_space(torch.zeros(1, 7, 2, 2))


class ExecutablePickle:
    def __reduce__(self):
        return eval, ("40 + 2",)


def test_pickle_global_code_execution_is_rejected(tmp_path):
    path = dump_pickle(tmp_path / "unsafe.npy", ExecutablePickle())
    with pytest.raises(ValueError, match="不支持的模型对象"):
        read_numpy_pickle(path)


def test_real_numpy_arrays_load_with_restricted_reader(tmp_path):
    expected = {"dense/weight:0": np.arange(12, dtype=np.float32).reshape(3, 4)}
    loaded = read_numpy_pickle(dump_pickle(tmp_path / "weights.npy", expected))
    np.testing.assert_array_equal(loaded["dense/weight:0"], expected["dense/weight:0"])


def test_old_four_dimensional_bias_is_flattened_for_convolution(tmp_path):
    # DFL kernels are HWIO; the chosen 1x1 kernel exercises channel conversion.
    arrays = {
        "layer/weight:0": np.array([[[[1, 3], [2, 4]]]], np.float32),
        "layer/bias:0": np.array([[[[7, 11]]]], np.float32),
    }
    weights = TensorWeights(dump_pickle(tmp_path / "old.npy", arrays))
    source = torch.tensor([[[[2., 5.]], [[10., 20.]]]])
    result = weights.conv(source, "layer", use_bias=True).numpy()
    expected = np.array([[[[29., 52.]], [[57., 106.]]]], np.float32)
    np.testing.assert_array_equal(result, expected)


def test_missing_weight_never_falls_back_to_random_initialization(tmp_path):
    weights = TensorWeights(dump_pickle(tmp_path / "partial.npy", {"other/weight:0": np.ones((1,), np.float32)}))
    with pytest.raises(ValueError, match="不会用随机权重替代"):
        weights.tensor("required/weight:0")


@pytest.mark.parametrize("array", [np.array([np.nan], np.float32), np.array([np.inf], np.float32), np.array([1], np.int32)])
def test_nonfinite_or_nonnumeric_weight_is_rejected(tmp_path, array):
    path = dump_pickle(tmp_path / "bad.npy", {"layer/weight:0": array})
    with pytest.raises(ValueError):
        TensorWeights(path)


def metadata_fixture(tmp_path, archi="df-d", resolution=224, face_type="f"):
    prefix = "Example_SAEHD"
    dump_pickle(tmp_path / (prefix + "_data.dat"), {
        "options": {"archi": archi, "resolution": resolution, "face_type": face_type}, "iter": 42,
    })
    for part in ("encoder", "inter", "decoder_src", "decoder_dst"):
        (tmp_path / (prefix + "_" + part + ".npy")).write_bytes(b"component-placeholder")
    return prefix


def test_descriptor_accepts_supported_model_configuration(tmp_path):
    metadata_fixture(tmp_path)
    info = describe_model(tmp_path)
    assert info["archi"] == "df-d"
    assert info["resolution"] == 224
    assert info["iteration"] == 42


@pytest.mark.parametrize("architecture", ["liae", "df-ud", "AMP", "df-t"])
def test_unsupported_architecture_has_explicit_error(tmp_path, architecture):
    metadata_fixture(tmp_path, archi=architecture)
    with pytest.raises(ValueError, match="仅支持 SAEHD df / df-d"):
        describe_model(tmp_path)


def test_missing_decoder_component_is_rejected_before_inference(tmp_path):
    prefix = metadata_fixture(tmp_path)
    (tmp_path / (prefix + "_decoder_dst.npy")).unlink()
    with pytest.raises(ValueError, match="decoder_dst"):
        describe_model(tmp_path)
