"""Strict reader for DeepFaceLab's NumPy-pickle weights; no TensorFlow dependency.

DFL weights use pickle despite their .npy suffix. Only NumPy array reconstruction
is permitted here; ordinary pickle.load must not be used for model uploads.
"""
import io
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

try:
    from numpy._core import multiarray as _multiarray
except ImportError:  # NumPy 1.x uses the public spelling in its pickle records.
    from numpy.core import multiarray as _multiarray


class NumpyUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "numpy" and name in ("ndarray", "dtype"):
            return getattr(np, name)
        if module in ("numpy.core.multiarray", "numpy._core.multiarray") and name in ("_reconstruct", "scalar"):
            return getattr(_multiarray, name)
        raise ValueError(f"不支持的模型对象：{module}.{name}")


def read_numpy_pickle(path):
    path = Path(path)
    if path.stat().st_size > 768 * 1024**2:
        raise ValueError("模型文件超过首版支持的 768 MiB 限制")
    return NumpyUnpickler(io.BytesIO(path.read_bytes())).load()


class TensorWeights:
    def __init__(self, path, device="cpu"):
        self.path, self.device = Path(path), device
        self.arrays = read_numpy_pickle(path)
        if not isinstance(self.arrays, dict) or not self.arrays:
            raise ValueError(f"不是有效的 DFL 权重字典：{self.path.name}")
        for key, value in self.arrays.items():
            if not isinstance(key, str) or not isinstance(value, np.ndarray) or value.dtype.kind != "f":
                raise ValueError(f"权重格式错误：{self.path.name}")
            if not np.isfinite(value).all():
                raise ValueError(f"权重包含 NaN 或无穷值：{key}")
        self.cache = {}

    def tensor(self, key):
        if key not in self.arrays:
            raise ValueError(f"模型缺少必要权重 {self.path.name}: {key}，不会用随机权重替代")
        if key not in self.cache:
            self.cache[key] = torch.from_numpy(self.arrays[key].astype(np.float32, copy=True)).to(self.device)
        return self.cache[key]

    def conv(self, x, prefix, stride=1, padding=None, use_bias=None):
        key = prefix + "/weight:0"
        cache_key = key + "@conv"
        if cache_key not in self.cache:
            self.cache[cache_key] = self.tensor(key).permute(3, 2, 0, 1).contiguous()
        w = self.cache[cache_key]
        bkey = prefix + "/bias:0"
        bias = self.tensor(bkey).reshape(-1) if (use_bias is True or bkey in self.arrays) else None
        return F.conv2d(x, w, bias, stride=stride, padding=w.shape[2] // 2 if padding is None else padding)

    def dense(self, x, prefix):
        return x @ self.tensor(prefix + "/weight:0") + self.tensor(prefix + "/bias:0").reshape(-1)

    def bn(self, x, prefix, eps=1e-5):
        def get(n):
            return self.tensor(prefix + "/" + n + ":0").reshape(1, -1, 1, 1)
        return (x - get("running_mean")) / torch.sqrt(get("running_var") + eps) * get("weight") + get("bias")
