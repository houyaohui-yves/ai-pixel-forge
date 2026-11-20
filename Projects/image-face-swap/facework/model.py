"""PyTorch inference adaptation of DeepFaceLab SAEHD df / df-d.

Derived from DeepFaceLab/core/leras/archis/DeepFakeArchi.py and
models/Model_SAEHD/Model.py. GPL-3.0; see LICENSE and THIRD_PARTY.md.
This is a strict inference implementation, not a port of the DFL trainer.
"""
from pathlib import Path
import json
import re

import numpy as np
import torch
import torch.nn.functional as F

from .weights import TensorWeights, read_numpy_pickle


def depth_to_space(x, block=2):
    # TensorFlow DFL channel order is [dy, dx, output_channel], not the
    # [output_channel, dy, dx] order of torch.pixel_shuffle.
    b, c, h, w = x.shape
    if c % (block * block):
        raise ValueError("Pixel shuffle channel count is not divisible by block²")
    return x.reshape(b, block, block, c // block**2, h, w).permute(0, 3, 4, 1, 5, 2).reshape(b, c // block**2, h * block, w * block)


def describe_model(directory, prefix=None):
    directory = Path(directory).resolve()
    if prefix is None:
        files = sorted(directory.glob("*_SAEHD_data.dat"))
        if len(files) != 1:
            raise ValueError("目录需包含一个 SAEHD 模型；多个模型请分别存放")
        prefix = files[0].name.removesuffix("_data.dat")
    data = read_numpy_pickle(directory / (prefix + "_data.dat"))
    if not isinstance(data, dict) or not isinstance(data.get("options"), dict):
        raise ValueError("模型缺少 options 配置")
    options = data["options"]
    archi = str(options.get("archi", ""))
    if archi not in ("df", "df-d"):
        raise ValueError(f"首版仅支持 SAEHD df / df-d，当前为 {archi}。不会猜测其他架构。")
    resolution = int(options.get("resolution", 0))
    if resolution < 64 or resolution > 640 or resolution % 32:
        raise ValueError("模型分辨率需为 64～640 范围内的 32 倍数")
    face_type = options.get("face_type", "f")
    if face_type not in ("h", "mf", "f", "wf"):
        raise ValueError("首版支持 half / mid-full / full / whole face，不支持 head")
    for part in ("encoder", "inter", "decoder_src", "decoder_dst"):
        if not (directory / (prefix + "_" + part + ".npy")).is_file():
            raise ValueError("模型缺少组件：" + part)
    return {"directory": str(directory), "prefix": prefix, "name": prefix,
            "archi": archi, "resolution": resolution, "face_type": face_type,
            "iteration": int(data.get("iter", 0)), "options": options}


class SAEHDPredictor:
    def __init__(self, descriptor, device="cpu"):
        self.info, self.device = descriptor, device
        self.resolution = descriptor["resolution"]
        self.double = descriptor["archi"] == "df-d"
        self.parts = {name: TensorWeights(Path(descriptor["directory"]) / (descriptor["prefix"] + "_" + name + ".npy"), device)
                      for name in ("encoder", "inter", "decoder_src", "decoder_dst")}

    def upscale(self, weights, x, prefix):
        return depth_to_space(F.leaky_relu(weights.conv(x, prefix + "/conv1", use_bias=True), 0.1))

    def residual(self, weights, x, prefix):
        v = F.leaky_relu(weights.conv(x, prefix + "/conv1", use_bias=True), 0.2)
        return F.leaky_relu(x + weights.conv(v, prefix + "/conv2", use_bias=True), 0.2)

    def decode(self, weights, x, mask_only=False):
        m = x
        for i in range(3):
            m = self.upscale(weights, m, f"upscalem{i}")
        if self.double:
            m = self.upscale(weights, m, "upscalem3")
        mask = torch.sigmoid(weights.conv(m, "out_convm", use_bias=True))
        if mask_only:
            return mask
        for i in range(3):
            x = self.residual(weights, self.upscale(weights, x, f"upscale{i}"), f"res{i}")
        if self.double:
            x = depth_to_space(torch.cat([weights.conv(x, p, use_bias=True) for p in ("out_conv", "out_conv1", "out_conv2", "out_conv3")], dim=1))
        else:
            x = weights.conv(x, "out_conv", use_bias=True)
        return torch.sigmoid(x), mask

    @torch.inference_mode()
    def predict(self, aligned_bgr):
        if aligned_bgr.shape != (self.resolution, self.resolution, 3):
            raise ValueError("输入图片尺寸与模型不匹配")
        x = torch.from_numpy(np.ascontiguousarray(aligned_bgr.transpose(2, 0, 1))).float().unsqueeze(0).to(self.device) / 255.0
        encoder, inter = self.parts["encoder"], self.parts["inter"]
        for i in range(4):
            x = F.leaky_relu(encoder.conv(x, f"down1/downs_{i}/conv1", stride=2, use_bias=True), 0.1)
        x = inter.dense(inter.dense(x.flatten(1), "dense1"), "dense2")
        low = self.resolution // (32 if self.double else 16)
        x = x.reshape(1, -1, low, low)
        x = self.upscale(inter, x, "upscale1")
        face, source_mask = self.decode(self.parts["decoder_src"], x)
        target_mask = self.decode(self.parts["decoder_dst"], x, mask_only=True)
        face = face[0].permute(1, 2, 0).cpu().numpy()
        mask = (source_mask * target_mask)[0, 0].cpu().numpy()
        if not np.isfinite(face).all() or not np.isfinite(mask).all():
            raise ValueError("模型产生非有限值，请检查模型或切换 CPU")
        return face, mask
