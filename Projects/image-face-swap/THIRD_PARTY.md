# 来源与许可

本工作区的 DFL 派生代码采用 GNU GPL version 3，许可证全文见 `LICENSE`。

原项目：[iperov/DeepFaceLab](https://github.com/iperov/DeepFaceLab)。本地参考源统一放在仓库根目录 `LocalReferences/DeepFaceLab/`，搬迁前兼容原位置 `DeepFaceLab/`。`LocalReferences/` 的内容由根目录 `.gitignore` 排除，不随本项目提交。

| 本地实现 | 参考来源 | 修改内容 |
| --- | --- | --- |
| `facework/detection.py` | `facelib/S3FDExtractor.py`、`facelib/FANExtractor.py` | 推理迁移到 PyTorch，保留卷积、归一化和坐标约定；输入校验；最多 8 脸 |
| `facework/model.py` | `core/leras/archis/DeepFakeArchi.py`、`models/Model_SAEHD/Model.py`、`core/leras/ops/__init__.py` | 独立 df/df-d 推理、DFL 像素重排、严格模型检查 |
| `facework/geometry.py` | `facelib/LandmarksProcessor.py`、`core/mathlib/umeyama.py` | 33 点模板和对齐，输入校验、启发式质量提示、预乘 alpha 融合 |
| `facework/storage.py` 的 DFLJPG 导出 | `DFLIMG/DFLJPG.py` | 独立 APP15 元数据写入，保留原图和来源变换 |
| `facework/weights.py` | DFL `core/leras/layers/Saveable.py` 的权重格式 | 白名单 NumPy 反序列化，不允许任意 Python 全局对象；缺权重报错 |

DeepFaceLab 的 FAN 文件说明其源自 [1adrianb/face-alignment](https://github.com/1adrianb/face-alignment)。这里保留该来源说明。

PyTorch、NumPy、Pillow、OpenCV 通过独立依赖安装，各自使用其上游许可证。该工作区没有复制或重新分发训练权重；本机权重来自已有 DFL 目录与旧整合包。若以后打包分发权重，应单独核对这些文件的来源和许可。

本工作区新增的网页、HTTP 适配层、素材管理与测试，作为该 GPLv3 派生应用的一部分提供。
