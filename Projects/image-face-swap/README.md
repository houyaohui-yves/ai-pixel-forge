# Facework · 本地图片换脸工作区

这是一个独立的本地浏览器工具：整理人物素材，使用已训练的 DeepFaceLab SAEHD 模型处理图片，检查原图与结果，再保存 PNG。处理核心可以被后续视频程序直接调用。

本项目将 DFL 的检测、关键点、对齐和部分 SAEHD **推理**移植到 PyTorch；没有运行旧版 BAT，也不需要 TensorFlow。原始 `DeepFaceLab/` 和旧整合包保持不变。

## 先了解当前边界

- 支持人脸检测、多人图片中选择目标脸、68 点对齐、SAEHD `df` / `df-d` 推理、学习遮罩融合、羽化、收缩、颜色匹配。
- 支持人物素材库、清晰度/亮度/朝向提示、精确重复去重、删除、导出带 DFLJPG 元数据的全脸 256×256 素材包。原图单独保存。
- **上传朋友照片不会自动训练模型。** 换脸身份来自选择的已训练模型。每个人的人脸素材库和模型分别管理。
- 首版没有训练器、XSeg 遮挡分割、超分、视频处理和帧间一致性模块，也不支持 AMP / Quick96 / LIAE / SAEHD `u` / `t` 架构。
- 自动发现的“历史样例 DFD224F · 非朋友模型”仅用于验证流程，它来自旧包，摘要记录 377 次迭代，质量和身份并非为你的朋友定制。
- 已移植的算子按 DFL 源码处理张量顺序、RGB/BGR、padding、像素重排；尚未完成 TensorFlow 与 PyTorch 全网络数值对照，不能声称完全等价。

## 启动

使用 Python **3.11 或以上**。每台电脑单独创建虚拟环境，不要把 Mac 的 `.venv` 复制到 Windows。

macOS，在此文件所在目录打开终端：

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python run.py --open-browser
```

环境安装好后，也可以双击 `start-mac.command`。当前这台 Mac 的项目虚拟环境已安装依赖。

Windows，在本目录的 PowerShell 中运行：

```powershell
py -3.11 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe run.py --open-browser
```

环境安装好后双击 `start-windows.bat`。Windows 若要使用 RTX 3070 Ti 的 CUDA 加速，需要安装与驱动匹配、带 CUDA 支持的 PyTorch，按 [PyTorch 官方安装选择器](https://pytorch.org/get-started/locally/) 选择；仅有 CPU 版 PyTorch 时仍可处理图片。

默认地址为 `http://127.0.0.1:7861`，仅监听本机。关闭终端或按 Ctrl+C 结束服务。端口冲突时加 `--port 7862`。

设备自动选择顺序：CUDA → MPS → CPU；界面左下角显示实际设备。可使用 `--device cpu` 强制 CPU，或 `--device mps` / `--device cuda` 显式检查加速器。显式设备不可用会报错，不伪装成成功。

当前导入限制：单张文件不超过 30 MiB，宽和高均至少 32 像素，宽 × 高不超过 64,000,000 像素（6400 万）。6192×4128 的相机照片可以直接导入。导入时按 EXIF 纠正方向，保留完整像素尺寸，换脸结果也使用相同尺寸；检测网络和换脸模型内部的人脸裁剪不改变输出尺寸。300 PPI 等打印密度不参与像素上限检查。

## 基础权重和模型

基础检测权重为 `S3FD.npy` 和 `2DFAN.npy`。默认按顺序查找：本工作区的 `assets/facelib/` → 仓库根目录的 `LocalReferences/DeepFaceLab/facelib/` → 原位置 `DeepFaceLab/facelib/`。因此参考资料移动前后均可启动。如果要把本工具单独搬走，请将这两个文件放入 `assets/facelib/`，或指定：

```bash
python run.py --assets /path/to/facelib
```

Windows 路径含空格时用双引号。也可设置 `FACEWORK_ASSETS_DIR`。本工具不会自动联网下载权重。

历史样例模型优先从仓库根目录的 `LocalReferences/AI face swap /AI/AI/workspace/model/` 自动发现，再回退原位置 `AI face swap /AI/AI/workspace/model/`。`AI face swap ` 目录名带尾部空格，移动时保留其名称。手动导入的模型保存的是绝对路径；这类模型目录搬迁后需在界面重新导入。

导入朋友的已训练模型：在“模型与融合”中展开“导入本地训练模型”，填写模型文件夹的绝对路径和显示名称。文件夹需包含同一模型的：

```text
人物名_SAEHD_data.dat
人物名_SAEHD_encoder.npy
人物名_SAEHD_inter.npy
人物名_SAEHD_decoder_src.npy
人物名_SAEHD_decoder_dst.npy
```

一个导入目录仅放一个 SAEHD 模型。也可以把完整模型子目录放入 `data/models/` 后重启。导入先检查配置与必要文件，实际生成时检查权重；缺失权重直接报错，不以随机初始化代替。

## 整理更有价值的素材

1. 为每位朋友创建独立素材库，每张照片只保留对应人物；多人图会要求先裁剪。
2. 采集清晰、不过曝的原始照片，覆盖正脸、左右转头、抬头低头和说话表情。避免大量几乎相同的连续帧。
3. 查看质量提示并人工复核：是否同一人、五官是否对齐、是否有遮挡、是否存在美颜和运动模糊。数值是启发式诊断，不是身份验证或效果评分。
4. 导出 ZIP。`aligned/` 内是包含关键点、原图路径、变换矩阵的 DFLJPG；`sources/` 内是原始上传图片经方向纠正后的 PNG；`manifest.json` 保存质量和来源记录。
5. 人脸包需要进入后续训练流程。当前工具不把这一步标成已完成训练。

要评估换脸效果，请保留不参与训练的照片，覆盖正脸、侧脸、不同光线和遮挡。分别检查身份相似度、眼口结构、肤色和边缘；不要仅凭一个正脸结果判断。遇到明显遮挡时，当前学习遮罩不能替代 XSeg 或手工遮罩修正。

## 视频如何复用

`facework.pipeline.FacePipeline.process_frame` 接收 BGR 图片数组、模型描述和可选的目标脸，返回结果图、融合遮罩、耗时与警告，不依赖浏览器或文件目录。

后续视频程序可以复用同一个 pipeline 和缓存模型，对每帧调用；跟踪器可传入已选择的人脸。视频侧仍需实现身份跟踪、遮挡处理、时序平滑、帧率/时间戳及音画同步。详见 `docs/architecture.md`。

## 数据位置与隐私

代码在 `facework/`、`web/`；导入图片、预览、人物资料、结果、模型路径记录都在 `data/`。`data/`、`assets/`、`.venv/` 已被此工作区的 `.gitignore` 排除。

页面无 CDN、分析服务或外部字体。运行时无上传服务、无外网模型调用。删除素材会删除工作区中的该条记录、对齐脸和源图副本；不会修改你原来选取的外部文件。工作台“清空”只重置页面，已生成文件仍留在 `data/images/` 和 `data/results/`，可在服务停止后自行清理。

## 验证与许可

```bash
.venv/bin/python -m pip install -r requirements-dev.txt
.venv/bin/python -m pytest -q
```

测试覆盖 DFL 几何和通道重排、融合区域外像素保留、DFLJPG 导出、重复素材、EXIF、无效模型和受限权重读取。实际平台验证记录见 `docs/validation.md`；源码包含 Windows 启动方式不代表已经在 Windows 实机完成验收。

派生代码遵循 GPLv3，保留 `LICENSE` 和 `THIRD_PARTY.md`。个人私下修改和使用无需公开项目；若分发软件，需遵守许可证。预训练权重和素材保留其原始许可，代码许可证不自动覆盖所有权重。
