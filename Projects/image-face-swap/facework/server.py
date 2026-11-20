"""Loopback-only HTTP adapter. All inference remains in FacePipeline."""
import argparse
import hashlib
import json
import mimetypes
import os
from pathlib import Path
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit, unquote
import uuid
import webbrowser

import cv2
import numpy as np

from .geometry import align_face, quality_report
from .model import describe_model
from .pipeline import FacePipeline, BlendOptions
from .storage import MAX_IMAGE_BYTES, Store, decode_image, save_png, write_json

WORKSPACE = Path(__file__).resolve().parents[1]
# Base64 expands image bytes by 4/3; leave room for JSON and other form fields.
MAX_JSON_REQUEST_BYTES = 4 * ((MAX_IMAGE_BYTES + 2) // 3) + 1024**2


class Runtime:
    def __init__(self, data_dir=None, asset_dir=None, device="auto", discover_legacy=True):
        self.store = Store(data_dir or os.environ.get("FACEWORK_DATA_DIR", WORKSPACE / "data"))
        repository = WORKSPACE.parent.parent
        reference_roots = (repository / "LocalReferences", repository)
        asset_candidates = (WORKSPACE / "assets" / "facelib",) + tuple(
            root / "DeepFaceLab" / "facelib" for root in reference_roots
        )
        default_assets = next((path for path in asset_candidates if path.is_dir()), asset_candidates[1])
        self.asset_dir = Path(asset_dir or os.environ.get("FACEWORK_ASSETS_DIR", default_assets)).resolve()
        self.pipeline = FacePipeline(self.asset_dir, device)
        self.lock = threading.RLock()
        self.models = {}
        self.issues = []
        self.registry_path = self.store.root / "models.json"
        self.registry = json.loads(self.registry_path.read_text(encoding="utf-8")) if self.registry_path.exists() else []
        for entry in self.registry:
            try:
                self.add_model(entry["directory"], entry.get("name"), persist=False)
            except (ValueError, OSError) as exc:
                self.issues.append(str(exc))
        for folder in sorted((self.store.root / "models").iterdir()):
            if folder.is_dir():
                try:
                    self.add_model(folder, persist=False)
                except (ValueError, OSError) as exc:
                    self.issues.append(f"{folder.name}: {exc}")
        legacy_candidates = tuple(
            root / "AI face swap " / "AI" / "AI" / "workspace" / "model"
            for root in reference_roots
        )
        legacy = next((path for path in legacy_candidates if path.is_dir()), None)
        if discover_legacy and legacy is not None:
            try:
                self.add_model(legacy, "历史样例 DFD224F · 非朋友模型", persist=False)
            except (ValueError, OSError) as exc:
                self.issues.append(f"旧样例模型不可用：{exc}")

    def add_model(self, directory, name=None, persist=True):
        info = describe_model(directory)
        model_id = hashlib.md5((info["directory"] + info["prefix"]).encode()).hexdigest()
        info["id"] = model_id
        if name:
            info["name"] = str(name).strip()[:100]
        self.models[model_id] = info
        self.pipeline.invalidate_model()
        if persist:
            self.registry = [entry for entry in self.registry if entry["directory"] != info["directory"]]
            self.registry.append({"directory": info["directory"], "name": info["name"]})
            write_json(self.registry_path, self.registry)
        return self.public_models()

    def public_models(self):
        return [{k: model[k] for k in ("id", "name", "resolution", "face_type", "archi", "iteration")} for model in self.models.values()]

    def status(self):
        ready = all((self.asset_dir / name).is_file() for name in ("S3FD.npy", "2DFAN.npy"))
        return {"device": self.pipeline.device, "models": self.public_models(),
                "identities": self.store.identities(), "assets_ready": ready,
                "issues": self.issues + ([] if ready else ["缺少 S3FD.npy / 2DFAN.npy；请配置 FACEWORK_ASSETS_DIR"])}

    def analyze(self, encoded):
        image = decode_image(encoded)
        faces = self.pipeline.detect(image)
        image_id = uuid.uuid4().hex
        directory = self.store.root / "images"
        save_png(directory / (image_id + ".png"), image)
        records = []
        for index, face in enumerate(faces):
            crop, _ = align_face(image, face["landmarks"], 256, "f")
            name = f"{image_id}-{index}.png"
            save_png(directory / name, crop)
            records.append({"index": index, "box": [int(v) for v in face["box"]],
                            "landmarks": np.asarray(face["landmarks"]).tolist(),
                            "quality": quality_report(image, face["box"], face["landmarks"]),
                            "crop_url": "/data/images/" + name})
        info = {"image_id": image_id, "url": f"/data/images/{image_id}.png",
                "width": image.shape[1], "height": image.shape[0], "faces": records}
        write_json(directory / (image_id + ".json"), info)
        return info

    def swap(self, payload):
        image_id = self.store.validate_id(payload["image_id"])
        path = self.store.root / "images" / (image_id + ".png")
        info_path = path.with_suffix(".json")
        if not info_path.exists():
            raise ValueError("原图记录不存在，请重新导入")
        info = json.loads(info_path.read_text(encoding="utf-8"))
        index = int(payload.get("face_index", 0))
        if not 0 <= index < len(info["faces"]):
            raise ValueError("请选择有效的目标人脸")
        model = self.models.get(payload.get("model_id"))
        if model is None:
            raise ValueError("请选择已加载的人物模型")
        image = cv2.imdecode(np.frombuffer(path.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
        options = BlendOptions(int(payload.get("feather", 8)), int(payload.get("erode", 4)), bool(payload.get("color_match", True)))
        result = self.pipeline.process_frame(image, model, face=info["faces"][index], options=options)
        result_id = uuid.uuid4().hex
        save_png(self.store.root / "results" / (result_id + ".png"), result.image)
        save_png(self.store.root / "results" / (result_id + "-mask.png"), result.mask)
        record = {"result_url": f"/data/results/{result_id}.png", "mask_url": f"/data/results/{result_id}-mask.png",
                  "elapsed_ms": result.elapsed_ms, "device": result.device, "warnings": result.warnings}
        write_json(self.store.root / "results" / (result_id + ".json"),
                   {**record, "source_image_id": image_id, "face_index": index, "model_id": model["id"],
                    "model_name": model["name"], "options": options.__dict__})
        return record

    def add_item(self, identity_id, payload):
        self.store.identity_path(identity_id)
        image = decode_image(payload["image"])
        faces = self.pipeline.detect(image)
        if not faces:
            raise ValueError("未检测到人脸，此图片没有加入素材库")
        if len(faces) > 1 and "face_index" not in payload:
            raise ValueError("图片中有多张人脸，请先裁出需要的人物再加入素材库，避免混入其他身份")
        index = int(payload.get("face_index", 0))
        if not 0 <= index < len(faces):
            raise ValueError("素材人脸选择无效")
        item = self.store.add_face(identity_id, image, faces[index], payload.get("filename", "photo"))
        return {"items": [item]}


class LocalServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def make_handler(runtime):
    class Handler(BaseHTTPRequestHandler):
        server_version = "Facework/0.1"

        def log_message(self, format, *args):
            # Do not put original filenames or uploaded content in terminal logs.
            pass

        def allowed(self):
            port = self.server.server_address[1]
            hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}
            if self.headers.get("Host") not in hosts:
                return False
            origin = self.headers.get("Origin")
            return not origin or origin in {"http://" + host for host in hosts}

        def send_bytes(self, data, content_type, status=200, attachment=None):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("Content-Security-Policy", "default-src 'self'; img-src 'self' data: blob:; style-src 'self' 'unsafe-inline'; script-src 'self'; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'")
            if attachment:
                self.send_header("Content-Disposition", f'attachment; filename="{attachment}"')
            self.end_headers()
            self.wfile.write(data)

        def send_json(self, payload, status=200):
            self.send_bytes(json.dumps(payload, ensure_ascii=False, allow_nan=False).encode(), "application/json; charset=utf-8", status)

        def dispatch(self):
            if not self.allowed():
                self.send_json({"error": "仅允许本机页面访问"}, 403)
                return
            parts = [unquote(p) for p in urlsplit(self.path).path.split("/") if p]
            try:
                if self.command == "GET":
                    if parts == ["api", "status"]:
                        with runtime.lock:
                            self.send_json(runtime.status())
                        return
                    if len(parts) in (3, 4) and parts[:2] == ["api", "identities"]:
                        with runtime.lock:
                            if len(parts) == 4 and parts[3] == "export":
                                self.send_bytes(runtime.store.export(parts[2]), "application/zip", attachment="facework-faceset.zip")
                            elif len(parts) == 3:
                                self.send_json(runtime.store.identity(parts[2]))
                            else:
                                raise ValueError("未知素材操作")
                        return
                    if parts and parts[0] == "data":
                        base = runtime.store.root
                        path = base.joinpath(*parts[1:]).resolve()
                        if not path.is_relative_to(base) or path.suffix.lower() not in (".jpg", ".png"):
                            raise ValueError("无效文件路径")
                    else:
                        base = WORKSPACE / "web"
                        path = base.joinpath(*(parts or ["index.html"])).resolve()
                        if not path.is_relative_to(base) or path.suffix not in (".html", ".css", ".js", ".svg", ".ico"):
                            raise ValueError("无效页面路径")
                    if not path.is_file():
                        self.send_json({"error": "文件不存在"}, 404)
                    else:
                        self.send_bytes(path.read_bytes(), mimetypes.guess_type(path.name)[0] or "application/octet-stream")
                    return
                if self.command == "DELETE" and len(parts) == 5 and parts[:2] == ["api", "identities"] and parts[3] == "items":
                    with runtime.lock:
                        runtime.store.delete_item(parts[2], parts[4])
                    self.send_json({"ok": True})
                    return
                if self.command != "POST" or self.headers.get("Content-Type", "").split(";")[0] != "application/json":
                    self.send_json({"error": "不支持的请求"}, 405)
                    return
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= MAX_JSON_REQUEST_BYTES:
                    raise ValueError("请求体为空或超过大小限制")
                payload = json.loads(self.rfile.read(size))
                if not isinstance(payload, dict):
                    raise ValueError("请求必须是 JSON 对象")
                with runtime.lock:
                    if parts == ["api", "analyze"]:
                        response = runtime.analyze(payload["image"])
                    elif parts == ["api", "swap"]:
                        response = runtime.swap(payload)
                    elif parts == ["api", "models"]:
                        response = {"models": runtime.add_model(payload["directory"], payload.get("name"))}
                    elif parts == ["api", "identities"]:
                        response = runtime.store.create_identity(payload["name"])
                    elif len(parts) == 4 and parts[:2] == ["api", "identities"] and parts[3] == "items":
                        response = runtime.add_item(parts[2], payload)
                    else:
                        self.send_json({"error": "接口不存在"}, 404)
                        return
                self.send_json(response)
            except (ValueError, KeyError, TypeError, OSError) as exc:
                self.send_json({"error": str(exc)}, 400)
            except Exception as exc:
                self.send_json({"error": f"处理失败：{type(exc).__name__}: {exc}。可尝试以 --device cpu 启动。"}, 500)

        do_GET = dispatch
        do_POST = dispatch
        do_DELETE = dispatch

    return Handler


def main():
    parser = argparse.ArgumentParser(description="Facework 本地图片换脸工作区")
    parser.add_argument("--port", type=int, default=7861)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--assets", type=Path)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--open-browser", action="store_true")
    args = parser.parse_args()
    runtime = Runtime(args.data, args.assets, args.device)
    server = LocalServer(("127.0.0.1", args.port), make_handler(runtime))
    url = f"http://127.0.0.1:{server.server_address[1]}"
    print(f"Facework: {url}  device={runtime.pipeline.device}", flush=True)
    print(f"Local data: {runtime.store.root}", flush=True)
    if args.open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
