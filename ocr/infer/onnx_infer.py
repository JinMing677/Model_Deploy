"""
PP-OCR 识别 ONNX 模型 GPU 推理：按 ``inference.yml`` 做预处理与 CTC 解码。

依赖：``onnxruntime-gpu``、``PyYAML``、``opencv-python``、``numpy``。

TensorRT 引擎缓存默认写在 ONNX 同目录的 ``trt_engine_cache/``，也可用环境变量
``ORT_TRT_ENGINE_CACHE`` 或参数 ``trt_engine_cache_path`` 指定，避免每次启动重新编译。
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import glob
import os
import sys

import cv2
import numpy as np


def _preload_nvidia_libs() -> None:
    """在 import onnxruntime 前预加载 pip 安装的 CUDA / TensorRT 运行库（仅 Linux）。

    同时将库目录追加到 LD_LIBRARY_PATH，确保 onnxruntime 内部 dlopen 也能找到。
    """
    if sys.platform != "linux":
        return
    import ctypes

    site_pkg = os.path.join(os.path.dirname(os.__file__), "site-packages")

    lib_dirs: list[str] = []
    lib_dirs.extend(glob.glob(os.path.join(site_pkg, "nvidia", "*", "lib")))
    trt_dir = os.path.join(site_pkg, "tensorrt_libs")
    if os.path.isdir(trt_dir):
        lib_dirs.append(trt_dir)

    if not lib_dirs:
        return

    ld_path = os.environ.get("LD_LIBRARY_PATH", "")
    for d in lib_dirs:
        if d not in ld_path:
            ld_path = d + (":" + ld_path if ld_path else "")
    os.environ["LD_LIBRARY_PATH"] = ld_path

    _NEEDED = [
        "libcudart.so.12",
        "libcublasLt.so.12",
        "libcublas.so.12",
        "libcudnn.so.9",
        "libcufft.so.11",
        "libcurand.so.10",
        "libnvJitLink.so.12",
        "libnvinfer.so.10",
        "libnvinfer_plugin.so.10",
        "libnvonnxparser.so.10",
    ]
    for lib_name in _NEEDED:
        for d in lib_dirs:
            path = os.path.join(d, lib_name)
            if os.path.isfile(path):
                try:
                    ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
                except OSError:
                    pass
                break


_preload_nvidia_libs()

try:
    import onnxruntime as ort
except ImportError as e:  # pragma: no cover
    raise ImportError("请安装 onnxruntime-gpu: pip install onnxruntime-gpu") from e

try:
    import yaml
except ImportError as e:  # pragma: no cover
    raise ImportError("请安装 PyYAML: pip install pyyaml") from e

Array = np.ndarray

# TensorRT 引擎缓存目录：可通过环境变量覆盖，便于复用、避免每次重新编译引擎。
# 优先级：构造参数 > ORT_TRT_ENGINE_CACHE > 与 ONNX 同目录下的 trt_engine_cache
_TRT_CACHE_ENV = "ORT_TRT_ENGINE_CACHE"


def _resolve_trt_engine_cache_path(
    onnx_path: Union[str, Path],
    explicit: Optional[Union[str, Path]] = None,
) -> str:
    """
    返回 TensorRT 引擎缓存目录的绝对路径字符串。

    使用与 ``onnx_path`` 同目录的固定子目录（不依赖当前工作目录），程序多次运行、
    换目录启动时仍能命中同一套缓存。
    """
    if explicit is not None:
        p = Path(explicit).expanduser().resolve()
    else:
        env = os.environ.get(_TRT_CACHE_ENV, "").strip()
        if env:
            p = Path(env).expanduser().resolve()
        else:
            onnx = Path(onnx_path).expanduser().resolve()
            p = onnx.parent / "trt_engine_cache"
    p.mkdir(parents=True, exist_ok=True)
    return str(p)


def _load_inference_yml(path: Union[str, Path]) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _rec_image_shape_from_yml(cfg: Dict[str, Any]) -> Tuple[int, int, int]:
    for op in cfg.get("PreProcess", {}).get("transform_ops", []):
        if not isinstance(op, dict):
            continue
        if "RecResizeImg" in op:
            sh = op["RecResizeImg"]["image_shape"]
            return int(sh[0]), int(sh[1]), int(sh[2])
    raise ValueError("YAML 中未找到 RecResizeImg.image_shape")


def _character_list_from_yml(cfg: Dict[str, Any]) -> List[str]:
    pp = cfg.get("PostProcess", {})
    chars = pp.get("character_dict")
    if not isinstance(chars, list) or not chars:
        raise ValueError("YAML 中未找到 PostProcess.character_dict")
    char_list = [str(c) for c in chars]
    # Paddle CTCLabelDecode: blank 在首位；末尾追加空格（与 use_space_char=True 对齐）
    return ["blank"] + char_list + [" "]


def preprocess_rec_image_bgr(
    img_bgr: Array,
    image_shape: Tuple[int, int, int],
) -> Array:
    """
    与 ``ppocr/data/imaug/rec_img_aug.resize_norm_img`` 一致（固定宽 padding）。
    ``image_shape`` 为 ``(C, H, W)``，输入为 BGR ``uint8``。
    """
    img_c, img_h, img_w = image_shape
    h, w = img_bgr.shape[:2]
    ratio = w / float(h)
    if math.ceil(img_h * ratio) > img_w:
        resized_w = img_w
    else:
        resized_w = int(math.ceil(img_h * ratio))
    resized = cv2.resize(img_bgr, (resized_w, img_h))
    resized = resized.astype(np.float32)
    if img_c == 1:
        resized = resized / 255.0
        resized = resized[np.newaxis, :]
    else:
        resized = resized.transpose((2, 0, 1)) / 255.0
    resized -= 0.5
    resized /= 0.5
    padding_im = np.zeros((img_c, img_h, img_w), dtype=np.float32)
    padding_im[:, :, 0:resized_w] = resized
    return padding_im


def _decode_ctc(
    preds_idx: Array,
    preds_prob: Optional[Array],
    character: List[str],
    ignored_tokens: Tuple[int, ...] = (0,),
) -> str:
    selection = np.ones(len(preds_idx), dtype=bool)

    selection[1:] = preds_idx[1:] != preds_idx[:-1]

    for ignored_token in ignored_tokens:
        selection &= preds_idx != ignored_token

    char_list = [
        character[text_id] for text_id in preds_idx[selection]
    ]

    if preds_prob is not None:
        conf_list = preds_prob[selection]
    else:
        conf_list = [1] * len(char_list)

    if len(conf_list) == 0:
        conf = 0
    else:
        conf = np.mean(conf_list)

    text = "".join(char_list)

    return "".join(char_list)


def _pick_logits_output(outputs: List[Array], num_classes: int) -> Array:
    for o in outputs:
        if o is None or not isinstance(o, np.ndarray) or o.ndim != 3:
            continue
        # 允许模型维度 >= 字符表长度（多出的维度不会被解码到有效字符）
        if o.shape[-1] >= num_classes or o.shape[1] >= num_classes:
            return o
    raise ValueError(
        f"无法在 ONNX 输出中找到类别维 >= {num_classes} 的 3D logits，"
        f"shapes={[getattr(x, 'shape', None) for x in outputs]}"
    )


def _align_logits_for_ctc(
    logits: Array,
    num_classes: int,
) -> Tuple[Array, Array]:
    """返回 (preds_idx 1D, preds_prob 1D)。"""
    if logits.ndim != 3:
        raise ValueError(f"期望 logits 为 3 维，得到 shape={logits.shape}")
    b, d1, d2 = logits.shape
    if b != 1:
        raise ValueError("当前仅支持 batch_size=1")
    if d2 == num_classes:
        seq = logits[0]
    elif d1 == num_classes:
        seq = logits[0].T
    else:
        raise ValueError(
            f"logits shape {logits.shape} 与 num_classes={num_classes} 无法对齐"
        )
    idx = seq.argmax(axis=-1)
    prob = seq.max(axis=-1)
    return idx, prob


def _build_provider_list(
    prefer_gpu: bool,
    gpu_mem_limit: Optional[int],
    try_trt: bool = True,
    *,
    trt_engine_cache_path: Optional[str] = None,
    trt_fp16: bool = True,
) -> List[Union[str, Tuple[str, Dict[str, Any]]]]:
    available = ort.get_available_providers()
    providers: List[Union[str, Tuple[str, Dict[str, Any]]]] = []
    if prefer_gpu:
        if try_trt and "TensorrtExecutionProvider" in available:
            trt_opts: Dict[str, Any] = {
                "trt_engine_cache_enable": True,
                "trt_fp16_enable": trt_fp16,
            }
            if trt_engine_cache_path:
                trt_opts["trt_engine_cache_path"] = trt_engine_cache_path
            providers.append(("TensorrtExecutionProvider", trt_opts))
        if "CUDAExecutionProvider" in available:
            cuda_opts: Dict[str, Any] = {}
            if gpu_mem_limit is not None:
                cuda_opts["gpu_mem_limit"] = int(gpu_mem_limit)
            providers.append(("CUDAExecutionProvider", cuda_opts) if cuda_opts else "CUDAExecutionProvider")
    providers.append("CPUExecutionProvider")
    return providers


def create_rec_session(
    onnx_path: Union[str, Path],
    *,
    prefer_gpu: bool = True,
    gpu_mem_limit: Optional[int] = None,
    trt_engine_cache_path: Optional[Union[str, Path]] = None,
) -> ort.InferenceSession:
    """创建 ONNX Runtime 会话，优先 TensorRT > CUDA > CPU；TensorRT 加载失败时自动降级。

    TensorRT 引擎会缓存在 ``trt_engine_cache_path`` 指定目录（默认：与 ONNX 同级的
    ``trt_engine_cache``，或通过环境变量 ``ORT_TRT_ENGINE_CACHE`` 设置），下次启动直接加载缓存，
    无需重新编译（模型/TRT 选项未变时）。
    """
    onnx_path_resolved = Path(onnx_path).expanduser().resolve()
    onnx_path = str(onnx_path_resolved)
    trt_cache = _resolve_trt_engine_cache_path(onnx_path_resolved, trt_engine_cache_path)

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    providers = _build_provider_list(
        prefer_gpu, gpu_mem_limit, try_trt=True, trt_engine_cache_path=trt_cache
    )

    has_trt = any(
        (p[0] if isinstance(p, tuple) else p) == "TensorrtExecutionProvider"
        for p in providers
    )
    if has_trt:
        try:
            return ort.InferenceSession(onnx_path, sess_options=so, providers=providers)
        except Exception:
            providers = _build_provider_list(
                prefer_gpu, gpu_mem_limit, try_trt=False, trt_engine_cache_path=trt_cache
            )

    return ort.InferenceSession(onnx_path, sess_options=so, providers=providers)


class PPOcrRecOnnxGpu:
    """加载 ``inference.onnx`` + ``inference.yml``，GPU 推理单行文本。"""

    def __init__(
        self,
        onnx_path: Union[str, Path],
        yml_path: Union[str, Path],
        *,
        prefer_gpu: bool = True,
        gpu_mem_limit: Optional[int] = None,
        trt_engine_cache_path: Optional[Union[str, Path]] = None,
    ):
        self.yml_path = Path(yml_path).expanduser().resolve()
        cfg = _load_inference_yml(self.yml_path)
        self._image_shape = _rec_image_shape_from_yml(cfg)
        self._character = _character_list_from_yml(cfg)
        self._session = create_rec_session(
            onnx_path,
            prefer_gpu=prefer_gpu,
            gpu_mem_limit=gpu_mem_limit,
            trt_engine_cache_path=trt_engine_cache_path,
        )
        inp = self._session.get_inputs()
        self._input_name = inp[0].name
        self._extra_inputs = [i.name for i in inp[1:]]
        # 根据模型输出维度校准字符表长度
        out_shape = self._session.get_outputs()[0].shape
        model_nc = max(d for d in out_shape if isinstance(d, int) and d > 1)
        while len(self._character) < model_nc:
            self._character.append("")
        self._num_classes = model_nc

    @property
    def image_shape(self) -> Tuple[int, int, int]:
        return self._image_shape

    def infer(self, image_bgr: Array) -> str:
        """
        输入 BGR 图像 ``uint8``，``H×W×3``；返回识别字符串。
        """
        if image_bgr.dtype != np.uint8:
            image_bgr = np.clip(image_bgr, 0, 255).astype(np.uint8)
        if image_bgr.ndim == 2:
            image_bgr = cv2.cvtColor(image_bgr, cv2.COLOR_GRAY2BGR)
        if image_bgr.shape[2] == 4:
            image_bgr = cv2.cvtColor(image_bgr, cv2.COLOR_BGRA2BGR)
        blob = preprocess_rec_image_bgr(image_bgr, self._image_shape)
        feed: Dict[str, Any] = {self._input_name: blob[np.newaxis, ...].astype(np.float32)}
        # 部分导出含 valid_ratio 等额外输入，按类型补零（若存在）
        for name in self._extra_inputs:
            meta = next(i for i in self._session.get_inputs() if i.name == name)
            concrete = []
            for d in meta.shape:
                if isinstance(d, int) and d > 0:
                    concrete.append(d)
                else:
                    concrete.append(1)
            if "int64" in meta.type:
                feed[name] = np.zeros(concrete, dtype=np.int64)
            else:
                feed[name] = np.zeros(concrete, dtype=np.float32)
        outs = self._session.run(None, feed)
        logits = _pick_logits_output(outs, self._num_classes)
        idx, _prob = _align_logits_for_ctc(logits, self._num_classes)
        return _decode_ctc(idx, None, self._character)


def infer_rec_text_gpu(
    image: Array,
    onnx_path: Union[str, Path, None] = None,
    yml_path: Union[str, Path, None] = None,
    *,
    session: Optional[PPOcrRecOnnxGpu] = None,
    prefer_gpu: bool = True,
    trt_engine_cache_path: Optional[Union[str, Path]] = None,
) -> str:
    """
    便捷函数：输入 BGR 图片数组，输出文本。

    默认路径指向本仓库导出模型；也可传入已构造的 ``PPOcrRecOnnxGpu`` 以复用会话。

    Parameters
    ----------
    image :
        OpenCV BGR，``uint8``，形状 ``(H,W,3)``（或灰度，会自动转 BGR）。
    onnx_path, yml_path :
        模型与配置；均为 ``None`` 时使用内置默认路径。
    session :
        若提供则忽略 ``onnx_path`` / ``yml_path``，直接使用该会话推理。
    prefer_gpu :
        是否优先使用 CUDA EP。
    trt_engine_cache_path :
        TensorRT 引擎缓存目录；``None`` 时见 ``create_rec_session`` 说明。
    """
    if session is not None:
        return session.infer(image)
    base = Path(__file__).resolve().parent.parent / "output" / "onnx_model"
    onnx_path = Path(onnx_path or base / "inference.onnx").expanduser().resolve()
    yml_path = Path(yml_path or base / "inference.yml").expanduser().resolve()
    rec = PPOcrRecOnnxGpu(
        onnx_path,
        yml_path,
        prefer_gpu=prefer_gpu,
        trt_engine_cache_path=trt_engine_cache_path,
    )
    return rec.infer(image)



# ---------------------------------------------------------------------------
#  Detection: preprocessing / DB post‑processing / ONNX inference
# ---------------------------------------------------------------------------

def _det_preprocess_params_from_yml(
    cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """Extract DetResizeForTest / NormalizeImage params from inference.yml."""
    params: Dict[str, Any] = {}
    resize_kwargs: Dict[str, Any] = {}
    norm_mean = [0.485, 0.456, 0.406]
    norm_std = [0.229, 0.224, 0.225]
    norm_scale = 1.0 / 255.0

    for op in cfg.get("PreProcess", {}).get("transform_ops", []):
        if not isinstance(op, dict):
            continue
        if "DetResizeForTest" in op:
            v = op["DetResizeForTest"]
            if isinstance(v, dict):
                resize_kwargs = v
        if "NormalizeImage" in op:
            v = op["NormalizeImage"]
            if isinstance(v, dict):
                norm_mean = v.get("mean", norm_mean)
                norm_std = v.get("std", norm_std)
                s = v.get("scale", norm_scale)
                if isinstance(s, str):
                    norm_scale = eval(s)
                else:
                    norm_scale = float(s)

    params["resize_kwargs"] = resize_kwargs
    params["norm_mean"] = np.array(norm_mean, dtype=np.float32)
    params["norm_std"] = np.array(norm_std, dtype=np.float32)
    params["norm_scale"] = norm_scale
    return params


def _det_postprocess_params_from_yml(cfg: Dict[str, Any]) -> Dict[str, Any]:
    pp = cfg.get("PostProcess", {})
    return {
        "thresh": float(pp.get("thresh", 0.3)),
        "box_thresh": float(pp.get("box_thresh", 0.6)),
        "max_candidates": int(pp.get("max_candidates", 1000)),
        "unclip_ratio": float(pp.get("unclip_ratio", 1.5)),
        "box_type": str(pp.get("box_type", "quad")),
        "score_mode": str(pp.get("score_mode", "fast")),
    }


def _det_resize_for_test(
    img: Array,
    resize_kwargs: Dict[str, Any],
    max_side_len: int = 960,
) -> Tuple[Array, Array]:
    """Resize to multiples of 32, returns (resized_img, shape_info[src_h, src_w, ratio_h, ratio_w]).

    ``max_side_len`` caps the longest side to prevent GPU OOM on large images
    (applied after the YAML-configured resize logic).
    """
    src_h, src_w = img.shape[:2]
    if src_h + src_w < 64:
        h, w, c = img.shape
        pad = np.zeros((max(32, h), max(32, w), c), np.uint8)
        pad[:h, :w, :] = img
        img = pad

    if "image_shape" in resize_kwargs:
        resize_h, resize_w = resize_kwargs["image_shape"]
        keep_ratio = resize_kwargs.get("keep_ratio", False)
        ori_h, ori_w = img.shape[:2]
        if keep_ratio:
            resize_w = ori_w * resize_h / ori_h
            resize_w = math.ceil(resize_w / 32) * 32
        ratio_h = float(resize_h) / ori_h
        ratio_w = float(resize_w) / ori_w
        img = cv2.resize(img, (int(resize_w), int(resize_h)))
    else:
        limit_side_len = int(resize_kwargs.get("limit_side_len", 736))
        limit_type = resize_kwargs.get("limit_type", "min")
        h, w = img.shape[:2]
        if limit_type == "max":
            ratio = float(limit_side_len) / max(h, w) if max(h, w) > limit_side_len else 1.0
        elif limit_type == "min":
            ratio = float(limit_side_len) / min(h, w) if min(h, w) < limit_side_len else 1.0
        elif limit_type == "resize_long":
            ratio = float(limit_side_len) / max(h, w)
        else:
            ratio = 1.0

        # Clamp the long side to avoid GPU OOM on very large images.
        new_h, new_w = h * ratio, w * ratio
        if max_side_len > 0 and max(new_h, new_w) > max_side_len:
            shrink = float(max_side_len) / max(new_h, new_w)
            ratio *= shrink

        resize_h = max(int(round(h * ratio / 32) * 32), 32)
        resize_w = max(int(round(w * ratio / 32) * 32), 32)
        img = cv2.resize(img, (resize_w, resize_h))
        ratio_h = resize_h / float(src_h)
        ratio_w = resize_w / float(src_w)

    shape_info = np.array([src_h, src_w, ratio_h, ratio_w], dtype=np.float32)
    return img, shape_info


def preprocess_det_image_bgr(
    img_bgr: Array,
    resize_kwargs: Dict[str, Any],
    norm_mean: Array,
    norm_std: Array,
    norm_scale: float,
    max_side_len: int = 960,
) -> Tuple[Array, Array]:
    """Full det preprocessing: resize → normalize → CHW.  Returns (blob_1CHW, shape_info)."""
    img, shape_info = _det_resize_for_test(img_bgr, resize_kwargs, max_side_len=max_side_len)
    img = img.astype(np.float32) * norm_scale
    img = (img - norm_mean) / norm_std
    img = img.transpose((2, 0, 1))[np.newaxis, ...]  # 1×C×H×W
    return img.astype(np.float32), shape_info


class _DBPostProcess:
    """Lightweight DB post‑processing (no PaddlePaddle dependency)."""

    def __init__(
        self,
        thresh: float = 0.3,
        box_thresh: float = 0.6,
        max_candidates: int = 1000,
        unclip_ratio: float = 1.5,
        box_type: str = "quad",
        score_mode: str = "fast",
    ):
        self.thresh = thresh
        self.box_thresh = box_thresh
        self.max_candidates = max_candidates
        self.unclip_ratio = unclip_ratio
        self.box_type = box_type
        self.score_mode = score_mode
        self.min_size = 3

    @staticmethod
    def _get_mini_boxes(
        contour: Array,
    ) -> Tuple[List[List[float]], float]:
        bounding_box = cv2.minAreaRect(contour)
        points = sorted(cv2.boxPoints(bounding_box).tolist(), key=lambda x: x[0])
        if points[1][1] > points[0][1]:
            idx1, idx4 = 0, 1
        else:
            idx1, idx4 = 1, 0
        if points[3][1] > points[2][1]:
            idx2, idx3 = 2, 3
        else:
            idx2, idx3 = 3, 2
        box = [points[idx1], points[idx2], points[idx3], points[idx4]]
        return box, min(bounding_box[1])

    @staticmethod
    def _box_score_fast(bitmap: Array, box: Array) -> float:
        h, w = bitmap.shape[:2]
        _box = box.copy()
        xmin = np.clip(np.floor(_box[:, 0].min()).astype(int), 0, w - 1)
        xmax = np.clip(np.ceil(_box[:, 0].max()).astype(int), 0, w - 1)
        ymin = np.clip(np.floor(_box[:, 1].min()).astype(int), 0, h - 1)
        ymax = np.clip(np.ceil(_box[:, 1].max()).astype(int), 0, h - 1)
        mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
        _box[:, 0] -= xmin
        _box[:, 1] -= ymin
        cv2.fillPoly(mask, _box.reshape(1, -1, 2).astype(np.int32), 1)
        return cv2.mean(bitmap[ymin:ymax + 1, xmin:xmax + 1], mask)[0]

    @staticmethod
    def _unclip(box: Array, unclip_ratio: float) -> list:
        from shapely.geometry import Polygon as ShapelyPolygon
        import pyclipper
        poly = ShapelyPolygon(box)
        if poly.length == 0:
            return []
        distance = poly.area * unclip_ratio / poly.length
        offset = pyclipper.PyclipperOffset()
        offset.AddPath(box.tolist(), pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        return offset.Execute(distance)

    def _boxes_from_bitmap(
        self,
        pred: Array,
        bitmap: Array,
        dest_w: float,
        dest_h: float,
    ) -> Tuple[Array, List[float]]:
        height, width = bitmap.shape
        outs = cv2.findContours(
            (bitmap * 255).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
        )
        contours = outs[-2]
        boxes, scores = [], []
        for contour in contours[: self.max_candidates]:
            points, sside = self._get_mini_boxes(contour)
            if sside < self.min_size:
                continue
            points = np.array(points)
            score = self._box_score_fast(pred, points.reshape(-1, 2))
            if self.box_thresh > score:
                continue
            expanded = self._unclip(points, self.unclip_ratio)
            if len(expanded) != 1:
                continue
            box = np.array(expanded).reshape(-1, 1, 2)
            box, sside = self._get_mini_boxes(box)
            if sside < self.min_size + 2:
                continue
            box = np.array(box)
            box[:, 0] = np.clip(np.round(box[:, 0] / width * dest_w), 0, dest_w)
            box[:, 1] = np.clip(np.round(box[:, 1] / height * dest_h), 0, dest_h)
            boxes.append(box.astype(np.int32))
            scores.append(score)
        return np.array(boxes, dtype=np.int32) if boxes else np.zeros((0, 4, 2), dtype=np.int32), scores

    def _polygons_from_bitmap(
        self,
        pred: Array,
        bitmap: Array,
        dest_w: float,
        dest_h: float,
    ) -> Tuple[List[list], List[float]]:
        height, width = bitmap.shape
        outs = cv2.findContours(
            (bitmap * 255).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
        )
        contours = outs[-2]
        boxes, scores = [], []
        for contour in contours[: self.max_candidates]:
            epsilon = 0.002 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            points = approx.reshape((-1, 2))
            if points.shape[0] < 4:
                continue
            score = self._box_score_fast(pred, points)
            if self.box_thresh > score:
                continue
            expanded = self._unclip(points, self.unclip_ratio)
            if len(expanded) != 1:
                continue
            box = np.array(expanded).reshape(-1, 2)
            if len(box) == 0:
                continue
            _, sside = self._get_mini_boxes(box.reshape((-1, 1, 2)))
            if sside < self.min_size + 2:
                continue
            box[:, 0] = np.clip(np.round(box[:, 0] / width * dest_w), 0, dest_w)
            box[:, 1] = np.clip(np.round(box[:, 1] / height * dest_h), 0, dest_h)
            boxes.append(box.tolist())
            scores.append(score)
        return boxes, scores

    def __call__(
        self,
        pred_map: Array,
        shape_list: Array,
    ) -> List[Dict[str, Any]]:
        """
        Parameters
        ----------
        pred_map : shape ``(B, 1, H, W)`` probability map from the network.
        shape_list : shape ``(B, 4)``  — ``[src_h, src_w, ratio_h, ratio_w]``.

        Returns
        -------
        List of dicts, one per batch, each with key ``"points"`` (list of
        quad boxes as ``(4,2)`` int32 arrays) and ``"scores"``.
        """
        pred = pred_map[:, 0, :, :]
        seg = pred > self.thresh
        results = []
        for bi in range(pred.shape[0]):
            src_h, src_w = shape_list[bi, 0], shape_list[bi, 1]
            mask = seg[bi].astype(np.uint8)
            if self.box_type == "poly":
                boxes, scores = self._polygons_from_bitmap(pred[bi], mask, src_w, src_h)
            else:
                boxes, scores = self._boxes_from_bitmap(pred[bi], mask, src_w, src_h)
            results.append({"points": boxes, "scores": scores})
        return results


def _order_points_clockwise(box: Array) -> Array:
    """Sort four corner points: top‑left, top‑right, bottom‑right, bottom‑left."""
    rect = np.zeros((4, 2), dtype=box.dtype)
    s = box.sum(axis=1)
    rect[0] = box[np.argmin(s)]
    rect[2] = box[np.argmax(s)]
    d = np.diff(box, axis=1).squeeze()
    rect[1] = box[np.argmin(d)]
    rect[3] = box[np.argmax(d)]
    return rect


def _crop_roi(img_bgr: Array, box: Array, expand_px: int = 2) -> Array:
    """Perspective‑transform a quadrilateral ROI to an axis‑aligned rectangle."""
    box = _order_points_clockwise(box.astype(np.float32))
    w = int(max(np.linalg.norm(box[0] - box[1]), np.linalg.norm(box[3] - box[2]))) + expand_px
    h = int(max(np.linalg.norm(box[0] - box[3]), np.linalg.norm(box[1] - box[2]))) + expand_px
    dst = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(box, dst)
    return cv2.warpPerspective(img_bgr, M, (w, h), borderMode=cv2.BORDER_REPLICATE)


def _draw_det_boxes(img_bgr: Array, boxes: Union[Array, list]) -> Array:
    vis = img_bgr.copy()
    for box in boxes:
        pts = np.asarray(box, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(vis, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
    return vis


class PPOcrDetOnnxGpu:
    """PP-OCR 检测 ONNX 推理：加载 ``model.onnx`` + ``inference.yml``，
    返回检测到的文本区域 ROI，可选保存可视化图片。"""

    def __init__(
        self,
        onnx_path: Union[str, Path],
        yml_path: Union[str, Path],
        *,
        prefer_gpu: bool = True,
        gpu_mem_limit: Optional[int] = None,
        trt_engine_cache_path: Optional[Union[str, Path]] = None,
        max_side_len: int = 960,
        try_trt: bool = True,
        trt_fp16: bool = False,
    ):
        self.yml_path = Path(yml_path).expanduser().resolve()
        cfg = _load_inference_yml(self.yml_path)

        pre = _det_preprocess_params_from_yml(cfg)
        self._resize_kwargs = pre["resize_kwargs"]
        self._norm_mean = pre["norm_mean"]
        self._norm_std = pre["norm_std"]
        self._norm_scale = pre["norm_scale"]
        self._max_side_len = max_side_len

        post = _det_postprocess_params_from_yml(cfg)
        self._postprocess = _DBPostProcess(**post)

        onnx_path_resolved = Path(onnx_path).expanduser().resolve()
        trt_cache = _resolve_trt_engine_cache_path(onnx_path_resolved, trt_engine_cache_path)
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        providers = _build_provider_list(
            prefer_gpu, gpu_mem_limit, try_trt=try_trt,
            trt_engine_cache_path=trt_cache, trt_fp16=trt_fp16,
        )
        has_trt = any(
            (p[0] if isinstance(p, tuple) else p) == "TensorrtExecutionProvider"
            for p in providers
        )
        if has_trt:
            try:
                self._session = ort.InferenceSession(
                    str(onnx_path_resolved), sess_options=so, providers=providers,
                )
            except Exception:
                providers = _build_provider_list(
                    prefer_gpu, gpu_mem_limit, try_trt=False, trt_engine_cache_path=trt_cache,
                )
                self._session = ort.InferenceSession(
                    str(onnx_path_resolved), sess_options=so, providers=providers,
                )
        else:
            self._session = ort.InferenceSession(
                str(onnx_path_resolved), sess_options=so, providers=providers,
            )

        self._input_name = self._session.get_inputs()[0].name

    def infer(
        self,
        image_bgr: Array,
        *,
        save_vis: Optional[Union[str, Path]] = None,
    ) -> List[Dict[str, Any]]:
        """
        检测文本区域。

        Parameters
        ----------
        image_bgr : BGR uint8 图像 (H, W, 3)。
        save_vis : 若提供路径，则保存绘制了检测框的可视化图片。

        Returns
        -------
        列表，每个元素为一个 dict::

            {
                "box": np.ndarray (4, 2) int32,  # 四点坐标 (顺时针)
                "score": float,                   # 置信度
                "roi": np.ndarray (h, w, 3),      # 透视校正后的 ROI 图像
            }
        """
        if image_bgr.dtype != np.uint8:
            image_bgr = np.clip(image_bgr, 0, 255).astype(np.uint8)
        if image_bgr.ndim == 2:
            image_bgr = cv2.cvtColor(image_bgr, cv2.COLOR_GRAY2BGR)

        blob, shape_info = preprocess_det_image_bgr(
            image_bgr, self._resize_kwargs, self._norm_mean, self._norm_std, self._norm_scale,
            max_side_len=self._max_side_len,
        )
        pred_map = self._session.run(None, {self._input_name: blob})[0]
        post_results = self._postprocess(pred_map, shape_info[np.newaxis, ...])

        detections: List[Dict[str, Any]] = []
        if post_results:
            boxes = post_results[0]["points"]
            scores = post_results[0]["scores"]
            for box, score in zip(boxes, scores):
                box_arr = np.asarray(box, dtype=np.int32).reshape(-1, 2)
                roi = _crop_roi(image_bgr, box_arr)
                detections.append({"box": box_arr, "score": float(score), "roi": roi})

        if save_vis is not None and len(detections) > 0:
            vis = _draw_det_boxes(image_bgr, [d["box"] for d in detections])
            save_path = Path(save_vis)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(save_path), vis)

        return detections


__all__ = [
    "PPOcrRecOnnxGpu",
    "PPOcrDetOnnxGpu",
    "infer_rec_text_gpu",
    "create_rec_session",
    "preprocess_rec_image_bgr",
    "preprocess_det_image_bgr",
]


if __name__ == "__main__":
    onnx_base = Path(__file__).resolve().parent.parent / "output" / "onnx_model"
    trt_cache = Path(__file__).resolve().parent / "trt_engine_cache"

    rec_base = onnx_base / "rec"
    rec = PPOcrRecOnnxGpu(
        rec_base / "inference.onnx",
        rec_base / "inference.yml",
        prefer_gpu=True,
        trt_engine_cache_path=trt_cache,
    )

    det_base = onnx_base / "det"
    det = PPOcrDetOnnxGpu(
        onnx_path=det_base / "model.onnx",
        yml_path=det_base / "inference.yml",
        prefer_gpu=True,
        trt_engine_cache_path=det_base / "trt_engine_cache",
        max_side_len=960,
    )

    image = cv2.imread("/home/unitx/CJM/Model_Deploy/ocr/ocr_det_dataset_examples/123/2.png")
    vis_dir = Path(__file__).resolve().parent / "det_vis"

    for i in range(10):
        _ = det.infer(image, save_vis=vis_dir / "1_det_roi.png")

    start_time = time.time()
    det_results = det.infer(image)
    det_time = time.time() - start_time

    print(f"Det: {len(det_results)} regions in {det_time:.4f}s")
    for i, r in enumerate(det_results):
        roi = r["roi"]
        cv2.imwrite(vis_dir / f"roi_{i}.png",roi)
        text = rec.infer(roi)
        print(f"  [{i}] score={r['score']:.3f}  text=\"{text}\"  box={r['box'].tolist()}")
