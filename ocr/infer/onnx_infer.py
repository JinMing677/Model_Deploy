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
) -> List[Union[str, Tuple[str, Dict[str, Any]]]]:
    available = ort.get_available_providers()
    providers: List[Union[str, Tuple[str, Dict[str, Any]]]] = []
    if prefer_gpu:
        if try_trt and "TensorrtExecutionProvider" in available:
            trt_opts: Dict[str, Any] = {
                "trt_engine_cache_enable": True,
                "trt_fp16_enable": True,
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


__all__ = [
    "PPOcrRecOnnxGpu",
    "infer_rec_text_gpu",
    "create_rec_session",
    "preprocess_rec_image_bgr",
]


if __name__ == "__main__":
    image = cv2.imread("/home/unitx/CJM/train_ocr_data/test.bmp")

    base = Path(__file__).resolve().parent.parent / "output" / "onnx_model"
    onnx_path = Path(base / "inference.onnx").expanduser().resolve()
    yml_path = Path(base / "inference.yml").expanduser().resolve()
    rec = PPOcrRecOnnxGpu(onnx_path, yml_path, prefer_gpu=True, trt_engine_cache_path=Path(__file__).resolve().parent / "trt_engine_cache")

    for i in range(0,10):
        _ = rec.infer(image)

    start_time = time.time()
    text = rec.infer(image)
    infer_time = time.time() - start_time
    print(f'{text=}')
    print(f'{infer_time=}')