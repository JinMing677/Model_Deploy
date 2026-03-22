"""
将多张 RGB 字符图随机拼接成一行，支持重叠、缩放、纹理遮挡填充。
字典中键 'backend' 保留为纹理图，不参与标签序列。

对外调用请使用入口函数 :func:`combination_augment`，通过关键字参数调整各项随机范围。
"""

from __future__ import annotations

from typing import Dict, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np
from pathlib import Path

try:
    import cv2
except ImportError as e:  # pragma: no cover
    raise ImportError("combination.py 需要 opencv-python: pip install opencv-python") from e

Array = np.ndarray


def _ensure_rgb_uint8(img: Array) -> Array:
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    elif img.shape[2] == 3:
        # 假定输入为 RGB；若为 BGR 请在外部转换
        pass
    else:
        raise ValueError(f"不支持的通道数: {img.shape[2]}")
    return np.ascontiguousarray(img)


def _sample_texture_patch(texture: Array, th: int, tw: int, rng: np.random.Generator) -> Array:
    """从纹理图中随机裁剪 th x tw，不足则平铺。"""
    H, W = texture.shape[:2]
    if th <= H and tw <= W:
        y0 = int(rng.integers(0, H - th + 1))
        x0 = int(rng.integers(0, W - tw + 1))
        return texture[y0 : y0 + th, x0 : x0 + tw].copy()
    # 平铺
    ty = int(rng.integers(0, max(1, H)))
    tx = int(rng.integers(0, max(1, W)))
    patch = np.zeros((th, tw, 3), dtype=np.uint8)
    for yy in range(0, th, H):
        for xx in range(0, tw, W):
            h_end = min(yy + H, th)
            w_end = min(xx + W, tw)
            sh = h_end - yy
            sw = w_end - xx
            ys = (ty + yy) % H
            xs = (tx + xx) % W
            if ys + sh <= H and xs + sw <= W:
                patch[yy:h_end, xx:w_end] = texture[ys : ys + sh, xs : xs + sw]
            else:
                sub = _sample_texture_patch(texture, sh, sw, rng)
                patch[yy:h_end, xx:w_end] = sub
    return patch


def _apply_occlusion(
    rgb: Array,
    texture: Array,
    occlusion_ratio: float,
    rng: np.random.Generator,
    max_rects: int = 32,
) -> Array:
    """用纹理随机遮挡约 occlusion_ratio 面积的像素（矩形并集）。"""
    if occlusion_ratio <= 0:
        return rgb
    h, w = rgb.shape[:2]
    out = rgb.copy()
    target = int(h * w * float(occlusion_ratio))
    covered = np.zeros((h, w), dtype=bool)
    n = 0
    while covered.sum() < target and n < max_rects:
        n += 1
        rh = int(rng.integers(max(1, h // 16), max(2, h // 2 + 1)))
        rw = int(rng.integers(max(1, w // 16), max(2, w // 2 + 1)))
        y0 = int(rng.integers(0, max(1, h - rh + 1)))
        x0 = int(rng.integers(0, max(1, w - rw + 1)))
        patch = _sample_texture_patch(texture, rh, rw, rng)
        sub_cov = covered[y0 : y0 + rh, x0 : x0 + rw]
        fill = ~sub_cov
        sub = out[y0 : y0 + rh, x0 : x0 + rw]
        sub[fill] = patch[fill]
        out[y0 : y0 + rh, x0 : x0 + rw] = sub
        covered[y0 : y0 + rh, x0 : x0 + rw] = True
    return out


def stitch_horizontal_overlap(
    images: Sequence[Array],
    overlap_ratio: Union[float, Sequence[float]],
    background: Tuple[int, int, int] = (255, 255, 255),
    texture: Optional[Array] = None,
    rng: Optional[np.random.Generator] = None,
) -> Array:
    """
    水平拼接。``overlap_ratio`` 可为标量（每对相邻图相同）或长度 ``len(images)-1`` 的序列，
    表示该缝的重叠占 ``min(左宽, 右宽)`` 的比例，取值建议 ``[0, 0.95)``。

    若传入 ``texture``，画布（含纵向对齐产生的上下留白）先用纹理随机平铺/裁切铺底，再叠放各图；
    否则用 ``background`` 纯色填充。
    """
    if not images:
        raise ValueError("images 为空")
    hs = [im.shape[0] for im in images]
    H = max(hs)
    n = len(images)
    if n == 1:
        ratios = []
    elif np.isscalar(overlap_ratio):
        r = float(np.clip(float(overlap_ratio), 0.0, 0.95))
        ratios = [r] * (n - 1)
    else:
        seq = list(overlap_ratio)
        if len(seq) != n - 1:
            raise ValueError(f"overlap_ratio 序列长度应为 {n - 1}，得到 {len(seq)}")
        ratios = [float(np.clip(float(x), 0.0, 0.95)) for x in seq]

    placements: List[Tuple[int, int, Array]] = []
    x_cursor = 0
    for i, im in enumerate(images):
        h, w = im.shape[:2]
        pad_top = (H - h) // 2
        if i == 0:
            placements.append((x_cursor, pad_top, im))
            x_cursor += w
        else:
            prev_w = images[i - 1].shape[1]
            r = ratios[i - 1]
            overlap_px = int(min(prev_w, w) * r)
            overlap_px = max(0, min(overlap_px, w - 1, prev_w - 1))
            x_cursor = x_cursor - overlap_px
            placements.append((x_cursor, pad_top, im))
            x_cursor += w

    W = max(1, x_cursor)
    if texture is not None:
        r = rng or np.random.default_rng()
        canvas = _sample_texture_patch(_ensure_rgb_uint8(texture), H, W, r)
    else:
        canvas = np.full((H, W, 3), background, dtype=np.uint8)
    for x0, y0, im in placements:
        h, w = im.shape[:2]
        canvas[y0 : y0 + h, x0 : x0 + w] = im
    return canvas


def combine_line_augmentation(
    char_images: Dict[str, Array],
    *,
    concat_count_range: Tuple[int, int] = (2, 6),
    overlap_ratio_range: Tuple[float, float] = (0.0, 0.25),
    scale_range: Tuple[float, float] = (0.6, 1.2),
    occlusion_ratio_range: Tuple[float, float] = (0.0, 0.15),
    charset: Optional[Sequence[str]] = None,
    rng: Optional[np.random.Generator] = None,
    background: Tuple[int, int, int] = (255, 255, 255),
) -> Tuple[Array, str]:
    """
    从 char_images 中随机选若干张字符图（不含 'backend'），缩放、遮挡后水平拼接。

    Parameters
    ----------
    char_images :
        字符 -> RGB 图像。键 'backend' 为纹理图，用于遮挡填充，不出现在标签中。
    concat_count_range :
        拼接段数 [min, max]，闭区间。
    overlap_ratio_range :
        相邻图重叠占 min(左宽, 右宽) 的比例范围。
    scale_range :
        每张字符图相对原尺寸的缩放系数范围。
    occlusion_ratio_range :
        单张图上用纹理替换的近似面积比例范围。
    charset :
        允许参与拼接的字符集合；默认使用除 'backend' 外所有键。
    rng :
        随机数生成器；默认 ``np.random.default_rng()``。
    background :
        画布纯色后备；实际拼接时优先用 ``'backend'`` 纹理铺满画布（含缩放后高度不齐时的上下留白）。

    Returns
    -------
    image : ndarray, uint8, shape (H, W, 3), RGB
    label : str
        按从左到右拼接顺序连接的字符（每个键一个字符；多字符键则整体作为一段标签）。
    """
    rng = rng or np.random.default_rng()
    if "backend" not in char_images:
        raise KeyError("char_images 必须包含键 'backend' 作为纹理图")

    texture = _ensure_rgb_uint8(char_images["backend"])
    keys = list(char_images.keys())
    keys_no_tex = [k for k in keys if k != "backend"]
    if not keys_no_tex:
        raise ValueError("除 'backend' 外至少需要一张字符图")

    if charset is not None:
        allowed = [c for c in charset if c in char_images and c != "backend"]
        if not allowed:
            raise ValueError("charset 中没有可用的字符键")
        pool = allowed
    else:
        pool = keys_no_tex

    low, high = concat_count_range
    low = max(1, int(low))
    high = max(low, int(high))
    n = int(rng.integers(low, high + 1))

    chosen: List[str] = [str(rng.choice(pool)) for _ in range(n)]
    label = "".join(chosen)

    processed: List[Array] = []
    o_low, o_high = overlap_ratio_range
    s_low, s_high = scale_range
    occ_low, occ_high = occlusion_ratio_range

    for ch in chosen:
        img = _ensure_rgb_uint8(char_images[ch])
        h0, w0 = img.shape[:2]
        scale = float(rng.uniform(s_low, s_high))
        nh = max(1, int(round(h0 * scale)))
        nw = max(1, int(round(w0 * scale)))
        resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)

        occ = float(rng.uniform(occ_low, occ_high))
        resized = _apply_occlusion(resized, texture, occ, rng)
        processed.append(resized)

    if len(processed) <= 1:
        ratios: List[float] = []
    else:
        ratios = [float(rng.uniform(o_low, o_high)) for _ in range(len(processed) - 1)]
    out = stitch_horizontal_overlap(
        processed,
        ratios if ratios else 0.0,
        background=background,
        texture=texture,
        rng=rng,
    )
    return out, label


def combination_augment(
    char_images: Dict[str, Array],
    *,
    concat_count_range: Tuple[int, int] = (2, 6),
    overlap_ratio_range: Tuple[float, float] = (0.0, 0.25),
    scale_range: Tuple[float, float] = (0.6, 1.2),
    occlusion_ratio_range: Tuple[float, float] = (0.0, 0.15),
    charset: Optional[Sequence[str]] = None,
    seed: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
    background: Tuple[int, int, int] = (255, 255, 255),
) -> Tuple[Array, str]:
    """
    入口函数：传入字符图字典与各项范围，返回拼接后的 RGB 图与标签字符串。

    Parameters
    ----------
    char_images
        字符键 → RGB 图像；必须含 ``'backend'`` 纹理图。
    concat_count_range
        拼接段数 ``[min, max]``（闭区间）。
    overlap_ratio_range
        相邻两图重叠占 ``min(左宽, 右宽)`` 的比例范围，每一缝单独随机。
    scale_range
        每张字符图相对原图的缩放系数范围。
    occlusion_ratio_range
        每张字符图上用纹理替换的近似面积比例范围。
    charset
        参与抽样的字符键；``None`` 表示除 ``'backend'`` 外全部键。
    seed
        随机种子；与 ``rng`` 同时传入时以 ``rng`` 为准。
    rng
        ``numpy.random.Generator``，用于可复现实验。
    background
        无纹理铺底时的纯色 ``(R, G, B)``；本流程会传入 ``backend`` 纹理铺画布，一般仅作后备。

    Returns
    -------
    image : ndarray
        ``uint8``，``(H, W, 3)``，RGB。
    label : str
        从左到右各段键名拼接（不含 ``backend``）。
    """
    if rng is None and seed is not None:
        rng = np.random.default_rng(seed)
    return combine_line_augmentation(
        char_images,
        concat_count_range=concat_count_range,
        overlap_ratio_range=overlap_ratio_range,
        scale_range=scale_range,
        occlusion_ratio_range=occlusion_ratio_range,
        charset=charset,
        rng=rng,
        background=background,
    )

def load_bmp_folder_as_char_dict(folder: str | Path) -> Dict[str, Array]:
    root = Path(folder).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"不是有效目录: {root}")

    out: Dict[str, Array] = {}
    candidates = sorted(
        p for p in root.rglob("*") if p.is_file() and p.suffix.lower() == ".bmp"
    )

    for path in candidates:
        key = path.stem
        if key in out:
            continue
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is None:
            continue
        if img.ndim == 2:
            rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 4:
            rgb = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        elif img.shape[2] == 3:
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            continue
        if rgb.dtype != np.uint8:
            rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        out[key] = rgb

    return out


def _save_rgb_png(path: Path, rgb: Array) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(str(path), bgr):
        raise OSError(f"无法写入图片: {path}")


def generate_combination_rec_dataset(
    *,
    num_samples: int,
    output_index_file: str | Path,
    bmp_folder: str | Path | None = None,
    char_images: Optional[Dict[str, Array]] = None,
    image_subdir: str = "images",
    image_name_fmt: str = "val_word_{index}.png",
    start_index: int = 1,
    index_mode: Literal["w", "a"] = "w",
    encoding: str = "utf-8",
    concat_count_range: Tuple[int, int] = (2, 6),
    overlap_ratio_range: Tuple[float, float] = (0.0, 0.25),
    scale_range: Tuple[float, float] = (0.6, 1.2),
    occlusion_ratio_range: Tuple[float, float] = (0.0, 0.15),
    charset: Optional[Sequence[str]] = None,
    seed: Optional[int] = None,
    background: Tuple[int, int, int] = (255, 255, 255),
    save_images: bool = True,
) -> Tuple[Path, Path]:
    """
    生成若干条组合增强样本，将图片保存为 PNG，并写入与 ``ocr_rec_dataset_examples/val.txt`` 相同的格式：
    每行 ``相对路径<TAB>标签``（相对路径相对于索引文件所在目录）。

    图片目录：``Path(output_index_file).parent / image_subdir /``（与索引中的相对路径一致）。

    Parameters
    ----------
    num_samples :
        生成图片条数。
    output_index_file :
        索引文件路径，例如 ``.../ocr_rec_dataset_examples/val.txt``。
    bmp_folder :
        BMP 根目录；与 ``char_images`` 二选一。
    char_images :
        已构建好的 ``dict``（须含 ``backend`` 纹理）；若提供则忽略 ``bmp_folder``。
    image_subdir :
        图片子目录名，默认 ``images``，与示例中 ``images/val_word_1.png`` 一致。
    image_name_fmt :
        文件名模板，占位符 ``{index}`` 为整数编号。
    start_index :
        起始编号（含）。
    index_mode :
        写索引文件：``"w"`` 覆盖，``"a"`` 追加。
    encoding :
        索引文件编码。
    save_images :
        为 ``True`` 时将每张生成图写入上述图片目录；为 ``False`` 只写索引（一般不推荐）。
    concat_count_range, overlap_ratio_range, scale_range, occlusion_ratio_range, charset, seed, background :
        传给 :func:`python_tools.combination.combination_augment`。

    Returns
    -------
    tuple[Path, Path]
        ``(索引文件绝对路径, 图片目录绝对路径)``。
    """
    if num_samples < 1:
        raise ValueError("num_samples 至少为 1")
    if char_images is None:
        if bmp_folder is None:
            raise ValueError("必须指定 bmp_folder 或 char_images")
        char_images = load_bmp_folder_as_char_dict(bmp_folder)
    else:
        char_images = dict(char_images)

    index_path = Path(output_index_file).expanduser().resolve()
    dataset_root = index_path.parent
    img_root = (dataset_root / image_subdir).resolve()
    if save_images:
        img_root.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    lines: list[str] = []

    for k in range(num_samples):
        idx = start_index + k
        name = image_name_fmt.format(index=idx)
        rel = f"{image_subdir}/{name}".replace("\\", "/")
        img, label = combination_augment(
            char_images,
            concat_count_range=concat_count_range,
            overlap_ratio_range=overlap_ratio_range,
            scale_range=scale_range,
            occlusion_ratio_range=occlusion_ratio_range,
            charset=charset,
            rng=rng,
            background=background,
        )
        label_clean = label.replace("\t", " ").replace("\n", " ").replace("\r", " ")
        if save_images:
            out_img = img_root / name
            _save_rgb_png(out_img, img)
        lines.append(f"{rel}\t{label_clean}\n")

    index_path.parent.mkdir(parents=True, exist_ok=True)
    with open(index_path, index_mode, encoding=encoding, newline="") as f:
        f.writelines(lines)

    return index_path, img_root


__all__ = [
    "combination_augment",
    "combine_line_augmentation",
    "stitch_horizontal_overlap",
    "generate_combination_rec_dataset",
    "load_bmp_folder_as_char_dict",
]
