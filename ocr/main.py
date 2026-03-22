"""
从 BMP 目录加载字符图，调用组合增强，按 OCR 识别集格式写出索引与图片。
"""

from __future__ import annotations

from python_tools.combination import generate_combination_rec_dataset


if __name__ == "__main__":
    # 与 val.txt 相同格式；图片与索引在同一数据集根目录下（如 images/val_word_1.png）
    idx_path, img_dir = generate_combination_rec_dataset(
        bmp_folder="/home/unitx/CJM/train_ocr_data/char",
        num_samples=100,
        output_index_file="/home/unitx/CJM/test_dataset_examples/train_combo.txt",
        image_name_fmt="train_word_combin_{index}.png",
        occlusion_ratio_range=(0.0, 0.05),
        seed=0,
        save_images=True,
    )
    print(f"索引: {idx_path}")
    print(f"图片目录: {img_dir}")
