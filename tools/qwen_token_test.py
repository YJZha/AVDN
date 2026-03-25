import sys
from PIL import Image
import numpy as np
import torch

# adapt import path
sys.path.append("/mnt/nfs/dell_15td/avdn_1/Aerial-Vision-and-Dialog-Navigation-origin/src")
sys.path.append("/mnt/nfs/dell_15td/avdn_1/Aerial-Vision-and-Dialog-Navigation-origin")

from models.qwen_vl_adapter import QwenVLAdapter


def make_grid_image(patch_size=14, grid_h=16, grid_w=16):
    H = patch_size * grid_h
    W = patch_size * grid_w
    img = np.zeros((H, W, 3), dtype=np.uint8)
    # assign unique colors per patch
    for i in range(grid_h):
        for j in range(grid_w):
            idx = i * grid_w + j
            r = (idx * 37) % 256
            g = (idx * 97) % 256
            b = (idx * 131) % 256
            img[i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size] = (r, g, b)
    return Image.fromarray(img)


def main():
    model_dir = "/mnt/nfs/dell_15td/avdn_1/Aerial-Vision-and-Dialog-Navigation-origin/datasets/Qwen2.5-VL-7B"
    print("Loading QwenVLAdapter from:", model_dir)
    try:
        adapter = QwenVLAdapter(model_dir, device="cuda")
    except Exception as e:
        print("Failed to initialize QwenVLAdapter:", e)
        return

    img = make_grid_image(patch_size=adapter.qwen_model.config.vision_config.patch_size if hasattr(adapter.qwen_model.config, 'vision_config') else 14,
                          grid_h=16, grid_w=16)
    print("Test image size:", img.size)
    try:
        vision_tokens = adapter.encode_image([img])
        print("vision_tokens type:", type(vision_tokens))
        if hasattr(vision_tokens, 'shape'):
            print("vision_tokens.shape:", vision_tokens.shape)
        else:
            print(repr(vision_tokens)[:200])
    except Exception as e:
        print("Failed to run encode_image:", e)


if __name__ == '__main__':
    main()
