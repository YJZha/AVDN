print("start")
import numpy as np
print("start")
import torch
import torch.nn.functional as F
print("start")
from PIL import Image
print("loading model...")
from transformers import AutoModel, AutoProcessor
print("end loading model")

model_dir = "/mnt/nfs/dell_15td/avdn_1/Aerial-Vision-and-Dialog-Navigation-origin/datasets/Qwen2.5-VL-7B"
processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True, local_files_only=True, use_fast=False)
model = AutoModel.from_pretrained(
    model_dir,
    trust_remote_code=True,
    local_files_only=True,
    dtype=torch.float16,
).cuda().eval()

# 参数
patch = 14
H, W = 16, 16
img = np.zeros((H*patch, W*patch, 3), dtype=np.uint8)

# 生成不同颜色 patch
for i in range(H):
    for j in range(W):
        idx = i*W + j
        img[i*patch:(i+1)*patch, j*patch:(j+1)*patch] = [(idx*37)%255, (idx*97)%255, (idx*131)%255]

img = Image.fromarray(img)

# Qwen 编码
inputs = processor(images=img, text=processor.image_token, return_tensors="pt")
inputs = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in inputs.items()}
with torch.no_grad():
    outputs = model(**inputs, output_hidden_states=True, return_dict=True)

# 取视觉 token
if hasattr(outputs, "vision_hidden_states"):
    tokens = outputs.vision_hidden_states[-1][0]  # S x D
else:
    tokens = outputs.last_hidden_state[0]

S, D = tokens.shape
print("S=", S)

# 打印 image_grid_thw 并输出 7x7 特征
image_grid_thw = inputs.get("image_grid_thw", None)
print("image_grid_thw=", image_grid_thw)
if image_grid_thw is not None:
    grid_thw = image_grid_thw[0].tolist() if hasattr(image_grid_thw, "tolist") else list(image_grid_thw[0])
    _, grid_h, grid_w = grid_thw
    merge_size = processor.image_processor.merge_size
    expected_raw = grid_h * grid_w
    expected_merged = expected_raw
    if grid_h % merge_size != 0 or grid_w % merge_size != 0:
        raise ValueError(f"Grid size {grid_h}x{grid_w} not divisible by merge_size={merge_size}.")
    grid_h_m = grid_h // merge_size
    grid_w_m = grid_w // merge_size
    grid_len = grid_h_m * grid_w_m
    expected_merged = grid_len
    print("expected_raw_tokens=", expected_raw)
    print("expected_merged_tokens=", expected_merged)
    print("actual_tokens=", S)
    if S == expected_raw + 1 or S == expected_merged + 1:
        print("special_token=", "likely +1 (CLS/vision_start/end)")
    else:
        print("special_token=", "no +1 token detected")
    if S < grid_len:
        raise ValueError(
            f"Token length {S} < merged grid size {grid_len} ({grid_h_m}x{grid_w_m})."
        )
    extra = S - grid_len
    print("extra_tokens=", extra)
    if extra > 0:
        front_ok = (S - extra) == grid_len
        back_ok = (S - extra) == grid_len
        print("trim_front_ok=", front_ok)
        print("trim_back_ok=", back_ok)
        if front_ok:
            grid_tokens = tokens[extra:extra + grid_len].reshape(grid_h_m, grid_w_m, D)
            print("trim_strategy=", "front")
        elif back_ok:
            grid_tokens = tokens[:grid_len].reshape(grid_h_m, grid_w_m, D)
            print("trim_strategy=", "back")
        else:
            grid_tokens = tokens[:grid_len].reshape(grid_h_m, grid_w_m, D)
            print("trim_strategy=", "default_back")
    else:
        grid_tokens = tokens[:grid_len].reshape(grid_h_m, grid_w_m, D)

    # 相邻 token cosine 相似度统计（空间连续性检查）
    grid_norm = F.normalize(grid_tokens, dim=-1)
    right = (grid_norm[:, 1:, :] * grid_norm[:, :-1, :]).sum(dim=-1)
    down = (grid_norm[1:, :, :] * grid_norm[:-1, :, :]).sum(dim=-1)
    neighbor_cos = torch.cat([right.flatten(), down.flatten()], dim=0)
    if neighbor_cos.numel() > 0:
        num_pairs = neighbor_cos.numel()
        flat = grid_norm.reshape(-1, D)
        idx_a = torch.randint(0, flat.shape[0], (num_pairs,), device=flat.device)
        idx_b = torch.randint(0, flat.shape[0], (num_pairs,), device=flat.device)
        rand_cos = (flat[idx_a] * flat[idx_b]).sum(dim=-1)
        print("neighbor_cos_mean=", neighbor_cos.mean().item())
        print("random_cos_mean=", rand_cos.mean().item())
    grid_tokens = grid_tokens.permute(2, 0, 1).unsqueeze(0)  # 1 x D x H x W
    grid_7x7 = torch.nn.functional.adaptive_avg_pool2d(grid_tokens, (7, 7))
    print("grid_7x7 shape=", tuple(grid_7x7.shape))