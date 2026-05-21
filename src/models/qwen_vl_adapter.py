import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from transformers import AutoModel, AutoProcessor
except Exception:  # pragma: no cover - optional dependency/runtime mismatch
    AutoModel = None
    AutoProcessor = None


class ProjectionMLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=None, use_layernorm=True, l2_norm=False, dropout=0.0):
        super().__init__()
        hidden_dim = hidden_dim or max(in_dim, out_dim)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )
        self.use_layernorm = use_layernorm
        self.l2_norm = l2_norm
        self.ln = nn.LayerNorm(out_dim) if use_layernorm else None

    def forward(self, x):
        x = x.to(self.net[0].weight.dtype)
        x = self.net(x)
        if self.use_layernorm:
            x = self.ln(x)
        if self.l2_norm:
            x = F.normalize(x, p=2, dim=-1)
        return x


class QwenVLAdapter(nn.Module):
    def __init__(
        self,
        model_name,
        proj_dim=768,
        lang_cls_dim=49,
        et_frame_dim=49,
        et_frame_tokens=512,
        lstm_frame_channels=512,
        lstm_frame_tokens=49,
        use_layernorm=True,
        l2_norm=False,
        dtype="float16",
        device="cuda",
        verbose=False,
    ):
        super().__init__()
        if AutoModel is None or AutoProcessor is None:
            raise RuntimeError("transformers is required to use QwenVLAdapter")

        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True, local_files_only=True)
        torch_dtype = _resolve_dtype(dtype)
        self.qwen_model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            local_files_only=True,
            torch_dtype=torch_dtype,
        ).to(device)

        text_dim, vision_dim = _infer_hidden_sizes(self.qwen_model)
        self.text_proj = ProjectionMLP(text_dim, proj_dim, use_layernorm=use_layernorm, l2_norm=l2_norm)
        self.text_cls_proj = ProjectionMLP(text_dim, lang_cls_dim, use_layernorm=use_layernorm, l2_norm=l2_norm)
        # project vision token channels to 512 (ET expects B x 512 x 49)
        self.et_vis_proj = ProjectionMLP(vision_dim, lstm_frame_channels, use_layernorm=use_layernorm, l2_norm=l2_norm)
        self.lstm_vis_proj = ProjectionMLP(vision_dim, lstm_frame_channels, use_layernorm=use_layernorm, l2_norm=l2_norm)

        self.et_frame_tokens = et_frame_tokens
        self.lstm_frame_tokens = lstm_frame_tokens
        self._diag_printed = False
        self.verbose = verbose

    def encode_with_frame_window(self, texts, frame_histories, window_size, device="cuda"):
        """
        Qwen-as-memory: encode text + a fixed-size sliding window of recent
        frames in one batched forward pass.  Qwen's own attention mechanism
        relates the instruction to the visual history — no external memory
        module is needed.

        texts:          List[str], len = B  (navigation instruction / dialog)
        frame_histories: List[List[np.ndarray]], len = B
                         frame_histories[i] = all frames up to current step
                         (BGR uint8, same format as obs['current_view'])
        window_size:    int W — number of recent frames to keep.
                         Early steps (fewer than W frames) are padded with
                         black images at the front so every item has exactly
                         W image slots → equal-length sequences → one batch.

        Returns
        -------
        text_tokens : [B, seq_len, text_dim]  — for ET lang input
        text_cls    : [B, text_dim]           — for lang_cls projection
        """
        try:
            from PIL import Image as PILImage
        except ImportError:
            raise RuntimeError("Pillow is required for encode_with_frame_window")

        B = len(texts)
        W = max(window_size, 1)

        # Determine image dimensions from the first available frame
        sample_frame = next(
            (f for hist in frame_histories for f in hist if f is not None), None
        )
        if sample_frame is not None:
            img_h, img_w = sample_frame.shape[:2]
        else:
            img_h, img_w = 224, 224
        black_pil = PILImage.fromarray(np.zeros((img_h, img_w, 3), dtype=np.uint8))

        all_prompts = []
        all_images_flat = []   # flat list length B*W, fed to processor

        for text, hist in zip(texts, frame_histories):
            # Take last W frames; pad front with black if fewer than W
            window = list(hist[-W:]) if len(hist) >= W else hist
            n_pad = W - len(window)
            pil_window = [black_pil] * n_pad + [
                PILImage.fromarray(f[:, :, ::-1].astype("uint8")) for f in window
            ]

            content = [{"type": "image"} for _ in range(W)]
            content.append({"type": "text", "text": text})
            messages = [{"role": "user", "content": content}]
            prompt = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            all_prompts.append(prompt)
            all_images_flat.extend(pil_window)   # W images per item

        # Single batched processor call (all items have identical prompt shape)
        inputs = self.processor(
            text=all_prompts,
            images=all_images_flat,
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        outputs = self.qwen_model(
            **inputs,
            output_hidden_states=True,
            return_dict=True,
        )

        hidden = _get_last_hidden(outputs)   # [B, seq_len, text_dim]
        text_cls = hidden[:, 0, :]           # [B, text_dim]
        return hidden, text_cls

    def encode_text(self, texts, device="cuda"):
        inputs = self.processor(text=texts, return_tensors="pt", padding=True)
        input_ids = inputs.get("input_ids").to(device)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        outputs = self.qwen_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        text_tokens = _get_last_hidden(outputs)
        text_cls = text_tokens[:, 0]
        return text_tokens, text_cls

    def encode_image(self, images, device="cuda"):
        image_inputs = self.processor.image_processor(images=images, return_tensors="pt")
        pixel_values = image_inputs.get("pixel_values")
        if pixel_values is None:
            raise ValueError("Image processor did not return pixel_values")
        pixel_values = pixel_values.to(device)
        image_grid_thw = image_inputs.get("image_grid_thw")
        if image_grid_thw is not None:
            image_grid_thw = image_grid_thw.to(device)
        image_embeds = self.qwen_model.get_image_features(pixel_values, image_grid_thw)
        if isinstance(image_embeds, (list, tuple)):
            image_embeds = torch.stack(image_embeds, dim=0)
        return image_embeds

    def project_text(self, text_tokens, text_cls):
        emb_lang = self.text_proj(text_tokens)
        emb_lang_cls = self.text_cls_proj(text_cls)
        return emb_lang, emb_lang_cls

    def project_vision_for_et(self, vision_tokens, image_size=None):
        """Project vision tokens for ET with spatial grid restoration when possible.
        Output: B x 512 x 49 (512 channels, 7x7 spatial positions)."""  # 目标输出形状
        bsz, s, dim = vision_tokens.shape  # 读取 batch、token 数、token 维度
        if self.verbose and not self._diag_printed:  # 只打印一次诊断日志
            cfg = getattr(self.qwen_model.config, 'vision_config', None)  # 读取视觉配置
            ps = getattr(cfg, 'patch_size', None) if cfg is not None else None  # patch 大小
            sm = getattr(cfg, 'spatial_merge_size', None) if cfg is not None else None  # merge 大小
            print(f"[QwenVLAdapter][diag] vision_tokens shape: B={bsz}, S={s}, D={dim}")  # 输出 token 形状
            print(f"[QwenVLAdapter][diag] patch_size={ps}, spatial_merge_size={sm}, image_size={image_size}")  # 输出配置
            if image_size is not None and ps is not None:  # 可推断网格时
                H = image_size[1]  # 图像高
                W = image_size[0]  # 图像宽
                h_patches = H // ps  # 高方向 patch 数
                w_patches = W // ps  # 宽方向 patch 数
                expected_raw = h_patches * w_patches  # 原始 token 数
                expected_merged = expected_raw  # 合并 token 数（默认等于原始）
                if sm is not None and sm > 1:  # 若有 merge
                    expected_merged = (h_patches // sm) * (w_patches // sm)  # 计算合并后 token 数
                print(f"[QwenVLAdapter][diag] expected_raw_tokens={expected_raw}, expected_merged_tokens={expected_merged}")  # 输出预期
                if s == expected_raw:  # 与原始网格一致
                    print("[QwenVLAdapter][diag] token count matches raw patch grid (no merge).")  # 匹配原始
                elif s == expected_merged:  # 与合并网格一致
                    print("[QwenVLAdapter][diag] token count matches merged grid (spatial_merge).")  # 匹配合并
                elif s - 1 == expected_raw or s - 1 == expected_merged:  # 可能多 1 个特殊 token
                    print("[QwenVLAdapter][diag] token count suggests 1 extra special token (CLS/vision_start/end).")  # 提示特殊 token
                else:  # 都不匹配
                    print("[QwenVLAdapter][diag] token count mismatch -> possible resize mismatch or internal reordering/pruning.")  # 提示异常
            else:  # 无法推断网格
                print("[QwenVLAdapter][diag] image_size or patch_size missing; cannot validate token/grid match.")  # 提示缺失
            self._diag_printed = True  # 标记日志已打印
        inferred_h = inferred_w = None  # 初始化推断的网格大小
        cfg = getattr(self.qwen_model.config, 'vision_config', None)  # 读取视觉配置
        if cfg is not None:  # 配置存在时
            ps = getattr(cfg, 'patch_size', None)  # patch 大小
            sm = getattr(cfg, 'spatial_merge_size', None)  # merge 大小
            if image_size is not None and ps is not None:  # 可推断时
                H = image_size[1]  # 图像高
                W = image_size[0]  # 图像宽
                h_patches = H // ps  # 高方向 patch 数
                w_patches = W // ps  # 宽方向 patch 数
                if sm is not None and sm > 1:  # 有 merge
                    h_merged = h_patches // sm  # 合并后高
                    w_merged = w_patches // sm  # 合并后宽
                    if h_merged * w_merged == s:  # token 数匹配合并网格
                        inferred_h, inferred_w = h_merged, w_merged  # 记录合并网格
                else:  # 无 merge
                    if h_patches * w_patches == s:  # token 数匹配原始网格
                        inferred_h, inferred_w = h_patches, w_patches  # 记录原始网格
        if inferred_h is None:  # 配置推断失败时
            r = int(round(s ** 0.5))  # 尝试平方根
            if r * r == s:  # 若 token 数为平方数
                inferred_h = inferred_w = r  # 视为 r×r 网格

        if inferred_h is not None:  # 成功推断网格
            try:
                vision_2d = vision_tokens.view(bsz, inferred_h, inferred_w, dim).permute(0, 3, 1, 2).contiguous()  # 还原为 B×D×H×W
                pooled = torch.nn.functional.adaptive_avg_pool2d(vision_2d, (7, 7))  # 池化到 7×7
                pooled_flat = pooled.view(bsz, dim, 49).transpose(1, 2)  # 变成 B×49×D
                frames = self.et_vis_proj(pooled_flat).transpose(1, 2).contiguous()  # 投影为 B×512×49
                if self.verbose:
                    print(f"[QwenVLAdapter] project_vision_for_et: reshaped {inferred_h}x{inferred_w} -> 7x7 -> {frames.shape}")  # 输出日志
                return frames  # 返回空间恢复结果
            except Exception as e:  # 还原失败则回退
                if self.verbose:
                    print("[QwenVLAdapter] reshaping tokens failed, fallback to pooling, error:", e)  # 打印错误

        # fallback: pool tokens to 49 and project channels  # 回退策略注释
        pooled_tokens = _pool_tokens(vision_tokens, 49)  # 将 token 数压到 49
        frames = self.et_vis_proj(pooled_tokens).transpose(1, 2).contiguous()  # 投影为 B×512×49
        return frames  # 返回回退结果

    def project_vision_for_lstm(self, vision_tokens, image_size=None):
        """Return a 7x7 feature map (B x 512 x 7 x 7)."""
        vision_tokens = self.lstm_vis_proj(vision_tokens)  # B x S x 512

        bsz, s, dim = vision_tokens.shape
        inferred_h = inferred_w = None
        cfg = getattr(self.qwen_model.config, 'vision_config', None)
        if cfg is not None:
            ps = getattr(cfg, 'patch_size', None)
            sm = getattr(cfg, 'spatial_merge_size', None)
            if image_size is not None and ps is not None:
                H = image_size[1]
                W = image_size[0]
                h_patches = H // ps
                w_patches = W // ps
                if sm is not None and sm > 1:
                    h_merged = h_patches // sm
                    w_merged = w_patches // sm
                    if h_merged * w_merged == s:
                        inferred_h, inferred_w = h_merged, w_merged
                else:
                    if h_patches * w_patches == s:
                        inferred_h, inferred_w = h_patches, w_patches
        if inferred_h is None:
            r = int(round(s ** 0.5))
            if r * r == s:
                inferred_h = inferred_w = r

        if inferred_h is not None:
            try:
                vision_2d = vision_tokens.view(bsz, inferred_h, inferred_w, dim).permute(0, 3, 1, 2).contiguous()
                pooled = torch.nn.functional.adaptive_avg_pool2d(vision_2d, (7, 7))  # B x 512 x 7 x 7
                if self.verbose:
                    print(f"[QwenVLAdapter] project_vision_for_lstm: reshaped {inferred_h}x{inferred_w} -> 7x7 -> {pooled.shape}")
                return pooled
            except Exception as e:
                if self.verbose:
                    print("[QwenVLAdapter] reshaping tokens failed in lstm path, fallback to pooling, error:", e)

        vision_tokens = _pool_tokens(vision_tokens, self.lstm_frame_tokens)
        bsz, tokens, channels = vision_tokens.shape
        if tokens != self.lstm_frame_tokens:
            raise ValueError("Unexpected token count after pooling")
        vision_map = vision_tokens.transpose(1, 2).contiguous().view(bsz, channels, 7, 7)
        return vision_map

    def set_qwen_trainable(self, freeze=True, unfreeze_last_n=0):
        for p in self.qwen_model.parameters():
            p.requires_grad = False
        if not freeze:
            if unfreeze_last_n > 0:
                layers = _get_transformer_layers(self.qwen_model)
                if layers:
                    for layer in layers[-unfreeze_last_n:]:
                        for p in layer.parameters():
                            p.requires_grad = True
                else:
                    for p in self.qwen_model.parameters():
                        p.requires_grad = True
            else:
                for p in self.qwen_model.parameters():
                    p.requires_grad = True


def _resolve_dtype(dtype):
    if dtype == "bfloat16":
        return torch.bfloat16
    if dtype == "float32":
        return torch.float32
    return torch.float16


def _get_last_hidden(outputs):
    if hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
        return outputs.last_hidden_state
    if isinstance(outputs, (tuple, list)):
        return outputs[0]
    raise ValueError("Cannot find last_hidden_state in model outputs")


def _get_vision_tokens(outputs):
    if hasattr(outputs, "vision_hidden_states") and outputs.vision_hidden_states is not None:
        return outputs.vision_hidden_states[-1]
    if hasattr(outputs, "image_embeds") and outputs.image_embeds is not None:
        return outputs.image_embeds
    if hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
        return outputs.last_hidden_state
    if isinstance(outputs, (tuple, list)):
        return outputs[0]
    raise ValueError("Cannot find vision tokens in model outputs")


def _pool_tokens(x, target_tokens):
    if x.shape[1] == target_tokens:
        return x
    x = x.transpose(1, 2)
    x = F.adaptive_avg_pool1d(x, target_tokens)
    return x.transpose(1, 2)


def _infer_hidden_sizes(model):
    text_dim = getattr(model.config, "hidden_size", None)
    vision_dim = None
    vision_cfg = getattr(model.config, "vision_config", None)
    if vision_cfg is not None:
        vision_dim = getattr(vision_cfg, "out_hidden_size", None)
        if vision_dim is None:
            vision_dim = getattr(vision_cfg, "hidden_size", None)
    if text_dim is None:
        text_dim = getattr(model.config, "d_model", None)
    if vision_dim is None:
        vision_dim = text_dim
    if text_dim is None or vision_dim is None:
        raise ValueError("Cannot infer hidden sizes from model config")
    return text_dim, vision_dim


def _get_transformer_layers(model):
    for attr in ("model", "transformer", "language_model"):
        if hasattr(model, attr):
            model = getattr(model, attr)
            break
    for layers_attr in ("layers", "h"):
        if hasattr(model, layers_attr):
            layers = getattr(model, layers_attr)
            if isinstance(layers, (list, nn.ModuleList)):
                return list(layers)
    return []
