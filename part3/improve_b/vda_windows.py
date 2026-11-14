"""Video Depth Anything inference split into (1) per-window prediction and (2) stitching.

predict_windows() and stitch_official() are a line-by-line copy of
VideoDepthAnything.infer_video_depth from external/Video-Depth-Anything
(https://github.com/DepthAnything/Video-Depth-Anything, commit 4f5ae23, Apache-2.0,
Copyright (2025) Bytedance Ltd. and/or its affiliates), separated so that the stitching can be
replaced. stitch_official() must reproduce the upstream output bitwise (checked in run_b.py).

Window layout (INFER_LEN 32, OVERLAP 10, KEYFRAMES [0,12,24..31], INTERP_LEN 8): window w starts at
frame 22w; its first 10 input positions are the previous window's inputs at KEYFRAMES, i.e. frame 0,
the previous window's position 12, and the previous window's positions 24..31 (the cross-fade zone).
"""
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

CFG = {"vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]}}


def import_upstream(vda_root):
    sys.path.insert(0, os.path.abspath(vda_root))
    import video_depth_anything.video_depth as vd
    from utils.util import compute_scale_and_shift, get_interpolate_frames
    return vd, compute_scale_and_shift, get_interpolate_frames


def load_model(vda_root, ckpt, encoder="vits"):
    vd, _, _ = import_upstream(vda_root)
    m = vd.VideoDepthAnything(**CFG[encoder])
    m.load_state_dict(torch.load(ckpt, map_location="cpu"), strict=True)
    return m.to("cuda").eval()


def make_transform(vd, frames, input_size):
    import cv2
    from torchvision.transforms import Compose
    frame_height, frame_width = frames[0].shape[:2]
    ratio = max(frame_height, frame_width) / min(frame_height, frame_width)
    if ratio > 1.78:
        input_size = int(input_size * 1.777 / ratio)
        input_size = round(input_size / 14) * 14
    return Compose([
        vd.Resize(width=input_size, height=input_size, resize_target=False, keep_aspect_ratio=True,
                  ensure_multiple_of=14, resize_method="lower_bound", image_interpolation_method=cv2.INTER_CUBIC),
        vd.NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        vd.PrepareForNet(),
    ])


def predict_windows(model, vd, frames, input_size=518, device="cuda", fp32=False):
    """Returns depth_list (flat list, 32 per window, upstream order), frame index and a 'real'
    flag (False for the padded copies of the last frame) for every entry."""
    INFER_LEN, OVERLAP, KEYFRAMES = vd.INFER_LEN, vd.OVERLAP, vd.KEYFRAMES
    frame_height, frame_width = frames[0].shape[:2]
    transform = make_transform(vd, frames, input_size)
    frame_list = [frames[i] for i in range(frames.shape[0])]
    frame_step = INFER_LEN - OVERLAP
    org_video_len = len(frame_list)
    append_frame_len = (frame_step - (org_video_len % frame_step)) % frame_step + (INFER_LEN - frame_step)
    frame_list = frame_list + [frame_list[-1].copy()] * append_frame_len

    depth_list, idx, real = [], [], []
    pre_input, pre_idx, pre_real = None, None, None
    for frame_id in range(0, org_video_len, frame_step):
        cur_list = []
        for i in range(INFER_LEN):
            cur_list.append(torch.from_numpy(transform({"image": frame_list[frame_id + i].astype(np.float32) / 255.0})["image"]).unsqueeze(0).unsqueeze(0))
        cur_input = torch.cat(cur_list, dim=1).to(device)
        cur_idx = [min(frame_id + i, org_video_len - 1) for i in range(INFER_LEN)]
        cur_real = [frame_id + i < org_video_len for i in range(INFER_LEN)]
        if pre_input is not None:
            cur_input[:, :OVERLAP, ...] = pre_input[:, KEYFRAMES, ...]
            cur_idx[:OVERLAP] = [pre_idx[k] for k in KEYFRAMES]
            cur_real[:OVERLAP] = [pre_real[k] for k in KEYFRAMES]

        with torch.no_grad():
            with torch.autocast(device_type=device, enabled=(not fp32)):
                depth = model.forward(cur_input)

        depth = depth.to(cur_input.dtype)
        depth = F.interpolate(depth.flatten(0, 1).unsqueeze(1), size=(frame_height, frame_width), mode="bilinear", align_corners=True)
        depth_list += [depth[i][0].cpu().numpy() for i in range(depth.shape[0])]
        idx += cur_idx
        real += cur_real
        pre_input, pre_idx, pre_real = cur_input, cur_idx, cur_real
    return depth_list, np.array(idx), np.array(real)


def forward_frames(model, vd, frames, sel, input_size=518, device="cuda", fp32=False):
    """One network forward on an arbitrary list of frame indices (len INFER_LEN), e.g. a sparse anchor."""
    frame_height, frame_width = frames[0].shape[:2]
    transform = make_transform(vd, frames, input_size)
    x = torch.cat([torch.from_numpy(transform({"image": frames[i].astype(np.float32) / 255.0})["image"]).unsqueeze(0).unsqueeze(0)
                   for i in sel], dim=1).to(device)
    with torch.no_grad():
        with torch.autocast(device_type=device, enabled=(not fp32)):
            depth = model.forward(x)
    depth = depth.to(x.dtype)
    depth = F.interpolate(depth.flatten(0, 1).unsqueeze(1), size=(frame_height, frame_width), mode="bilinear", align_corners=True)
    return np.stack([depth[i][0].cpu().numpy() for i in range(depth.shape[0])])


def stitch_official(depth_list, org_video_len, vd, compute_scale_and_shift, get_interpolate_frames, metric=False):
    """Upstream stitching, unchanged (copy of infer_video_depth lines after the inference loop)."""
    INFER_LEN, OVERLAP, KEYFRAMES, INTERP_LEN = vd.INFER_LEN, vd.OVERLAP, vd.KEYFRAMES, vd.INTERP_LEN
    depth_list = list(depth_list)
    depth_list_aligned = []
    ref_align = []
    align_len = OVERLAP - INTERP_LEN
    kf_align_list = KEYFRAMES[:align_len]
    scales = []

    for frame_id in range(0, len(depth_list), INFER_LEN):
        if len(depth_list_aligned) == 0:
            depth_list_aligned += depth_list[:INFER_LEN]
            for kf_id in kf_align_list:
                ref_align.append(depth_list[frame_id + kf_id])
        else:
            curr_align = []
            for i in range(len(kf_align_list)):
                curr_align.append(depth_list[frame_id + i])

            if metric:
                scale, shift = 1.0, 0.0
            else:
                scale, shift = compute_scale_and_shift(np.concatenate(curr_align),
                                                       np.concatenate(ref_align),
                                                       np.concatenate(np.ones_like(ref_align) == 1))
            scales.append([float(scale), float(shift)])

            pre_depth_list = depth_list_aligned[-INTERP_LEN:]
            post_depth_list = depth_list[frame_id + align_len:frame_id + OVERLAP]
            for i in range(len(post_depth_list)):
                post_depth_list[i] = post_depth_list[i] * scale + shift
                post_depth_list[i][post_depth_list[i] < 0] = 0
            depth_list_aligned[-INTERP_LEN:] = get_interpolate_frames(pre_depth_list, post_depth_list)

            for i in range(OVERLAP, INFER_LEN):
                new_depth = depth_list[frame_id + i] * scale + shift
                new_depth[new_depth < 0] = 0
                depth_list_aligned.append(new_depth)

            ref_align = ref_align[:1]
            for kf_id in kf_align_list[1:]:
                new_depth = depth_list[frame_id + kf_id] * scale + shift
                new_depth[new_depth < 0] = 0
                ref_align.append(new_depth)

    return np.stack(depth_list_aligned[:org_video_len], axis=0), scales


def stitch_with(windows, org_video_len, params, get_interpolate_frames, OVERLAP=10, INTERP_LEN=8):
    """Upstream blending layout, but with given per-window (scale, shift); params[0] is the first window."""
    align_len = OVERLAP - INTERP_LEN
    out = []
    for w, (s, b) in enumerate(params):
        al = [np.maximum(x * s + b, 0).astype(np.float32) for x in windows[w]]
        if w == 0:
            out += al
        else:
            out[-INTERP_LEN:] = get_interpolate_frames(out[-INTERP_LEN:], al[align_len:OVERLAP])
            out += al[OVERLAP:]
    return np.stack(out[:org_video_len], axis=0)
