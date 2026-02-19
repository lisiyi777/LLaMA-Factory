#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

def xywh_to_xyxy(b_xywh):
    x, y, w, h = map(float, b_xywh)
    return [x, y, x + w, y + h]

def load_jsonl(path, max_samples=None):
    out = []
    with open(path, "r") as f:
        for i, line in enumerate(f):
            if max_samples is not None and i >= max_samples:
                break
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def scale_xyxy_from_resized(b_xyxy, W, H, resized_size=1000):
    """Map bbox from resized_size×resized_size coords back to original (W,H)."""
    x1, y1, x2, y2 = map(float, b_xyxy)
    sx = W / float(resized_size)
    sy = H / float(resized_size)
    return [x1 * sx, y1 * sy, x2 * sx, y2 * sy]


def clamp_xyxy(b_xyxy, W, H):
    x1, y1, x2, y2 = map(float, b_xyxy)
    x1 = max(0.0, min(x1, W - 1))
    y1 = max(0.0, min(y1, H - 1))
    x2 = max(0.0, min(x2, W - 1))
    y2 = max(0.0, min(y2, H - 1))
    if x2 < x1: x1, x2 = x2, x1
    if y2 < y1: y1, y2 = y2, y1
    return [x1, y1, x2, y2]


def draw_boxes(
    img: Image.Image,
    gt_boxes,
    vqa_scores,
    thr: float,
    coords_are_resized_1000: bool = True,
    resized_size: int = 1000,
    show_gt=True,
    show_text=True,
):
    """
    gt_boxes: list of dicts with keys: bbox_2d (xyxy), label
    vqa_scores: list of dicts with keys: bbox_2d (xyxy), label, score

    If coords_are_resized_1000=True, bbox_2d are in 1000x1000 coords and will be scaled to img size.
    """
    out = img.convert("RGB").copy()
    draw = ImageDraw.Draw(out)
    W, H = out.size

    font = None
    if show_text:
        try:
            font = ImageFont.load_default()
        except Exception:
            font = None

    # n = min(len(gt_boxes), len(vqa_scores))
    n = len(vqa_scores) if not gt_boxes else min(len(gt_boxes), len(vqa_scores))

    def to_img_space(b):
        if coords_are_resized_1000:
            b = scale_xyxy_from_resized(b, W, H, resized_size=resized_size)
        return clamp_xyxy(b, W, H)

    # 1) Draw GT boxes (thin BLUE)
    if show_gt and gt_boxes:
        for i in range(min(len(gt_boxes), n)):
            b = to_img_space(gt_boxes[i]["bbox_2d"])
            x1, y1, x2, y2 = b
            draw.rectangle([x1, y1, x2, y2], outline="blue", width=2)
            if show_text:
                label = gt_boxes[i].get("label", "obj")
                draw.text((x1 + 2, y1 + 2), f"GT:{label}", fill="blue", font=font)

    # 2) Draw TP/FP overlays (GREEN/RED) based on VQA score
    for i in range(n):
        b = to_img_space(vqa_scores[i]["bbox_2d"])
        score = float(vqa_scores[i]["score"])
        # label = vqa_scores[i].get("label", gt_boxes[i].get("label", "obj"))
        label = vqa_scores[i].get("label", "obj")


        x1, y1, x2, y2 = b
        is_tp = score >= thr
        color = "green" if is_tp else "red"
        draw.rectangle([x1, y1, x2, y2], outline=color, width=4)

        if show_text:
            tag = "TP" if is_tp else "FP"
            draw.text((x1 + 2, y1 + 14), f"{tag}:{label} {score:.2f}", fill=color, font=font)

    return out


def save_preview_grid(images, out_path, cols=4):
    n = len(images)
    rows = (n + cols - 1) // cols
    plt.figure(figsize=(4 * cols, 4 * rows))
    for i, im in enumerate(images):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(im)
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(str(out_path), dpi=200)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--max_samples", type=int, default=32)
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--grid", action="store_true")

    # NEW:
    ap.add_argument("--coords_are_resized_1000", action="store_true",
                    help="Interpret bbox_2d coords as in 1000x1000 resized space and scale to original image.")
    ap.add_argument("--resized_size", type=int, default=1000)

    ap.add_argument("--no_gt", action="store_true")
    args = ap.parse_args()

    records = load_jsonl(args.jsonl, max_samples=args.max_samples)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    previews = []

    for rec in records:
        img_path = rec["images"][0]
        sample_idx = rec.get("sample_idx", None)
        if sample_idx is None:
            sample_idx = rec.get("image_id", -1)   # pred records

        # gt_boxes = rec.get("gt_boxes", [])
        # vqa_scores = rec.get("vqa_score", [])

        # If this is GT-style jsonl:
        if "gt_boxes" in rec and rec["gt_boxes"]:
            gt_boxes = rec.get("gt_boxes", [])
            vqa_scores = rec.get("vqa_score", [])

        # If this is PRED-style jsonl:
        else:
            gt_boxes = []  # no GT available in pred jsonl (unless you add it later)
            vqa_scores = rec.get("vqa_score", [])

            # Convert vqa_score entries bbox_xywh -> bbox_2d (xyxy) so draw_boxes() works
            for s in vqa_scores:
                if "bbox_2d" not in s and "bbox_xywh" in s:
                    s["bbox_2d"] = xywh_to_xyxy(s["bbox_xywh"])


        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"[WARN] Failed to open image {img_path}: {e}")
            continue

        vis = draw_boxes(
            img,
            gt_boxes=gt_boxes,
            vqa_scores=vqa_scores,
            thr=args.thr,
            coords_are_resized_1000=args.coords_are_resized_1000,
            resized_size=args.resized_size,
            show_gt=(not args.no_gt),
            show_text=True,
        )

        out_path = out_dir / f"sample_{sample_idx:06d}.jpg"

        vis.save(out_path, quality=95)
        previews.append(vis)
        print(f"Saved: {out_path}  (boxes={len(gt_boxes)})")

    if args.grid and previews:
        grid_path = out_dir / "preview_grid.png"
        save_preview_grid(previews[:min(len(previews), 16)], grid_path, cols=4)
        print(f"Saved grid preview: {grid_path}")


if __name__ == "__main__":
    main()
