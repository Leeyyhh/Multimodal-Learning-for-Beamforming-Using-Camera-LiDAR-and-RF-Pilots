import os
from ultralytics import YOLO
import torch
from PIL import Image
import numpy as np

# -------------------- utils --------------------
def iou_xyxy(a, b):
    tl = torch.max(a[:, None, :2], b[None, :, :2])
    br = torch.min(a[:, None, 2:4], b[None, :, 2:4])
    inter = (br - tl).clamp(min=0)
    inter = inter[:, :, 0] * inter[:, :, 1]

    area_a = (a[:, 2] - a[:, 0]).clamp(min=0) * (a[:, 3] - a[:, 1]).clamp(min=0)
    area_b = (b[:, 2] - b[:, 0]).clamp(min=0) * (b[:, 3] - b[:, 1]).clamp(min=0)

    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def nms_greedy(boxes, scores, iou_thr=0.5):
    if boxes.numel() == 0:
        return torch.empty(0, dtype=torch.long)

    idx = torch.argsort(scores, descending=True)
    keep = []

    while idx.numel() > 0:
        i = idx[0].item()
        keep.append(i)

        if idx.numel() == 1:
            break

        ious = iou_xyxy(boxes[i].unsqueeze(0), boxes[idx[1:]]).squeeze(0)
        idx = idx[1:][ious <= iou_thr]

    return torch.tensor(keep, dtype=torch.long)


# -------------------- run one grid --------------------
def run_grid_once(model, image_np, rows, cols, ovw_ratio, ovh_ratio, conf_th):
    H, W, _ = image_np.shape

    base_w = W // cols
    base_h = H // rows

    ovw = int(base_w * ovw_ratio)
    ovh = int(base_h * ovh_ratio)

    all_boxes = []
    all_kps = []
    all_cls = []
    all_conf = []

    slices = []
    for r in range(rows):
        for c in range(cols):
            x0 = max(0, c * base_w - (ovw if c > 0 else 0))
            y0 = max(0, r * base_h - (ovh if r > 0 else 0))
            x1 = min(W, (c + 1) * base_w + (ovw if c < cols - 1 else 0))
            y1 = min(H, (r + 1) * base_h + (ovh if r < rows - 1 else 0))

            if x1 > x0 and y1 > y0:
                slices.append((x0, y0, x1 - x0, y1 - y0))

    for x0, y0, w, h in slices:
        patch = Image.fromarray(image_np[y0:y0+h, x0:x0+w])

        result = model(
            patch,
            conf=conf_th,
            iou=0.6,
            imgsz=1568,
            verbose=False
        )[0]

        if result.boxes is None or result.boxes.data.numel() == 0:
            continue

        if result.keypoints is None:
            continue

        kps = result.keypoints.data.detach().cpu()
        boxes = result.boxes.data.detach().cpu()
        clsid = result.boxes.cls.detach().cpu().long()
        confs = result.boxes.conf.detach().cpu().float()

        boxes_glob = boxes.clone()
        boxes_glob[:, [0, 2]] += x0
        boxes_glob[:, [1, 3]] += y0

        kps_glob = kps.clone()
        kps_glob[:, :, 0] += x0
        kps_glob[:, :, 1] += y0

        all_boxes.append(boxes_glob)
        all_kps.append(kps_glob)
        all_cls.append(clsid)
        all_conf.append(confs)

    if len(all_boxes) == 0:
        return (
            torch.zeros((0, 6), dtype=torch.float32),
            torch.zeros((0, 8, 3), dtype=torch.float32),
            torch.zeros((0,), dtype=torch.long),
            torch.zeros((0,), dtype=torch.float32)
        )

    all_boxes = torch.cat(all_boxes, dim=0)
    all_kps = torch.cat(all_kps, dim=0)
    all_cls = torch.cat(all_cls, dim=0)
    all_conf = torch.cat(all_conf, dim=0)

    return all_boxes, all_kps, all_cls, all_conf


# -------------------- adaptive wrapper --------------------
def process_pose_adaptive(
    model,
    image_path,
    conf_threshold=0.5,
    nms_iou=0.55,
    use_slicing=True
):
    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image)

    if use_slicing:
        tries = [
            (1, 1, 0.0, 0.0),
            (1, 3, 0.50, 0.0),
        ]
    else:
        tries = [
            (1, 1, 0.0, 0.0),
        ]

    boxes_all = []
    kps_all = []
    cls_all = []
    conf_all = []

    for rows, cols, ovw, ovh in tries:
        b, k, c, s = run_grid_once(
            model,
            image_np,
            rows=rows,
            cols=cols,
            ovw_ratio=ovw,
            ovh_ratio=ovh,
            conf_th=conf_threshold
        )

        if b.numel() == 0:
            continue

        boxes_all.append(b)
        kps_all.append(k)
        cls_all.append(c)
        conf_all.append(s)

    if len(boxes_all) == 0:
        return {
            "boxes": torch.zeros((0, 6), dtype=torch.float32),
            "kps": torch.zeros((0, 8, 3), dtype=torch.float32),
            "cls": torch.zeros((0,), dtype=torch.long),
            "conf": torch.zeros((0,), dtype=torch.float32),
        }

    boxes = torch.cat(boxes_all, dim=0)
    kps = torch.cat(kps_all, dim=0)
    clsid = torch.cat(cls_all, dim=0)
    confs = torch.cat(conf_all, dim=0)

    keep = confs >= conf_threshold
    boxes = boxes[keep]
    kps = kps[keep]
    clsid = clsid[keep]
    confs = confs[keep]

    if boxes.numel() == 0:
        return {
            "boxes": torch.zeros((0, 6), dtype=torch.float32),
            "kps": torch.zeros((0, 8, 3), dtype=torch.float32),
            "cls": torch.zeros((0,), dtype=torch.long),
            "conf": torch.zeros((0,), dtype=torch.float32),
        }

    # --------------------------------------------------
    # IMPORTANT:
    # Use class-agnostic NMS.
    # This suppresses duplicated boxes even if their class labels differ.
    # --------------------------------------------------
    final_idx = nms_greedy(boxes[:, :4], confs, iou_thr=nms_iou)

    boxes_out = boxes[final_idx].clone()
    kps_out = kps[final_idx]
    cls_out = clsid[final_idx]
    conf_out = confs[final_idx]

    boxes_out[:, 4] = conf_out
    boxes_out[:, 5] = cls_out.to(boxes_out.dtype)

    return {
        "boxes": boxes_out,
        "kps": kps_out,
        "cls": cls_out,
        "conf": conf_out
    }


# -------------------- batch over samples & cameras --------------------
def main():
    model = YOLO(r"runs/pose/train10/weights/best.pt")

    init_value = 0
    end_value = 100

    img_directory = r"D:\Multi_user_beamforming_data\Review\generate_data_image_for_yolo_test\part_00000\images"
    save_dir = r"D:\Multi_user_beamforming_data\Review\generate_data_image_for_yolo_test\part_00000\det2"
    os.makedirs(save_dir, exist_ok=True)

    cam_ids = [4]

    for sample_idx in range(init_value, end_value):
        try:
            results_per_cam = [None] * len(cam_ids)

            for cam_i, cam_id in enumerate(cam_ids):
                img_path = os.path.join(
                    img_directory,
                    f"cam{cam_id}_sample_{sample_idx}.png"
                )

                if not os.path.exists(img_path):
                    raise FileNotFoundError(f"Missing image: {img_path}")

                cam_res = process_pose_adaptive(
                    model,
                    img_path,
                    conf_threshold=0.4,
                    nms_iou=0.55,
                    use_slicing=True   # 改成 False 可以只用整图测试
                )

                results_per_cam[cam_i] = cam_res

            torch.save(
                {
                    "detections": results_per_cam,
                    "cam_ids": cam_ids
                },
                os.path.join(save_dir, f"detection_{sample_idx}.pth")
            )

        except Exception as e:
            print(f"Sample {sample_idx}: Skipped due to error: {e}")


if __name__ == "__main__":
    main()
