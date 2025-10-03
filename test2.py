import math
import os

import cv2
import numpy as np
from ultralytics import YOLO

# YOLOポーズモデルをロード
model = YOLO("yolo11m-pose.pt")

# COCOキーポイントの接続定義 (骨格を描画するための線)
POSE_PAIRS = [
    [0, 1],
    [0, 2],
    [1, 3],
    [2, 4],  # 頭部: 鼻-目-耳
    [5, 6],
    [5, 7],
    [7, 9],
    [6, 8],
    [8, 10],  # 腕: 肩-肘-手首
    [5, 11],
    [6, 12],
    [11, 12],  # 胴体: 肩-腰
    [11, 13],
    [13, 15],
    [12, 14],
    [14, 16],  # 脚: 腰-膝-足首
]

ASSET_NAMES = [
    "左上腕",
    "左下腕",
    "右上腕",
    "右下腕",
    "左上足",
    "左下足",
    "右上足",
    "右下足",
    "胴体",
    "顔",
]

LIMB_CONFIGS = {
    "左上腕": (5, 7),
    "左下腕": (7, 9),
    "右上腕": (6, 8),
    "右下腕": (8, 10),
    "左上足": (11, 13),
    "左下足": (13, 15),
    "右上足": (12, 14),
    "右下足": (14, 16),
}


def rotate_bound(image, angle):
    """画像を切れないように回転"""
    h, w = image.shape[:2]
    center = (w / 2, h / 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos = abs(matrix[0, 0])
    sin = abs(matrix[0, 1])

    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))

    matrix[0, 2] += (new_w / 2) - center[0]
    matrix[1, 2] += (new_h / 2) - center[1]

    return cv2.warpAffine(
        image,
        matrix,
        (new_w, new_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )


def overlay_image_alpha(base, overlay, top_left):
    """アルファ付き画像を合成"""
    x, y = top_left
    h, w = overlay.shape[:2]
    if overlay.shape[2] < 4:
        return

    if x >= base.shape[1] or y >= base.shape[0] or x + w <= 0 or y + h <= 0:
        return

    x1 = max(x, 0)
    y1 = max(y, 0)
    x2 = min(x + w, base.shape[1])
    y2 = min(y + h, base.shape[0])

    overlay_x1 = max(0, -x)
    overlay_y1 = max(0, -y)
    overlay_x2 = overlay_x1 + (x2 - x1)
    overlay_y2 = overlay_y1 + (y2 - y1)

    overlay_crop = overlay[overlay_y1:overlay_y2, overlay_x1:overlay_x2]
    if overlay_crop.size == 0:
        return

    alpha = overlay_crop[:, :, 3:4].astype(np.float32) / 255.0
    rgb = overlay_crop[:, :, :3].astype(np.float32)

    base_region = base[y1:y2, x1:x2].astype(np.float32)
    base[y1:y2, x1:x2] = ((1.0 - alpha) * base_region + alpha * rgb).astype(np.uint8)


def overlay_centered(base, overlay, center):
    h, w = overlay.shape[:2]
    x = int(center[0] - w / 2)
    y = int(center[1] - h / 2)
    overlay_image_alpha(base, overlay, (x, y))


def overlay_limb(base, overlay, pt1, pt2):
    p1 = np.array(pt1, dtype=np.float32)
    p2 = np.array(pt2, dtype=np.float32)
    length = np.linalg.norm(p2 - p1)
    if length < 5:
        return

    scale = max(length / overlay.shape[0], 0.1)
    new_w = max(int(overlay.shape[1] * scale), 1)
    new_h = max(int(overlay.shape[0] * scale), 1)
    resized = cv2.resize(overlay, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    angle = math.degrees(math.atan2(p2[1] - p1[1], p2[0] - p1[0])) - 90
    rotated = rotate_bound(resized, angle)

    center = (p1 + p2) / 2.0
    overlay_centered(base, rotated, center)


def overlay_torso(base, overlay, points, confs, threshold):
    idxs = [5, 6, 11, 12]
    if not all(confs[i] > threshold for i in idxs):
        return

    left_shoulder, right_shoulder, left_hip, right_hip = [
        np.array(points[i], dtype=np.float32) for i in idxs
    ]
    shoulder_width = np.linalg.norm(left_shoulder - right_shoulder)
    hip_width = np.linalg.norm(left_hip - right_hip)
    torso_height = (
        np.linalg.norm(left_shoulder - left_hip)
        + np.linalg.norm(right_shoulder - right_hip)
    ) / 2

    width = max(shoulder_width * 1.2, hip_width * 1.1, 1.0)
    height = max(torso_height * 1.4, 1.0)

    scale_w = width / overlay.shape[1]
    scale_h = height / overlay.shape[0]
    scale = min(max(scale_w, 0.1), max(scale_h, 0.1))
    resized = cv2.resize(
        overlay,
        (max(int(overlay.shape[1] * scale), 1), max(int(overlay.shape[0] * scale), 1)),
        interpolation=cv2.INTER_LINEAR,
    )

    angle = math.degrees(
        math.atan2(
            right_shoulder[1] - left_shoulder[1], right_shoulder[0] - left_shoulder[0]
        )
    )
    rotated = rotate_bound(resized, angle)

    center = (left_shoulder + right_shoulder + left_hip + right_hip) / 4.0
    overlay_centered(base, rotated, center)


def overlay_face(base, overlay, points, confs, threshold):
    if confs[0] <= threshold:
        return

    center = np.array(points[0], dtype=np.float32)

    if confs[5] > threshold and confs[6] > threshold:
        width = np.linalg.norm(points[5] - np.array(points[6])) * 0.8
        angle = math.degrees(
            math.atan2(points[6][1] - points[5][1], points[6][0] - points[5][0])
        )
    elif confs[1] > threshold and confs[2] > threshold:
        width = np.linalg.norm(points[1] - np.array(points[2])) * 2.0
        angle = math.degrees(
            math.atan2(points[2][1] - points[1][1], points[2][0] - points[1][0])
        )
    else:
        width = 60.0
        angle = 0.0

    width = max(width, 20.0)
    scale = width / overlay.shape[1]
    resized = cv2.resize(
        overlay,
        (max(int(overlay.shape[1] * scale), 1), max(int(overlay.shape[0] * scale), 1)),
        interpolation=cv2.INTER_LINEAR,
    )

    rotated = rotate_bound(resized, angle)
    overlay_centered(base, rotated, center)


def load_bone_images():
    asset_dir = os.path.join(os.path.dirname(__file__), "data")
    assets = {}
    for name in ASSET_NAMES:
        path = os.path.join(asset_dir, f"{name}.png")
        image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if image is None:
            print(f"警告: {path} を読み込めませんでした。")
            continue
        if image.shape[2] == 3:
            alpha = np.full((image.shape[0], image.shape[1], 1), 255, dtype=image.dtype)
            image = np.concatenate([image, alpha], axis=2)
        assets[name] = image
    return assets


def draw_person(image, points, confs, assets, threshold):
    points = np.array(points, dtype=np.float32)
    confs = np.array(confs, dtype=np.float32)

    for i, (point, conf) in enumerate(zip(points, confs)):
        if conf > threshold:
            cv2.circle(image, tuple(point.astype(int)), 4, (0, 255, 0), -1)

    for pair in POSE_PAIRS:
        idx1, idx2 = pair
        if confs[idx1] > threshold and confs[idx2] > threshold:
            pt1 = tuple(points[idx1].astype(int))
            pt2 = tuple(points[idx2].astype(int))
            cv2.line(image, pt1, pt2, (255, 255, 255), 1)

    for name, (idx1, idx2) in LIMB_CONFIGS.items():
        overlay = assets.get(name)
        if overlay is None:
            continue
        if confs[idx1] > threshold and confs[idx2] > threshold:
            overlay_limb(image, overlay, points[idx1], points[idx2])

    torso = assets.get("胴体")
    if torso is not None:
        overlay_torso(image, torso, points, confs, threshold)

    face = assets.get("顔")
    if face is not None:
        overlay_face(image, face, points, confs, threshold)

    return image


def draw_pose(image, keypoints, assets, confidence_threshold=0.5):
    """キーポイントから骨格を描画"""
    if keypoints is None:
        return image

    points = keypoints.xy.cpu().numpy()
    confs = keypoints.conf.cpu().numpy()

    for idx in range(points.shape[0]):
        image = draw_person(
            image, points[idx], confs[idx], assets, confidence_threshold
        )

    return image


def main():
    assets = load_bone_images()

    # Webカメラを開く
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Webカメラを開けません")
        return

    print("Webカメラを開始しました。'q'キーで終了します。")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # YOLOでポーズ推定
        results = model(frame, conf=0.5)

        # 結果を描画
        for result in results:
            if result.keypoints is not None:
                frame = draw_pose(frame, result.keypoints, assets)

        # ウィンドウに表示
        cv2.imshow("Pose Detection", frame)

        # 'q'キーで終了
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # クリーンアップ
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
