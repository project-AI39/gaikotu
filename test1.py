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


def draw_pose(image, keypoints, confidence_threshold=0.5):
    """キーポイントから骨格を描画"""
    if keypoints is None:
        return image

    # キーポイントの座標を取得
    points = keypoints.xy[0].cpu().numpy()  # (17, 2)
    confs = keypoints.conf[0].cpu().numpy()  # (17,)

    # 各キーポイントを描画
    for i, (point, conf) in enumerate(zip(points, confs)):
        if conf > confidence_threshold:
            x, y = int(point[0]), int(point[1])
            cv2.circle(image, (x, y), 5, (0, 255, 0), -1)

    # 骨格の線を描画
    for pair in POSE_PAIRS:
        idx1, idx2 = pair
        if confs[idx1] > confidence_threshold and confs[idx2] > confidence_threshold:
            pt1 = tuple(points[idx1].astype(int))
            pt2 = tuple(points[idx2].astype(int))
            cv2.line(image, pt1, pt2, (255, 0, 0), 2)

    return image


def main():
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
                frame = draw_pose(frame, result.keypoints)

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
