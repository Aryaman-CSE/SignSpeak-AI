import os
import glob
import time
import threading
from collections import deque, defaultdict

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "1"

import cv2
import numpy as np
import tensorflow as tf

tf.config.threading.set_intra_op_parallelism_threads(0)
tf.config.threading.set_inter_op_parallelism_threads(0)

from object_detection.utils import label_map_util
from object_detection.builders import model_builder
from object_detection.utils import config_util

BASE = "SignSpeak-AI-main" if os.path.exists("SignSpeak-AI-main/Tensorflow") else "."
CONFIG_PATH = os.path.join(BASE, "Tensorflow", "workspace", "models", "pipeline.config")
MODEL_DIR = os.path.join(BASE, "Tensorflow", "workspace", "models", "my_ssd_mobnet")
LABEL_MAP_PATH = os.path.join(BASE, "Tensorflow", "workspace", "annotations", "label_map.pbtxt")

ckpt_files = glob.glob(os.path.join(MODEL_DIR, "ckpt-*.index"))
if not ckpt_files:
    raise FileNotFoundError(f"No model checkpoints found in {MODEL_DIR}")

latest_ckpt = os.path.splitext(
    max(ckpt_files, key=lambda x: int(os.path.basename(x).split("-")[1].split(".")[0]))
)[0]
print(f"[INFO] Loading Checkpoint: {latest_ckpt}")

configs = config_util.get_configs_from_pipeline_file(CONFIG_PATH)
detection_model = model_builder.build(model_config=configs["model"], is_training=False)
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(latest_ckpt).expect_partial()

@tf.function
def detect_batch_fn(images):
    images, shapes = detection_model.preprocess(images)
    prediction_dict = detection_model.predict(images, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    return detections

category_index = label_map_util.create_category_index_from_labelmap(
    LABEL_MAP_PATH, use_display_name=True
)

_ = detect_batch_fn(tf.zeros([4, 320, 320, 3], dtype=tf.float32))

face_cascade = cv2.CascadeClassifier(
    os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
)
profile_cascade = cv2.CascadeClassifier(
    os.path.join(cv2.data.haarcascades, "haarcascade_profileface.xml")
)

MIN_CONF = 0.38
WINDOW_SIZE = 7
LOCK_FRAMES_REQUIRED = 4
EMPTY_FRAMES_TO_UNLOCK = 4
BOX_EMA = 0.70

prediction_buffer = deque(maxlen=WINDOW_SIZE)

is_locked = False
locked_letter = None
locked_conf = 0.0
empty_counter = 0

candidate_letter = None
candidate_conf = 0.0
lock_progress = 0.0
smoothed_box = None
last_known_face = None
face_hold_frames = 0
top3_debug = "Place Left or Right hand in side box..."
infer_fps = 0.0
latest_bgr = None
running = True

def reset_lock():
    global is_locked, locked_letter, locked_conf, empty_counter, lock_progress, candidate_letter
    is_locked = False
    locked_letter = None
    locked_conf = 0.0
    candidate_letter = None
    empty_counter = 0
    lock_progress = 0.0
    prediction_buffer.clear()

def get_hand_rois(w, h):
    left_roi = (10, 105, int(w * 0.42), h - 15)
    right_roi = (int(w * 0.58), 105, w - 10, h - 15)
    return left_roi, right_roi

def get_persistent_face(frame_bgr):
    global last_known_face, face_hold_frames
    h, w, _ = frame_bgr.shape
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(50, 50))
    if len(faces) == 0:
        faces = profile_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(50, 50))

    if len(faces) > 0:
        largest = max(faces, key=lambda r: r[2] * r[3])
        fx, fy, fw, fh = largest
        pad_x = int(fw * 0.35)
        pad_top = int(fh * 0.35)
        pad_bot = int(fh * 0.55)
        x1 = max(0, fx - pad_x)
        y1 = max(0, fy - pad_top)
        x2 = min(w, fx + fw + pad_x)
        y2 = min(h, fy + fh + pad_bot)
        last_known_face = (x1, y1, x2, y2)
        face_hold_frames = 45
    elif face_hold_frames > 0:
        face_hold_frames -= 1
    else:
        cx1, cy1, cx2, cy2 = int(w * 0.32), int(h * 0.12), int(w * 0.68), int(h * 0.72)
        last_known_face = (cx1, cy1, cx2, cy2)

    return last_known_face

def box_overlaps_face_zone(box_px, face_zone):
    if face_zone is None:
        return False
    bx1, by1, bx2, by2 = box_px
    fx1, fy1, fx2, fy2 = face_zone
    bcx, bcy = (bx1 + bx2) / 2.0, (by1 + by2) / 2.0
    if fx1 <= bcx <= fx2 and fy1 <= bcy <= fy2:
        return True
    ix1, iy1 = max(bx1, fx1), max(by1, fy1)
    ix2, iy2 = min(bx2, fx2), min(by2, fy2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    box_area = max(1.0, float((bx2 - bx1) * (by2 - by1)))
    return (inter / box_area) > 0.15

def inference_worker():
    global latest_bgr, smoothed_box, top3_debug, infer_fps
    global is_locked, locked_letter, locked_conf, empty_counter
    global candidate_letter, candidate_conf, lock_progress

    while running:
        if latest_bgr is None:
            time.sleep(0.005)
            continue

        t0 = time.perf_counter()
        frame_copy = latest_bgr.copy()
        h, w, _ = frame_copy.shape

        face_zone = get_persistent_face(frame_copy)
        clean_bgr = frame_copy.copy()
        if face_zone is not None:
            fx1, fy1, fx2, fy2 = face_zone
            clean_bgr[fy1:fy2, fx1:fx2] = (128, 128, 128)

        left_roi, right_roi = get_hand_rois(w, h)
        rois = [left_roi, right_roi]

        crops_rgb = []
        for (rx1, ry1, rx2, ry2) in rois:
            crop = clean_bgr[ry1:ry2, rx1:rx2]
            crop_320 = cv2.resize(crop, (320, 320))
            crop_rgb = cv2.cvtColor(crop_320, cv2.COLOR_BGR2RGB)
            crops_rgb.append(crop_rgb)
            crops_rgb.append(cv2.flip(crop_rgb, 1))

        batch_np = np.stack(crops_rgb, axis=0)
        input_tensor = tf.convert_to_tensor(batch_np, dtype=tf.float32)
        dets = detect_batch_fn(input_tensor)

        scores_batch = dets["detection_scores"].numpy()
        boxes_batch = dets["detection_boxes"].numpy()
        classes_batch = dets["detection_classes"].numpy().astype(np.int64) + 1

        candidates = []
        for b_idx in range(4):
            roi_idx = b_idx // 2
            is_flipped = (b_idx % 2 == 1)
            rx1, ry1, rx2, ry2 = rois[roi_idx]
            rw, rh = float(rx2 - rx1), float(ry2 - ry1)

            for i in range(min(4, len(scores_batch[b_idx]))):
                sc = float(scores_batch[b_idx][i])
                cls_id = int(classes_batch[b_idx][i])
                nm = category_index.get(cls_id, {"name": "?"})["name"]
                ymin, xmin, ymax, xmax = boxes_batch[b_idx][i]

                if is_flipped:
                    xmin, xmax = 1.0 - xmax, 1.0 - xmin

                crop_area = (ymax - ymin) * (xmax - xmin)
                if crop_area < 0.06 or crop_area > 0.92:
                    continue

                gx1 = rx1 + xmin * rw
                gy1 = ry1 + ymin * rh
                gx2 = rx1 + xmax * rw
                gy2 = ry1 + ymax * rh

                if box_overlaps_face_zone((gx1, gy1, gx2, gy2), face_zone):
                    continue

                norm_global_box = np.array([gy1 / h, gx1 / w, gy2 / h, gx2 / w], dtype=np.float32)
                candidates.append((sc, nm, norm_global_box))

        candidates.sort(key=lambda x: x[0], reverse=True)

        if candidates:
            top3_debug = " | ".join([f"{c[1]}:{int(c[0]*100)}%" for c in candidates[:3]])
        else:
            top3_debug = "Put Left or Right hand inside a side box..."

        if candidates and candidates[0][0] >= MIN_CONF:
            empty_counter = 0
            best_sc, best_nm, best_box = candidates[0]
            smoothed_box = (
                best_box
                if smoothed_box is None
                else (BOX_EMA * best_box + (1.0 - BOX_EMA) * smoothed_box)
            )

            if not is_locked:
                prediction_buffer.append((best_nm, best_sc))
        else:
            empty_counter += 1
            if empty_counter >= 2:
                smoothed_box = None
            if not is_locked:
                prediction_buffer.append((None, 0.0))
            elif empty_counter >= EMPTY_FRAMES_TO_UNLOCK:
                reset_lock()

        if not is_locked:
            weights = defaultdict(float)
            counts = defaultdict(int)
            for lbl, sc in prediction_buffer:
                if lbl is not None:
                    weights[lbl] += sc
                    counts[lbl] += 1

            if weights:
                best_lbl = max(weights.items(), key=lambda x: x[1])[0]
                cnt = counts[best_lbl]
                avg_sc = weights[best_lbl] / float(cnt)
                candidate_letter = best_lbl
                candidate_conf = avg_sc
                lock_progress = min(1.0, cnt / float(LOCK_FRAMES_REQUIRED))

                if cnt >= LOCK_FRAMES_REQUIRED:
                    is_locked = True
                    locked_letter = best_lbl
                    locked_conf = avg_sc
                    lock_progress = 1.0
            else:
                candidate_letter = None
                candidate_conf = 0.0
                lock_progress = 0.0

        infer_fps = 1.0 / max(time.perf_counter() - t0, 1e-4)

threading.Thread(target=inference_worker, daemon=True).start()

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("\n[LIVE] SignSpeak AI - Left & Right Hand Zone Detector (Face Excluded)")
print("Controls: [SPACE / R] = Reset Lock | [Q] = Quit\n")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    latest_bgr = frame
    h, w, _ = frame.shape

    left_roi, right_roi = get_hand_rois(w, h)
    for idx_r, (rx1, ry1, rx2, ry2) in enumerate([left_roi, right_roi]):
        cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), (90, 90, 90), 1)
        zone_lbl = "LEFT HAND ZONE" if idx_r == 0 else "RIGHT HAND ZONE"
        cv2.putText(frame, zone_lbl, (rx1 + 8, ry2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (130, 130, 130), 1)

    if smoothed_box is not None:
        ymin, xmin, ymax, xmax = smoothed_box
        x1, y1, x2, y2 = int(xmin * w), int(ymin * h), int(xmax * w), int(ymax * h)

        green_col = (0, 255, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), green_col, 3)

        if is_locked:
            box_tag = f"{locked_letter} ({int(locked_conf * 100)}%) [LOCKED]"
        else:
            lbl_str = candidate_letter if candidate_letter else "..."
            box_tag = f"{lbl_str} ({int(candidate_conf * 100)}%)"

        (tw, th), _ = cv2.getTextSize(box_tag, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)
        tag_y1 = max(95, y1 - th - 14)
        tag_y2 = max(95 + th + 14, y1)
        cv2.rectangle(frame, (x1, tag_y1), (x1 + tw + 14, tag_y2), green_col, -1)
        cv2.putText(
            frame, box_tag, (x1 + 7, tag_y2 - 7),
            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2
        )

    cv2.rectangle(frame, (0, 0), (w, 90), (18, 18, 18), -1)

    if is_locked:
        hud_text = f"DETECTED LETTER: {locked_letter} ({int(locked_conf * 100)}%)"
        cv2.putText(frame, hud_text, (15, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (0, 255, 0), 3)
        cv2.putText(
            frame,
            f"LOCKED! Drop hand or press [SPACE] for next sign  |  {top3_debug}",
            (15, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (160, 255, 160), 1
        )
    else:
        scan_txt = f"Detecting: {candidate_letter}" if candidate_letter else "Show Hand in Left/Right Box"
        cv2.putText(frame, scan_txt, (15, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.78, (0, 255, 0), 2)

        bx, by, bw, bh = 385, 14, 120, 22
        cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (55, 55, 55), -1)
        if lock_progress > 0:
            cv2.rectangle(frame, (bx, by), (bx + int(bw * lock_progress), by + bh), (0, 255, 0), -1)
        cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (180, 180, 180), 1)

        cv2.putText(
            frame,
            f"Top-3: {top3_debug}  |  [SPACE] Reset  [Q] Quit",
            (15, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (200, 200, 200), 1
        )

    cv2.putText(
        frame, f"FPS: {infer_fps:.1f}", (w - 115, 36),
        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (180, 180, 180), 2
    )

    cv2.imshow("SignSpeak AI - Real-Time ASL Sign Detector", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        running = False
        break
    elif key == ord(" ") or key == ord("r"):
        reset_lock()

cap.release()
cv2.destroyAllWindows()