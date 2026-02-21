from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import time
from typing import List, Tuple

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from tracker import FaceTracker


EMOTION_DISPLAY = {
    "angry": "Tension",
    "disgust": "Ill",
    "fear": "Afraid",
    "happy": "Happy",
    "sad": "Sad",
    "surprise": "Surprised",
    "neutral": "Neutral",
}

FACE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
EYE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye_tree_eyeglasses.xml")
SMILE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_smile.xml")


@dataclass
class DetectionView:
    session_id: int
    box: Tuple[int, int, int, int]
    top_emotion_key: str
    confidence: float


def init_state() -> None:
    defaults = {
        "running": False,
        "cap": None,
        "tracker": None,
        "last_detections": [],
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def open_camera(camera_index: int) -> cv2.VideoCapture | None:
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        cap.release()
        return None

    ok, _ = cap.read()
    if not ok:
        cap.release()
        return None
    return cap


def close_camera() -> None:
    cap = st.session_state.get("cap")
    if cap is not None:
        try:
            cap.release()
        except Exception:
            pass
    st.session_state.cap = None


def ensure_runtime(camera_index: int) -> None:
    if st.session_state.tracker is None:
        st.session_state.tracker = FaceTracker(max_distance=130.0, max_age_seconds=1.2)
    if st.session_state.cap is None:
        st.session_state.cap = open_camera(camera_index)


def estimate_emotion_for_face(face_gray: np.ndarray) -> Tuple[str, float]:
    """
    Lightweight rule-based emotion estimator using OpenCV Haar features.
    This is heuristic and intended for demo/prototyping without heavy ML dependencies.
    """
    fh, fw = face_gray.shape[:2]

    eyes = EYE_CASCADE.detectMultiScale(face_gray, scaleFactor=1.1, minNeighbors=5, minSize=(12, 12))
    smiles = SMILE_CASCADE.detectMultiScale(face_gray, scaleFactor=1.7, minNeighbors=20, minSize=(25, 25))

    eye_count = len(eyes)
    brightness = float(np.mean(face_gray) / 255.0)

    smile_score = 0.0
    mouth_open_ratio = 0.0
    if len(smiles) > 0:
        sx, sy, sw, sh = max(smiles, key=lambda r: r[2] * r[3])
        smile_score = min(1.0, float(sw / max(fw, 1)))
        mouth_open_ratio = min(1.0, float(sh / max(fh, 1)))

    if smile_score >= 0.33:
        return "happy", min(0.95, 0.55 + smile_score * 0.5)

    if eye_count >= 2 and mouth_open_ratio >= 0.17:
        return "surprise", min(0.9, 0.5 + mouth_open_ratio)

    if eye_count >= 2 and brightness > 0.6 and mouth_open_ratio >= 0.1:
        return "fear", 0.62

    if eye_count <= 1 and brightness < 0.4 and smile_score < 0.2:
        return "sad", 0.56

    if eye_count >= 2 and brightness < 0.42 and smile_score < 0.2:
        return "angry", 0.54

    if eye_count <= 1 and brightness < 0.35:
        return "disgust", 0.52

    return "neutral", 0.5


def detect_faces_and_emotions(
    frame_bgr: np.ndarray,
    tracker: FaceTracker,
    min_confidence: float,
) -> Tuple[np.ndarray, List[DetectionView]]:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    faces = FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

    filtered_boxes: List[Tuple[int, int, int, int]] = []
    filtered_raw = []

    for (x, y, w, h) in faces:
        roi_gray = gray[y : y + h, x : x + w]
        emotion_key, conf = estimate_emotion_for_face(roi_gray)
        if conf < min_confidence:
            continue

        filtered_boxes.append((int(x), int(y), int(w), int(h)))
        filtered_raw.append((emotion_key, float(conf)))

    session_ids = tracker.update(filtered_boxes)
    detections: List[DetectionView] = []

    for (x, y, w, h), sid, (emotion_key, conf) in zip(filtered_boxes, session_ids, filtered_raw):
        detections.append(
            DetectionView(
                session_id=sid,
                box=(x, y, w, h),
                top_emotion_key=emotion_key,
                confidence=conf,
            )
        )

    annotated = frame_bgr.copy()
    for d in detections:
        x, y, w, h = d.box
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (60, 230, 255), 2)

        emotion_text = EMOTION_DISPLAY.get(d.top_emotion_key, d.top_emotion_key.title())
        tag = f"Face-{d.session_id} | {emotion_text} {d.confidence:.2f}"

        cv2.rectangle(annotated, (x, max(y - 28, 0)), (x + min(280, w + 220), max(y - 2, 0)), (20, 20, 20), -1)
        cv2.putText(
            annotated,
            tag,
            (x + 4, max(y - 8, 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

    return annotated, detections


def emotion_table(detections: List[DetectionView]) -> pd.DataFrame:
    if not detections:
        return pd.DataFrame(columns=["Face", "Emotion", "Confidence"])

    rows = []
    for d in detections:
        rows.append(
            {
                "Face": f"Face-{d.session_id}",
                "Emotion": EMOTION_DISPLAY.get(d.top_emotion_key, d.top_emotion_key.title()),
                "Confidence": round(d.confidence, 3),
            }
        )
    return pd.DataFrame(rows)


def emotion_distribution(detections: List[DetectionView]) -> pd.DataFrame:
    counts = Counter(EMOTION_DISPLAY.get(d.top_emotion_key, d.top_emotion_key.title()) for d in detections)
    if not counts:
        return pd.DataFrame(columns=["Emotion", "Count"])
    return pd.DataFrame({"Emotion": list(counts.keys()), "Count": list(counts.values())})


def sidebar_controls() -> Tuple[int, float, int]:
    st.sidebar.header("Settings")
    camera_index = st.sidebar.number_input("Camera Index", min_value=0, max_value=8, value=0, step=1)
    min_confidence = st.sidebar.slider("Min Emotion Confidence", 0.1, 0.95, 0.45, 0.05)
    fps_limit = st.sidebar.slider("Max FPS", 2, 30, 10)

    start_clicked = st.sidebar.button("Start", use_container_width=True)
    stop_clicked = st.sidebar.button("Stop", use_container_width=True)

    if start_clicked:
        st.session_state.running = True
    if stop_clicked:
        st.session_state.running = False

    return int(camera_index), float(min_confidence), int(fps_limit)


def run_once(camera_index: int, min_confidence: float, fps_limit: int) -> None:
    ensure_runtime(camera_index)

    if st.session_state.cap is None:
        st.error("Camera could not be opened. Close other camera apps and retry.")
        st.session_state.running = False
        return

    ok, frame = st.session_state.cap.read()
    if not ok:
        st.error("Could not read camera frame.")
        st.session_state.running = False
        close_camera()
        return

    annotated, detections = detect_faces_and_emotions(
        frame_bgr=frame,
        tracker=st.session_state.tracker,
        min_confidence=min_confidence,
    )
    st.session_state.last_detections = detections

    st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)

    left, right = st.columns(2)
    with left:
        st.subheader("Detected Faces")
        st.dataframe(emotion_table(detections), use_container_width=True, hide_index=True)
    with right:
        st.subheader("Emotion Summary")
        summary_df = emotion_distribution(detections)
        if not summary_df.empty:
            st.bar_chart(summary_df.set_index("Emotion"))
        else:
            st.info("No faces with confident emotion detected in this frame.")

    time.sleep(max(0.01, 1.0 / fps_limit))
    st.rerun()


def main() -> None:
    st.set_page_config(page_title="Face Detection + Emotion UI", layout="wide")
    st.title("Face Detection and Emotion Mapping")
    st.caption(
        "Live camera stream with session-based face IDs and emotion labels: "
        "Happy, Sad, Tension, Ill, Afraid, Surprised, Neutral."
    )

    st.info(
        "This app detects faces and assigns temporary session IDs (Face-1, Face-2...). "
        "It does not identify real-world identity/name. Emotion detection is rule-based."
    )

    init_state()
    camera_index, min_confidence, fps_limit = sidebar_controls()

    if not st.session_state.running:
        st.success("Click Start in the sidebar to begin live detection.")
        close_camera()
        st.session_state.tracker = None
        return

    run_once(camera_index, min_confidence, fps_limit)


if __name__ == "__main__":
    main()
