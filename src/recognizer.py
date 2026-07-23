"""Facial landmark detection and recognition, built directly on dlib
(HOG face detector + 68-point shape predictor + ResNet face encoder).

This intentionally avoids the `face_recognition` package: it hard-declares
a dependency on the real `dlib` PyPI package, which ships no prebuilt
wheels and must compile from source (~45 minutes on Streamlit Cloud). We
use `dlib-bin` instead, which has prebuilt wheels for Linux/macOS/Windows
and installs in seconds -- but `pip`/`uv` won't recognize it as satisfying
`face_recognition`'s declared "dlib>=19.7" requirement, so we call dlib's
API ourselves using the same models `face_recognition` bundles via
`face_recognition_models`.

The (comparatively expensive) landmark + encoding pipeline runs on a
background thread via RecognitionWorker so a live video callback can keep
processing frames without blocking.
"""
from __future__ import annotations

import queue
import threading
from dataclasses import dataclass, field
from typing import Optional

import dlib
import face_recognition_models
import numpy as np

DEFAULT_MATCH_TOLERANCE = 0.6

_face_detector = dlib.get_frontal_face_detector()
_pose_predictor = dlib.shape_predictor(face_recognition_models.pose_predictor_model_location())
_face_encoder = dlib.face_recognition_model_v1(face_recognition_models.face_recognition_model_location())


@dataclass
class RecognitionResult:
    name: Optional[str]
    distance: float
    box: tuple[int, int, int, int]  # (top, right, bottom, left)
    landmarks: dict = field(default_factory=dict)


def _rect_to_box(rect: "dlib.rectangle") -> tuple[int, int, int, int]:
    return rect.top(), rect.right(), rect.bottom(), rect.left()


def face_locations(image_rgb: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Detect faces, returning (top, right, bottom, left) boxes."""
    return [_rect_to_box(rect) for rect in _face_detector(image_rgb, 1)]


def encode_face(image_rgb: np.ndarray):
    """Return the first face's (encoding, location) found in an RGB image,
    or (None, None) if no face is found.
    """
    detections = _face_detector(image_rgb, 1)
    if not detections:
        return None, None

    shape = _pose_predictor(image_rgb, detections[0])
    encoding = np.array(_face_encoder.compute_face_descriptor(image_rgb, shape))
    return encoding, _rect_to_box(detections[0])


def get_landmarks(image_rgb: np.ndarray) -> list[list[tuple[int, int]]]:
    """Return raw 68-point landmark coordinates for each detected face."""
    landmarks = []
    for rect in _face_detector(image_rgb, 1):
        shape = _pose_predictor(image_rgb, rect)
        landmarks.append([(p.x, p.y) for p in shape.parts()])
    return landmarks


def match_encoding(
    encoding: np.ndarray,
    known_encodings: dict[str, np.ndarray],
    tolerance: float = DEFAULT_MATCH_TOLERANCE,
) -> tuple[Optional[str], float]:
    """Compare an encoding against all known faces, returning the closest
    match if it's within `tolerance`, else (None, best_distance_seen).
    """
    best_name, best_distance = None, float("inf")
    for name, known_encoding in known_encodings.items():
        distance = float(np.linalg.norm(known_encoding - encoding))
        if distance < best_distance:
            best_distance = distance
            best_name = name
    if best_name is not None and best_distance <= tolerance:
        return best_name, best_distance
    return None, best_distance


class RecognitionWorker:
    """Runs face landmark/encoding/matching on a background thread so a
    real-time video loop can keep boxing faces every frame without stalling
    on the heavier recognition work.
    """

    def __init__(self, known_encodings: dict[str, np.ndarray], tolerance: float = DEFAULT_MATCH_TOLERANCE):
        self.known_encodings = known_encodings
        self.tolerance = tolerance
        self._in_queue: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=1)
        self._latest_result: Optional[RecognitionResult] = None
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def submit(self, frame_rgb: np.ndarray) -> None:
        """Non-blocking: drop the frame if the worker is still busy on a previous one."""
        try:
            self._in_queue.put_nowait(frame_rgb)
        except queue.Full:
            pass

    def latest_result(self) -> Optional[RecognitionResult]:
        with self._lock:
            return self._latest_result

    def stop(self) -> None:
        self._stop_event.set()

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                frame = self._in_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            encoding, location = encode_face(frame)
            if encoding is None:
                result = None
            else:
                name, distance = match_encoding(encoding, self.known_encodings, self.tolerance)
                result = RecognitionResult(name=name, distance=distance, box=location)

            with self._lock:
                self._latest_result = result
