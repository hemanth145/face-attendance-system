"""Facial landmark detection and recognition, built on dlib via the
face_recognition package. The (comparatively expensive) landmark + encoding
pipeline runs on a background thread via RecognitionWorker so a live video
callback can keep processing frames without blocking.
"""
from __future__ import annotations

import queue
import threading
from dataclasses import dataclass, field
from typing import Optional

import face_recognition
import numpy as np

DEFAULT_MATCH_TOLERANCE = 0.6


@dataclass
class RecognitionResult:
    name: Optional[str]
    distance: float
    box: tuple[int, int, int, int]  # (top, right, bottom, left)
    landmarks: dict = field(default_factory=dict)


def encode_face(image_rgb: np.ndarray):
    """Return the first face's (encoding, location) found in an RGB image,
    or (None, None) if no face is found.
    """
    locations = face_recognition.face_locations(image_rgb)
    if not locations:
        return None, None
    encodings = face_recognition.face_encodings(image_rgb, known_face_locations=locations)
    if not encodings:
        return None, None
    return encodings[0], locations[0]


def get_landmarks(image_rgb: np.ndarray) -> list[dict]:
    return face_recognition.face_landmarks(image_rgb)


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
