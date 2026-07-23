"""Face detection using an Adaboost/Haar cascade classifier, with histogram
equalization as a preprocessing step to improve accuracy under uneven lighting.
"""
import cv2
import numpy as np

_FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


def equalize_histogram(gray_frame: np.ndarray) -> np.ndarray:
    """Normalize contrast so detection is more robust across lighting conditions."""
    return cv2.equalizeHist(gray_frame)


def detect_faces(
    frame_bgr: np.ndarray,
    scale_factor: float = 1.1,
    min_neighbors: int = 5,
    min_size: tuple[int, int] = (40, 40),
) -> list[tuple[int, int, int, int]]:
    """Detect faces in a BGR frame.

    Returns a list of (x, y, w, h) bounding boxes.
    """
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    gray = equalize_histogram(gray)
    boxes = _FACE_CASCADE.detectMultiScale(
        gray, scaleFactor=scale_factor, minNeighbors=min_neighbors, minSize=min_size
    )
    return [tuple(int(v) for v in box) for box in boxes]


def draw_boxes(
    frame_bgr: np.ndarray,
    boxes: list[tuple[int, int, int, int]],
    color: tuple[int, int, int] = (0, 200, 0),
    thickness: int = 2,
) -> np.ndarray:
    annotated = frame_bgr.copy()
    for (x, y, w, h) in boxes:
        cv2.rectangle(annotated, (x, y), (x + w, y + h), color, thickness)
    return annotated
