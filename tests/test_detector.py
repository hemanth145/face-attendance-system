import numpy as np

from src.detector import detect_faces, draw_boxes, equalize_histogram


def test_equalize_histogram_preserves_shape_and_dtype():
    gray = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
    result = equalize_histogram(gray)
    assert result.shape == gray.shape
    assert result.dtype == gray.dtype


def test_detect_faces_on_random_noise_returns_no_crash_and_a_list():
    frame = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
    boxes = detect_faces(frame)
    assert isinstance(boxes, list)
    for box in boxes:
        assert len(box) == 4


def test_draw_boxes_draws_without_mutating_input():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    original = frame.copy()
    annotated = draw_boxes(frame, [(10, 10, 30, 30)])
    assert np.array_equal(frame, original)
    assert not np.array_equal(annotated, frame)
