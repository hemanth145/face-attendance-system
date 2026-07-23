import time

import numpy as np

from src.recognizer import RecognitionWorker, encode_face, match_encoding


def test_encode_face_on_random_noise_finds_no_face():
    image = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
    encoding, location = encode_face(image)
    assert encoding is None
    assert location is None


def test_match_encoding_picks_closest_known_face_within_tolerance():
    target = np.zeros(128)
    known = {
        "Alice": np.full(128, 0.05),
        "Bob": np.full(128, 5.0),
    }
    name, distance = match_encoding(target, known, tolerance=0.6)
    assert name == "Alice"
    assert distance < 0.6


def test_match_encoding_returns_none_when_no_face_within_tolerance():
    target = np.zeros(128)
    known = {"Bob": np.full(128, 5.0)}
    name, _distance = match_encoding(target, known, tolerance=0.6)
    assert name is None


def test_recognition_worker_reports_no_result_for_noise_frames():
    worker = RecognitionWorker(known_encodings={})
    frame = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
    worker.submit(frame)
    time.sleep(1.0)
    result = worker.latest_result()
    assert result is None
    worker.stop()
