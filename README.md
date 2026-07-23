# Real-Time Face Recognition & Attendance System

A Streamlit web app for real-time face detection, facial landmark recognition, and attendance
logging — built with OpenCV and dlib.

## How it works

1. **Enroll a face** — upload a reference photo and a name. The app computes a 128-d face
   encoding (dlib's ResNet face-recognition model) and keeps it in memory for the session.
2. **Live webcam** — every frame is preprocessed with grayscale conversion + histogram
   equalization, then run through an Adaboost/Haar cascade classifier to box faces in real time.
   Every 10th frame is handed off to a background thread that computes facial landmarks and a
   face encoding (dlib's HOG detector + 68-point shape predictor), then matches it against
   enrolled faces — this keeps the live video loop fast while the heavier recognition work
   happens off the main thread.
3. **Upload a photo** — same detection + recognition pipeline, run once against a still image,
   for anyone who'd rather not use their webcam.
4. **Attendance log** — recognized faces are logged with a timestamp (60s cooldown per person to
   avoid duplicate entries across consecutive frames), viewable and downloadable as CSV.

All state (enrolled faces, attendance log) is in-memory for the current session only — nothing
is persisted to disk or a database.

## Tech stack

- Python, OpenCV (Haar cascade detection, histogram equalization)
- dlib (HOG face detector, 68-point landmarks, 128-d face encodings) — called directly rather
  than through the `face_recognition` wrapper package, see note below
- Streamlit + `streamlit-webrtc` (browser webcam access, deployable web UI)
- Threading (background recognition worker, decoupled from the video callback)

## Running locally

Requires Python 3.11+.

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

> **Why `dlib-bin` instead of `dlib`:** the real `dlib` PyPI package ships only a source tarball —
> no prebuilt wheels on any platform — so installing it means compiling from scratch every time
> (tens of minutes, and it needs a C++ compiler + CMake). [`dlib-bin`](https://pypi.org/project/dlib-bin/)
> publishes prebuilt wheels for Linux/macOS/Windows and installs in seconds; it's an
> unofficial but widely-used drop-in (same `import dlib`). The catch: the popular
> `face_recognition` wrapper package hard-declares a dependency on real `dlib`, so it can't be
> used alongside `dlib-bin` via a plain `pip install -r requirements.txt`. This repo calls dlib's
> HOG detector, shape predictor, and face-recognition model directly instead (see
> `src/recognizer.py`) — the same three primitives `face_recognition` wraps, using the same
> bundled models via `face_recognition_models`.

## Running tests

```bash
pytest
```

Covers histogram equalization, Haar cascade detection (shape/type contracts), the attendance
cooldown logic, and the encoding/matching logic. There's no bundled test photo (to avoid shipping
a third-party face image with unclear usage rights) — verify actual detection/recognition
accuracy yourself via the Enroll and Upload-a-photo tabs with your own photos.

## Deploying (Streamlit Community Cloud)

1. Push this repo to GitHub.
2. On [share.streamlit.io](https://share.streamlit.io): New app → select this repo → main file
   `app.py`.
3. A `.python-version` file pins Python 3.11 — Streamlit Cloud's `uv`-based installer reads this
   (not the older `runtime.txt` convention) to pick the interpreter.
4. `apt.txt` installs `ffmpeg`, needed by `av`/`streamlit-webrtc`. No C++ build toolchain is
   needed since `dlib-bin` is a prebuilt wheel — deploys take well under a minute.

**Live webcam tab caveat:** `streamlit-webrtc` needs a STUN server for the browser to reach the
deployed server; this app is configured with Google's public STUN server, which works for most
networks. If your webcam won't connect after deploying, check the browser's camera permission
prompt and console first.
