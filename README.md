# Real-Time Face Recognition & Attendance System

A Streamlit web app for real-time face detection, facial landmark recognition, and attendance
logging — built with OpenCV and dlib.

## How it works

1. **Enroll a face** — upload a reference photo and a name. The app computes a 128-d face
   encoding (dlib, via `face_recognition`) and keeps it in memory for the session.
2. **Live webcam** — every frame is preprocessed with grayscale conversion + histogram
   equalization, then run through an Adaboost/Haar cascade classifier to box faces in real time.
   Every 10th frame is handed off to a background thread that computes facial landmarks and a
   face encoding, then matches it against enrolled faces — this keeps the live video loop fast
   while the heavier recognition work happens off the main thread.
3. **Upload a photo** — same detection + recognition pipeline, run once against a still image,
   for anyone who'd rather not use their webcam.
4. **Attendance log** — recognized faces are logged with a timestamp (60s cooldown per person to
   avoid duplicate entries across consecutive frames), viewable and downloadable as CSV.

All state (enrolled faces, attendance log) is in-memory for the current session only — nothing
is persisted to disk or a database.

## Tech stack

- Python, OpenCV (Haar cascade detection, histogram equalization)
- dlib / `face_recognition` (facial landmark detection, 128-d face encodings)
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

> **Windows note:** `dlib` has no official prebuilt wheel for newer Python versions and needs a
> C++ compiler (CMake + MSVC) to build from source. If you don't have those installed, use the
> [`dlib-bin`](https://pypi.org/project/dlib-bin/) package instead (prebuilt wheels, same `import
> dlib` API) — `pip install dlib-bin` before installing the rest of `requirements.txt`, and remove
> the `dlib` line so pip doesn't try to rebuild it. On Linux (including Streamlit Cloud), `dlib`
> builds fine from source as long as `cmake` and a C++ toolchain are available (see `apt.txt`).

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
3. Streamlit Cloud reads `apt.txt` automatically to install system build dependencies (`cmake`,
   `build-essential`, etc.) before `pip install -r requirements.txt` — no extra configuration
   needed.
4. First deploy will take a while (dlib compiles from source, typically several minutes).

**Live webcam tab caveat:** `streamlit-webrtc` needs a STUN server for the browser to reach the
deployed server; this app is configured with Google's public STUN server, which works for most
networks. If your webcam won't connect after deploying, check the browser's camera permission
prompt and console first.
