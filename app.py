"""Real-Time Face Recognition & Attendance System.

Streamlit web app demonstrating: Haar cascade face detection with histogram
equalization preprocessing, dlib-based facial landmark detection and
recognition (via face_recognition), and multithreaded processing so the
live video path stays responsive while the heavier recognition work runs
in the background.
"""
from __future__ import annotations

import queue
import time

import av
import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import RTCConfiguration, WebRtcMode, webrtc_streamer

from src.attendance import AttendanceLog
from src.detector import detect_faces, draw_boxes
from src.recognizer import RecognitionResult, RecognitionWorker, encode_face, match_encoding

st.set_page_config(page_title="Face Recognition & Attendance", page_icon="🎥", layout="wide")

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)
MATCH_TOLERANCE = 0.6
ATTENDANCE_COOLDOWN_SECONDS = 60.0
RECOGNIZE_EVERY_N_FRAMES = 10


def _read_image(uploaded_file) -> np.ndarray:
    file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
    return cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)


if "known_faces" not in st.session_state:
    st.session_state.known_faces: dict[str, np.ndarray] = {}
if "attendance" not in st.session_state:
    st.session_state.attendance = AttendanceLog(cooldown_seconds=ATTENDANCE_COOLDOWN_SECONDS)

st.title("🎥 Real-Time Face Recognition & Attendance System")
st.caption(
    "Python · OpenCV (Haar cascade + histogram equalization) · dlib / face_recognition "
    "(facial landmarks + encodings) · multithreaded recognition worker"
)

tab_enroll, tab_webcam, tab_upload, tab_log = st.tabs(
    ["1. Enroll a face", "2. Live webcam", "3. Upload a photo", "4. Attendance log"]
)

with tab_enroll:
    st.write(
        "Upload a clear, front-facing photo and give it a name. The system remembers "
        "this face for the rest of your session only — nothing is written to disk."
    )
    name = st.text_input("Name")
    enroll_file = st.file_uploader("Reference photo", type=["jpg", "jpeg", "png"], key="enroll")
    if st.button("Enroll", disabled=not (name and enroll_file)):
        image_bgr = _read_image(enroll_file)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        encoding, _ = encode_face(image_rgb)
        if encoding is None:
            st.error("No face detected in that photo — try a clearer, front-facing image.")
        else:
            st.session_state.known_faces[name] = encoding
            st.success(f"Enrolled '{name}'.")

    if st.session_state.known_faces:
        st.write("Currently enrolled:", ", ".join(st.session_state.known_faces.keys()))
    else:
        st.info("No faces enrolled yet — recognition has nothing to match against until you add one.")


class AttendanceVideoProcessor:
    """Boxes every face each frame (cheap); submits every Nth frame to a
    background thread for landmark/encoding/recognition (expensive), and
    exposes recognized names through a thread-safe queue the main Streamlit
    thread can drain (touching st.session_state from this callback's thread
    is unsafe, so results are handed off instead).
    """

    def __init__(self, known_faces: dict[str, np.ndarray]) -> None:
        self.worker = RecognitionWorker(known_faces, tolerance=MATCH_TOLERANCE)
        self.result_queue: "queue.Queue[RecognitionResult]" = queue.Queue()
        self._frame_count = 0

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        boxes = detect_faces(img)
        annotated = draw_boxes(img, boxes)

        self._frame_count += 1
        if self._frame_count % RECOGNIZE_EVERY_N_FRAMES == 0:
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self.worker.submit(rgb)

        result = self.worker.latest_result()
        if result and result.name:
            top, right, bottom, left = result.box
            cv2.putText(
                annotated, result.name, (left, max(top - 10, 0)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 0), 2,
            )
            try:
                self.result_queue.put_nowait(result)
            except queue.Full:
                pass

        return av.VideoFrame.from_ndarray(annotated, format="bgr24")


with tab_webcam:
    if not st.session_state.known_faces:
        st.info("Enroll at least one face first so there's someone to recognize.")
    st.write(
        "Grant camera access when your browser prompts you. Every face is boxed in "
        "real time; recognized faces are labeled and logged to attendance "
        f"(with a {int(ATTENDANCE_COOLDOWN_SECONDS)}s cooldown per person)."
    )

    known_faces_snapshot = dict(st.session_state.known_faces)
    ctx = webrtc_streamer(
        key="attendance-webcam",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_processor_factory=lambda: AttendanceVideoProcessor(known_faces_snapshot),
        media_stream_constraints={"video": True, "audio": False},
    )

    status_placeholder = st.empty()
    if ctx.state.playing:
        while ctx.state.playing:
            if ctx.video_processor:
                try:
                    result = ctx.video_processor.result_queue.get(timeout=1.0)
                except queue.Empty:
                    result = None
                if result and result.name:
                    if st.session_state.attendance.mark_present(result.name):
                        status_placeholder.success(
                            f"Marked '{result.name}' present at {time.strftime('%H:%M:%S')}"
                        )
            else:
                time.sleep(0.1)

with tab_upload:
    st.write(
        "Prefer not to use your webcam? Upload a photo instead — same detection + "
        "recognition pipeline, run once."
    )
    upload_file = st.file_uploader("Photo", type=["jpg", "jpeg", "png"], key="upload")
    if upload_file is not None:
        image_bgr = _read_image(upload_file)
        boxes = detect_faces(image_bgr)
        annotated = draw_boxes(image_bgr, boxes)

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        encoding, _location = encode_face(image_rgb)
        label = "No face detected"
        if encoding is not None:
            best_name, best_distance = match_encoding(
                encoding, st.session_state.known_faces, MATCH_TOLERANCE
            )
            if best_name:
                st.session_state.attendance.mark_present(best_name)
                label = f"Recognized: {best_name} (distance {best_distance:.2f})"
            else:
                label = "Face detected, but not recognized"

        st.image(
            cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
            caption=label,
            width="stretch",
        )

with tab_log:
    df = st.session_state.attendance.to_dataframe()
    st.dataframe(df, width="stretch")
    if len(df):
        st.download_button("Download CSV", st.session_state.attendance.to_csv(), file_name="attendance.csv")
    else:
        st.info("No attendance recorded yet this session.")
