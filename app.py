
import streamlit as st
import cv2
import numpy as np
import mediapipe as mp

from PIL import Image
from pathlib import Path
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="AI Face Analyzer",
    page_icon="🧑",
    layout="wide"
)


# ==========================================
# CUSTOM DASHBOARD DESIGN
# ==========================================

st.markdown("""
<style>

/* Main background */
.stApp {
    background: linear-gradient(135deg, #101827, #17243a);
    color: white;
}

/* Main title */
.main-title {
    text-align: center;
    font-size: 38px;
    font-weight: 900;
    color: #60a5fa;
    padding: 15px;
}

/* Subtitle */
.sub-title {
    text-align: center;
    color: #cbd5e1;
    font-size: 17px;
    margin-bottom: 25px;
}

/* Section headings */
.section-title {
    color: #7dd3fc !important;
    font-size: 25px !important;
    font-weight: 900 !important;
    text-shadow: 0 0 9px rgba(125, 211, 252, 0.45);
}

/* Result cards */
.result-card {
    background-color: #1e3a5f;
    padding: 22px;
    border-radius: 12px;
    border-left: 5px solid #60a5fa;
    margin-top: 12px;
}

.result-card h3,
.result-card h4,
.result-card p {
    color: white !important;
}

/* Footer */
.footer {
    text-align: center;
    color: #94a3b8;
    padding: 20px;
    font-size: 13px;
}

/* Webcam instructions */
.camera-instruction {
    color: #ffffff !important;
    background-color: #23334d;
    border: 2px solid #60a5fa;
    border-radius: 10px;
    padding: 15px;
    font-size: 18px;
    font-weight: bold;
    text-align: center;
    margin: 12px 0;
}

/* Webcam text */
div[data-testid="stCameraInput"] label,
div[data-testid="stCameraInput"] p {
    color: #ffffff !important;
    font-size: 17px !important;
    font-weight: bold !important;
}

/* Dashboard metric cards */
div[data-testid="stMetric"] {
    background: linear-gradient(135deg, #263d5c, #1e3553) !important;
    padding: 20px !important;
    border: 2px solid #38bdf8 !important;
    border-radius: 12px !important;
    box-shadow: 0 0 14px rgba(56, 189, 248, 0.22) !important;
}

/* Dashboard numbers */
div[data-testid="stMetric"] [data-testid="stMetricValue"],
div[data-testid="stMetric"] [data-testid="stMetricValue"] *,
div[data-testid="stMetric"] [data-testid="stMetricValue"] span {
    color: #7dd3fc !important;
    -webkit-text-fill-color: #7dd3fc !important;
    font-size: 40px !important;
    font-weight: 900 !important;
    opacity: 1 !important;
    text-shadow: 0 0 10px rgba(125, 211, 252, 0.45) !important;
}

/* Dashboard headings */
div[data-testid="stMetric"] [data-testid="stMetricLabel"],
div[data-testid="stMetric"] [data-testid="stMetricLabel"] *,
div[data-testid="stMetric"] [data-testid="stMetricLabel"] p {
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
    font-size: 18px !important;
    font-weight: 900 !important;
    opacity: 1 !important;
}

/* Analysis mode radio buttons */
div[data-testid="stRadio"] label,
div[data-testid="stRadio"] label p {
    color: #ffffff !important;
    font-size: 16px !important;
    font-weight: bold !important;
}

/* ==========================================
   IMAGE RESULT TAB VISIBILITY
   ========================================== */

/* All tab buttons */
button[data-baseweb="tab"] {
    color: #ffffff !important;
    background-color: #263d5c !important;
    font-size: 17px !important;
    font-weight: 900 !important;
    opacity: 1 !important;
    border-radius: 8px 8px 0 0 !important;
    padding: 12px 18px !important;
    border: 1px solid #60a5fa !important;
}

/* Tab text and child elements */
button[data-baseweb="tab"] *,
button[data-baseweb="tab"] p,
button[data-baseweb="tab"] div,
button[data-baseweb="tab"] span {
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
    opacity: 1 !important;
    font-weight: 900 !important;
}

/* Selected tab */
button[data-baseweb="tab"][aria-selected="true"] {
    background-color: #075985 !important;
    border-bottom: 4px solid #38bdf8 !important;
}

/* Unselected tabs */
button[data-baseweb="tab"][aria-selected="false"] {
    background-color: #263d5c !important;
}

/* Hover effect */
button[data-baseweb="tab"]:hover {
    background-color: #365477 !important;
}

/* Tab content */
div[data-baseweb="tab-panel"] {
    background-color: transparent !important;
    padding-top: 15px !important;
}

/* Expander text */
details summary,
details summary p {
    color: white !important;
    font-weight: bold !important;
}

</style>
""", unsafe_allow_html=True)


# ==========================================
# HEADER
# ==========================================

st.markdown(
    '<div class="main-title">🧑 AI-Powered Intelligent Face Analyzer</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub-title">Deep Learning | Computer Vision | Facial Landmark Analysis</div>',
    unsafe_allow_html=True
)

st.divider()


# ==========================================
# PROJECT INFORMATION
# ==========================================

with st.expander("ℹ️ About This Project"):

    st.write("""
    This project uses Artificial Intelligence and Computer Vision
    to analyze facial images captured through image upload or webcam.

    **Technologies used:**

    - Python: Main programming language
    - Streamlit: Interactive dashboard
    - OpenCV: Face detection
    - MediaPipe: Facial landmark detection
    - NumPy: Image data processing
    - Pillow: Image handling
    """)


# ==========================================
# EXPRESSION PATTERN ESTIMATION
# ==========================================

def estimate_expression(landmarks):

    left_corner = landmarks[61]
    right_corner = landmarks[291]
    upper_lip = landmarks[13]
    lower_lip = landmarks[14]

    face_height = (
        max(p.y for p in landmarks) -
        min(p.y for p in landmarks)
    )

    if face_height <= 0:
        return "Unable to estimate"

    mouth_opening = abs(
        lower_lip.y - upper_lip.y
    )

    normalized_opening = mouth_opening / face_height

    corner_average = (
        left_corner.y + right_corner.y
    ) / 2

    mouth_center = (
        upper_lip.y + lower_lip.y
    ) / 2

    smile_curve = (
        mouth_center - corner_average
    ) / face_height

    if normalized_opening > 0.035:
        return "Mouth-open pattern"

    elif smile_curve > 0.015:
        return "Smile-like mouth pattern"

    else:
        return "No strong mouth pattern"


# ==========================================
# ANALYSIS MODE SELECTION
# ==========================================

st.markdown(
    '<div class="section-title">📷 Choose Analysis Mode</div>',
    unsafe_allow_html=True
)

analysis_mode = st.radio(
    "Select how you want to analyze:",
    ["Upload Image", "Live Webcam"],
    horizontal=True
)

st.divider()


# ==========================================
# IMAGE INPUT
# ==========================================

source_file = None

if analysis_mode == "Upload Image":

    st.markdown(
        '<div class="section-title">📤 Upload Your Image</div>',
        unsafe_allow_html=True
    )

    source_file = st.file_uploader(
        "Choose a face image",
        type=["jpg", "jpeg", "png"]
    )

else:

    st.markdown(
        '<div class="section-title">📸 LIVE WEBCAM CAPTURE</div>',
        unsafe_allow_html=True
    )

    st.markdown("""
    <div class="camera-instruction">
        📷 STEP 1: ALLOW CAMERA ACCESS<br><br>
        STEP 2: POSITION YOUR FACE IN THE CAMERA<br><br>
        STEP 3: CLICK THE SNAPSHOT BUTTON TO TAKE YOUR PHOTO
    </div>
    """, unsafe_allow_html=True)

    source_file = st.camera_input(
        "📸 TAKE YOUR PHOTO",
        help="Click Snapshot to capture your image."
    )

    st.markdown("""
    <div class="camera-instruction">
        ☝️ CLICK THE SNAPSHOT BUTTON ABOVE TO TAKE YOUR PHOTO
    </div>
    """, unsafe_allow_html=True)


# ==========================================
# IMAGE ANALYSIS
# ==========================================

if source_file is not None:

    try:

        # Read uploaded or captured image
        image = Image.open(source_file).convert("RGB")
        image_array = np.array(image)

        # --------------------------------------
        # FACE DETECTION USING OPENCV
        # --------------------------------------

        gray_image = cv2.cvtColor(
            image_array,
            cv2.COLOR_RGB2GRAY
        )

        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades +
            "haarcascade_frontalface_default.xml"
        )

        if face_cascade.empty():
            st.error("Could not load OpenCV face detection model.")
            st.stop()

        faces = face_cascade.detectMultiScale(
            gray_image,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )

        result_image = image_array.copy()

        for (x, y, w, h) in faces:

            # Draw green rectangle
            cv2.rectangle(
                result_image,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                4
            )

            # Add detection label
            cv2.putText(
                result_image,
                "Face Detected",
                (x, max(y - 10, 25)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

        # --------------------------------------
        # FACIAL LANDMARK DETECTION
        # --------------------------------------

        model_path = (
            Path(__file__).parent /
            "models" /
            "face_landmarker.task"
        )

        if not model_path.exists():
            st.error("MediaPipe model file not found!")
            st.stop()

        base_options = python.BaseOptions(
            model_asset_path=str(model_path)
        )

        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_faces=5
        )

        with vision.FaceLandmarker.create_from_options(
            options
        ) as landmarker:

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=image_array
            )

            detection_result = landmarker.detect(mp_image)

        landmark_image = image_array.copy()
        expression_results = []

        for index, face_landmarks in enumerate(
            detection_result.face_landmarks
        ):

            # Draw visible landmark dots
            for landmark in face_landmarks:

                x = int(
                    landmark.x * image_array.shape[1]
                )

                y = int(
                    landmark.y * image_array.shape[0]
                )

                if (
                    0 <= x < image_array.shape[1]
                    and 0 <= y < image_array.shape[0]
                ):

                    # Dark outer circle
                    cv2.circle(
                        landmark_image,
                        (x, y),
                        7,
                        (20, 20, 20),
                        -1
                    )

                    # Bright yellow inner circle
                    cv2.circle(
                        landmark_image,
                        (x, y),
                        5,
                        (255, 255, 0),
                        -1
                    )

            # Estimate mouth pattern
            expression = estimate_expression(
                face_landmarks
            )

            expression_results.append({
                "Face": index + 1,
                "Estimated Pattern": expression
            })

        # ======================================
        # DASHBOARD SUMMARY
        # ======================================

        st.divider()

        st.markdown(
            '<div class="section-title">📊 Analysis Dashboard</div>',
            unsafe_allow_html=True
        )

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "OpenCV Faces",
                len(faces)
            )

        with col2:
            st.metric(
                "Landmark Faces",
                len(detection_result.face_landmarks)
            )

        with col3:
            st.metric(
                "Image Size",
                f"{image_array.shape[1]} × {image_array.shape[0]}"
            )

        # ======================================
        # IMAGE ANALYSIS RESULTS
        # ======================================

        st.divider()

        st.markdown(
            '<div class="section-title">🖼️ Image Analysis Results</div>',
            unsafe_allow_html=True
        )

        # Three clearly labelled tabs
        tab1, tab2, tab3 = st.tabs([
            "🖼️ Original Image",
            "🟩 Face Detection",
            "🟡 Facial Landmarks"
        ])

        with tab1:

            st.image(
                image_array,
                caption="Original Image",
                width="stretch"
            )

        with tab2:

            st.image(
                result_image,
                caption="OpenCV Face Detection - Green Boxes",
                width="stretch"
            )

            if len(faces) > 0:
                st.success(
                    f"{len(faces)} face region(s) detected by OpenCV."
                )
            else:
                st.warning("No faces detected by OpenCV.")

        with tab3:

            st.image(
                landmark_image,
                caption="MediaPipe Facial Landmarks - Yellow Dots",
                width="stretch"
            )

            if len(detection_result.face_landmarks) > 0:
                st.success(
                    f"{len(detection_result.face_landmarks)} face(s) with landmarks detected."
                )
            else:
                st.warning("No facial landmarks detected.")

        # ======================================
        # DETECTION STATUS
        # ======================================

        st.divider()

        st.markdown(
            '<div class="section-title">✅ Detection Status</div>',
            unsafe_allow_html=True
        )

        if len(faces) > 0:
            st.success("OpenCV face detection completed!")
        else:
            st.warning("No faces detected by OpenCV.")

        if len(detection_result.face_landmarks) > 0:
            st.success("MediaPipe landmark detection completed!")
        else:
            st.warning("No facial landmarks detected.")

        # ======================================
        # EXPRESSION RESULTS
        # ======================================

        st.divider()

        st.markdown(
            '<div class="section-title">🧠 Expression Pattern Estimation</div>',
            unsafe_allow_html=True
        )

        if expression_results:

            for result in expression_results:

                st.markdown(
                    f"""
                    <div class="result-card">
                        <h3>👤 Face {result['Face']}</h3>
                        <h4>Estimated Pattern</h4>
                        <p>{result['Estimated Pattern']}</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        else:

            st.info(
                "No facial landmarks available for pattern estimation."
            )

        st.caption(
            "Note: These are approximate mouth-pattern classifications "
            "based on facial landmarks. They do not establish a person's "
            "actual emotions."
        )

    except Exception as error:

        st.error("An error occurred during image analysis.")
        st.exception(error)


else:

    if analysis_mode == "Upload Image":

        st.info(
            "👆 Upload an image above to begin face analysis."
        )

    else:

        st.info(
            "📸 Capture a webcam photo to begin face analysis."
        )


# ==========================================
# FOOTER
# ==========================================

st.divider()

st.markdown(
    """
    <div class="footer">
        AI-Powered Intelligent Face Analyzer<br>
        Built with Python, OpenCV, MediaPipe and Streamlit
    </div>
    """,
    unsafe_allow_html=True
)
