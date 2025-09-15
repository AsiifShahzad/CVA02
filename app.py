import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pathlib
from scipy import stats

st.set_page_config(page_title="Bone Fracture Enhancement", layout="wide")
st.markdown(
    "<h1 style='text-align:center; color:#2c3e50;'>Spatial Filtering for Bone X-ray Enhancement</h1>",
    unsafe_allow_html=True
)

def load_image_gray(file_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(file_bytes, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)

def apply_sobel(img: np.ndarray) -> np.ndarray:
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    sobel = cv2.magnitude(sobelx, sobely)
    sobel = cv2.convertScaleAbs(sobel)
    return cv2.normalize(sobel, None, 0, 255, cv2.NORM_MINMAX)

def apply_laplacian(img: np.ndarray) -> np.ndarray:
    laplacian = cv2.Laplacian(img, cv2.CV_64F, ksize=3)
    laplacian = cv2.convertScaleAbs(laplacian)
    laplacian = cv2.normalize(laplacian, None, 0, 255, cv2.NORM_MINMAX)
    return cv2.resize(laplacian, (img.shape[1], img.shape[0])) 

def apply_mean(img: np.ndarray, k: int = 3) -> np.ndarray:
    return cv2.blur(img, (k, k))

def apply_median(img: np.ndarray, k: int = 3) -> np.ndarray:
    return cv2.medianBlur(img, k)

def apply_mode(img: np.ndarray, k: int = 3) -> np.ndarray:
    """Apply mode filter using a sliding window."""
    pad = k // 2
    padded = np.pad(img, pad, mode="edge")
    output = np.zeros_like(img)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            window = padded[i:i+k, j:j+k].flatten()
            mode_val = stats.mode(window, keepdims=True)[0][0]
            output[i, j] = mode_val
    return output

def plot_image(image: np.ndarray, title: str):
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(image, cmap='gray', vmin=0, vmax=255)
    ax.set_title(title)
    ax.axis('off')
    st.pyplot(fig)

uploaded = st.file_uploader("Upload X-ray Image", type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"])

DEMO_PATH = pathlib.Path("sample.jpg")

if uploaded:
    image = load_image_gray(uploaded.read())
else:
    if DEMO_PATH.exists():
        with open(DEMO_PATH, "rb") as f:
            image = load_image_gray(f.read())
        st.info("Showing built-in demo X-ray (upload your own to override).")
    else:
        st.warning("No image uploaded and no demo file found.")
        st.stop()

st.sidebar.header("Choose Filter")
choice = st.sidebar.radio("Operation", [
    "Original", 
    "Sobel (1st Derivative)", 
    "Laplacian (2nd Derivative)",
    "Mean Filter", 
    "Median Filter", 
    "Mode Filter",
    "Smoothing + Sobel", 
    "Smoothing + Laplacian"
])

kernel_size = st.sidebar.slider("Kernel Size (for smoothing)", 3, 9, 3, step=2)

if choice == "Original":
    processed = None  
    title = "Original"

elif choice == "Sobel (1st Derivative)":
    processed = apply_sobel(image)
    title = "Sobel Edge Detection"

elif choice == "Laplacian (2nd Derivative)":
    processed = apply_laplacian(image)
    title = "Laplacian Edge Detection"

elif choice == "Mean Filter":
    processed = apply_mean(image, kernel_size)
    title = f"Mean Filter (k={kernel_size})"

elif choice == "Median Filter":
    processed = apply_median(image, kernel_size)
    title = f"Median Filter (k={kernel_size})"

elif choice == "Mode Filter":
    processed = apply_mode(image, kernel_size)
    title = f"Mode Filter (k={kernel_size})"

elif choice == "Smoothing + Sobel":
    smoothed = apply_median(image, kernel_size)
    processed = apply_sobel(smoothed)
    title = f"Smoothing + Sobel (k={kernel_size})"

elif choice == "Smoothing + Laplacian":
    smoothed = apply_median(image, kernel_size)
    processed = apply_laplacian(smoothed)
    title = f"Smoothing + Laplacian (k={kernel_size})"
    
col1, col2 = st.columns(2)

with col1:
    plot_image(image, "Original Image")

with col2:
    if processed is not None:
        plot_image(processed, title)
