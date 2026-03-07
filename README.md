# FractureVision AI 🦴

FractureVision AI is a Streamlit-based web application that uses a custom-trained YOLO object detection model to detect bone fractures in X-ray images. 

This repository contains the local application code and the necessary model weights.

## 🌟 Features
- **Medical-Grade UI:** A clean, glassmorphic, and highly responsive user interface.
- **Instant Inference:** Upload an X-ray image (JPG, PNG, JPEG) and get instant bounding box predictions around detected fractures.
- **Confidence Scoring:** Outputs the confidence metrics of the detections.

## 🛠️ Installation & Setup (Local)

Follow these steps to run the application on your local machine.

### 1. Prerequisites
- Python 3.8+ installed on your system.

### 2. Clone/Download the Repository
Ensure you have all the project files in a single folder (e.g., `Xray_detection\fracture`), including:
- `app.py` (The main application script)
- `best (1).pt` (The YOLO model weights)
- `requirements.txt` (List of dependencies)

### 3. Create a Virtual Environment (Recommended)
It's highly recommended to run this app inside a virtual environment to prevent dependency conflicts.
Open your terminal (or Command Prompt/PowerShell), navigate to your project folder, and run:

```bash
# Create the virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows:
.\venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 4. Install Dependencies
With your virtual environment activated, install the required Python packages:

```bash
pip install -r requirements.txt
```

*(Note: The main dependencies are `streamlit`, `ultralytics`, and `Pillow`)*

## 🚀 Running the App

Once everything is installed, you can launch the app locally using Streamlit:

```bash
# Ensure your virtual environment is activated, then run:
streamlit run app.py
```

This will start a local server, and your default web browser will automatically open to `http://localhost:8501` displaying the FractureVision AI dashboard.

## 📖 How to Use
1. Once the app is running in your browser, look for the file upload dropzone.
2. Drag and drop an X-ray image or click to browse your files.
3. The AI will process the image (this takes just a few seconds).
4. View the results on the screen! Detected fractures will be highlighted with bounding boxes.

## ⚠️ Disclaimer
This tool is for **educational and research purposes only**. It should NOT be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare provider with any questions you may have regarding a medical condition.
