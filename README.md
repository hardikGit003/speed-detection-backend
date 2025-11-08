# Vehicle Speed Detection using YOLO

This project uses YOLO (You Only Look Once) object detection to detect vehicles and estimate their speed using computer vision techniques.

## Features
- Real-time vehicle detection using YOLOv8
- Speed estimation using pixel-based tracking
- Support for video input (webcam or video file)
- Visual display of detected vehicles and their speeds

## Setup Instructions

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Linux/Mac
# or
.\venv\Scripts\activate  # On Windows
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the program:
```bash
python speed_detection.py
```

## Usage
- Press 'q' to quit the program
- The program will automatically download the YOLOv8 model on first run
- Adjust the `SPEED_LIMIT` variable in the code to set the speed threshold for alerts

## Note
- The speed estimation is approximate and depends on camera calibration and distance
- For accurate speed measurements, proper camera calibration is required
- The default model is trained on general object detection and may need fine-tuning for specific use cases 