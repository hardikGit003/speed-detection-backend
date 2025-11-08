from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import shutil
import os
import re
import cv2
from speed_detection import SpeedDetector

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # React app URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create directories if not exist
os.makedirs("uploads", exist_ok=True)
os.makedirs("processed", exist_ok=True)


@app.post("/upload-video")
async def upload_video(file: UploadFile = File(...)):
    try:
        # ✅ Make filename safe (remove spaces and special chars)
        original_filename = file.filename
        safe_filename = re.sub(r'[^A-Za-z0-9._-]', '_', original_filename)

        # ✅ Define paths
        input_path = os.path.join("uploads", safe_filename)
        temp_output_path = os.path.join("processed", f"temp_output_{safe_filename}")
        output_path = os.path.join("processed", f"output_{safe_filename}")

        # ✅ Save uploaded file
        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # ✅ Process video with your model
        detector = SpeedDetector()
        cap = cv2.VideoCapture(input_path)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30  # fallback if FPS=0

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_output_path, fourcc, fps, (frame_width, frame_height))

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            processed_frame = detector.process_frame(frame)
            out.write(processed_frame)

        cap.release()
        out.release()

        # ✅ Use FFmpeg safely (with quotes)
        os.system(f'ffmpeg -i "{temp_output_path}" -vcodec libx264 "{output_path}" -y')

        # ✅ Clean up
        if os.path.exists(temp_output_path):
            os.remove(temp_output_path)

        print(f"✅ Video saved at: {output_path}")

        # ✅ Return processed filename
        return {"filename": f"output_{safe_filename}"}

    except Exception as e:
        print("❌ Error:", str(e))
        return {"error": str(e)}


@app.get("/video/{filename}")
async def get_video(filename: str):
    video_path = os.path.join("processed", filename)
    if not os.path.exists(video_path):
        return {"error": "File not found"}
    return FileResponse(video_path, media_type="video/mp4")
