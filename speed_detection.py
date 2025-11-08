import cv2
import numpy as np
from ultralytics import YOLO
from license_plate_recognition import LicensePlateRecognizer
import time
from collections import defaultdict
import torch

class SpeedDetector:
    def __init__(self):
        # Initialize YOLO model (use GPU if available)
        self.model = YOLO('yolov8n.pt')
        if torch.cuda.is_available():
            self.model.to('cuda')

        # Initialize license plate recognizer
        self.lpr = LicensePlateRecognizer()

        # Speed detection parameters
        self.PIXELS_PER_METER = 15  
        self.speed_history = {}
        self.position_history = {}

        # Vehicle classes to track (car, bus, truck)
        self.target_classes = [2, 5, 7]
        self.class_names = ['car', 'bus', 'truck']
        self.class_mapping = {2: 0, 5: 1, 7: 2}

        # Speed limit settings
        self.SPEED_LIMIT = 30  
        self.smoothed_speeds = {}  

        # Track vehicle history
        self.track_history = defaultdict(lambda: [])

    def calculate_speed(self, prev_pos, curr_pos, fps):
        """Calculate speed in km/h"""
        if prev_pos is None or curr_pos is None:
            return 0
        
        # Calculate pixel movement
        pixel_distance = np.sqrt(
            (curr_pos[0] - prev_pos[0])**2 + 
            (curr_pos[1] - prev_pos[1])**2
        )
        
        # Convert pixels to meters
        distance_meters = pixel_distance / self.PIXELS_PER_METER
        
        # Convert to km/h
        return (distance_meters / (1 / fps)) * 3.6

    def get_smoothed_speed(self, track_id, current_speed):
        """Apply moving average for smooth speed readings"""
        if track_id not in self.speed_history:
            self.speed_history[track_id] = []
        
        self.speed_history[track_id].append(current_speed)

        # Keep only last 3 readings
        if len(self.speed_history[track_id]) > 3:
            self.speed_history[track_id].pop(0)

        return sum(self.speed_history[track_id]) / len(self.speed_history[track_id])

    def process_frame(self, frame):
        """Process a single frame for vehicle detection and speed calculation"""
        # Lower confidence for faster processing
        results = self.model.track(frame, persist=True, classes=self.target_classes, conf=0.25, iou=0.5)

        # Get first detection result
        result = results[0]

        if result.boxes is not None:
            for box in result.boxes:
                track_id = int(box.id) if box.id is not None else None
                conf = float(box.conf)
                
                if track_id is None or conf < 0.25:
                    continue

                class_id = int(box.cls)
                if class_id not in self.class_mapping:
                    continue
                class_idx = self.class_mapping[class_id]

                # Get bounding box
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                # Find center point
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                current_pos = (center_x, center_y)

                if track_id not in self.position_history:
                    self.position_history[track_id] = []

                self.position_history[track_id].append(current_pos)

                if len(self.position_history[track_id]) > 2:
                    self.position_history[track_id].pop(0)

                if len(self.position_history[track_id]) == 2:
                    speed = self.calculate_speed(
                        self.position_history[track_id][0],
                        self.position_history[track_id][1],
                        30  # Approximate FPS
                    )
                    
                    # Smooth speed values
                    smoothed_speed = self.get_smoothed_speed(track_id, speed)

                    # Ignore unrealistic speeds
                    if smoothed_speed > 150 or smoothed_speed < 1:
                        continue

                    # Draw vehicle box and speed label
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f"{self.class_names[class_idx]} {smoothed_speed:.1f} km/h"
                    cv2.putText(frame, label, (x1, y1-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Only process license plates if vehicles are detected
        if len(result.boxes) > 0:
            frame = self.lpr.process_frame(frame)

        return frame

def main():
    cap = cv2.VideoCapture('videoplayback.mp4')

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Reduce output FPS for faster playback
    output_fps = min(fps, 10)

    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output_speed_detection.mp4', fourcc, output_fps, (frame_width, frame_height))

    detector = SpeedDetector()

    frame_count = 0  

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Skip 2 out of every 3 frames to increase speed
        if frame_count % 3 != 0:
            continue

        # Process frame
        processed_frame = detector.process_frame(frame)

        # Write frame to output video
        out.write(processed_frame)

        # Display frame
        cv2.imshow('Speed Detection', processed_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Clean up
    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
