import cv2
import numpy as np
import datetime
import csv
import os
from collections import deque
from Video_recorder import MotionVideoRecorder

class MotionDetector:
    def __init__(self):
        # Configuration
        self.THRESHOLD = 5
        self.MIN_CONTOUR_AREA = 1000
        self.BLUR_SIZE = (11, 11)
        self.DILATE_ITERATIONS = 2
        self.LOG_DATA = True
        self.SHOW_DEBUG_WINDOWS = False
        self.MOTION_PERSISTENCE = 5
        
        # Initialize components
        self.back_sub = cv2.createBackgroundSubtractorMOG2(
            history=500, 
            varThreshold=16, 
            detectShadows=False
        )
        self.motion_history = deque(maxlen=self.MOTION_PERSISTENCE)
        self.prev_time = datetime.datetime.now()
        self.cap = None
        self.recorder = None
        self.log_file = None
        self.csv_writer = None
        self.current_frame = None
        
    def initialize_camera(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("Error: Camera not accessible")
            
        frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        self.recorder = MotionVideoRecorder(
            output_dir="motion_clips",
            pre_buffer_sec=1,
            clip_duration=5,
            fps=30,
            width=frame_width,
            height=frame_height
        )
        
    def initialize_logging(self):
        if self.LOG_DATA:
            log_dir = "motion_logs"
            os.makedirs(log_dir, exist_ok=True)
            self.log_file = open(f"{log_dir}/motion_events_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", "w")
            self.csv_writer = csv.writer(self.log_file)
            self.csv_writer.writerow(["Timestamp", "Event", "Contour_Area", "Position"])
    
    def process_frame(self, frame):
        # Background subtraction
        fg_mask = self.back_sub.apply(frame)
        
        # Post-processing
        _, thresh = cv2.threshold(fg_mask, self.THRESHOLD, 255, cv2.THRESH_BINARY)
        blurred = cv2.GaussianBlur(thresh, self.BLUR_SIZE, 0)
        dilated = cv2.dilate(blurred, None, iterations=self.DILATE_ITERATIONS)
        eroded = cv2.erode(dilated, None, iterations=self.DILATE_ITERATIONS)
        
        return eroded
    
    def detect_motion(self, processed_frame):
        contours, _ = cv2.findContours(processed_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        current_motion = False
        
        for contour in contours:
            if cv2.contourArea(contour) < self.MIN_CONTOUR_AREA:
                continue
                
            current_motion = True
            x, y, w, h = cv2.boundingRect(contour)
            centroid = (x + w//2, y + h//2)
            
            # Draw bounding box and centroid
            cv2.rectangle(self.current_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.circle(self.current_frame, centroid, 5, (0, 0, 255), -1)
            
            if self.LOG_DATA:
                self.log_event(contour, centroid)
        
        return current_motion
    
    def log_event(self, contour, centroid):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        event_id = f"EV_{timestamp.replace(':', '').replace(' ', '_')}"
        self.csv_writer.writerow([timestamp, event_id, cv2.contourArea(contour), centroid])
        cv2.imwrite(f"motion_logs/{event_id}.jpg", self.current_frame)
    
    def update_motion_history(self, current_motion):
        self.motion_history.append(current_motion)
        return all(self.motion_history) if len(self.motion_history) == self.MOTION_PERSISTENCE else False
    
    def add_status_info(self, status, fps):
        cv2.putText(self.current_frame, f"Status: {status}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(self.current_frame, f"FPS: {fps:.1f}", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(self.current_frame, datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), (10, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    def run(self):
        self.initialize_camera()
        self.initialize_logging()
        
        try:
            while self.cap.isOpened():
                ret, self.current_frame = self.cap.read()
                if not ret:
                    break
                
                # Calculate FPS
                current_time = datetime.datetime.now()
                fps = 1 / ((current_time - self.prev_time).total_seconds() + 1e-6)
                self.prev_time = current_time
                
                # Process frame and detect motion
                processed_frame = self.process_frame(self.current_frame)
                current_motion = self.detect_motion(processed_frame)
                confirmed_motion = self.update_motion_history(current_motion)
                
                # Update recorder
                is_recording = self.recorder.update(self.current_frame, confirmed_motion)
                if is_recording:
                    cv2.putText(self.current_frame, "REC", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                # Display status
                status = "Motion detected!" if current_motion else "No motion"
                self.add_status_info(status, fps)
                
                # Show debug windows
                if self.SHOW_DEBUG_WINDOWS:
                    fg_mask = self.process_frame
                    cv2.imshow("Foreground Mask", fg_mask)
                    cv2.imshow("Processed Mask", processed_frame)
                
                # Display output
                cv2.imshow("Advanced Motion Detection", self.current_frame)
                
                # Exit on 'q'
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        finally:
            self.cleanup()
    
    def cleanup(self):
        if self.cap:
            self.cap.release()
        if self.LOG_DATA and self.log_file:
            self.log_file.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    detector = MotionDetector()
    detector.run()