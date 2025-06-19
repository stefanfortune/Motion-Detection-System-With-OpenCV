import cv2
import numpy as np
import datetime
import csv
import os
from collections import deque
from Video_recorder import MotionVideoRecorder

# ======================
# CONFIGURATION
# ======================
THRESHOLD = 5                # Sensitivity (5-50)
MIN_CONTOUR_AREA = 100        # Minimum contour area to detect
BLUR_SIZE = (11, 11)          # Noise reduction kernel size
DILATE_ITERATIONS = 2         # Morphological processing strength
LOG_DATA = False               # Enable/disable event logging
SHOW_DEBUG_WINDOWS = False    # Show processing stages
MOTION_PERSISTENCE = 5        # Number of consecutive frames required to confirm motion
# ======================

# Initialize background subtractor
back_sub = cv2.createBackgroundSubtractorMOG2(
    history=500, 
    varThreshold=16, 
    detectShadows=False
)


# Setup logging
if LOG_DATA:
    log_dir = "motion_logs"
    os.makedirs(log_dir, exist_ok=True)
    log_file = open(f"{log_dir}/motion_events_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", "w")
    csv_writer = csv.writer(log_file)
    csv_writer.writerow(["Timestamp", "Event", "Contour_Area", "Position"])

# Initialize video capture
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Camera not accessible")
    exit()

# Get frame dimensions
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# After initializing camera, add:
recorder = MotionVideoRecorder(
    output_dir="motion_clips",
    pre_buffer_sec=1,
    clip_duration=5,
    fps=30,
    width= frame_width,
    height= frame_height
    
)

# For FPS calculation
prev_time = (datetime.datetime.now())
fps = 0
motion_history = deque(maxlen=MOTION_PERSISTENCE)  # Initialize motion history



while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Calculate FPS
    current_time = (datetime.datetime.now())
    fps = 1 / ((current_time - prev_time).total_seconds() + 1e-6)
    prev_time = current_time

    # Apply background subtraction
    fg_mask = back_sub.apply(frame)
    
    # Post-processing
    _, thresh = cv2.threshold(fg_mask, THRESHOLD, 255, cv2.THRESH_BINARY)
    blurred = cv2.GaussianBlur(thresh, BLUR_SIZE, 0)
    dilated = cv2.dilate(blurred, None, iterations=DILATE_ITERATIONS)
    eroded = cv2.erode(dilated, None, iterations=DILATE_ITERATIONS)
    
    # Find contours
    contours, _ = cv2.findContours(eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    current_motion = False
    status = "No motion"

    for contour in contours:
        if cv2.contourArea(contour) < MIN_CONTOUR_AREA:
            continue
            
        current_motion = True
        status = "Motion detected!"
        
        # Get bounding box and centroid
        x, y, w, h = cv2.boundingRect(contour)
        centroid = (x + w//2, y + h//2)
        
        # Draw bounding box and centroid
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.circle(frame, centroid, 5, (0, 0, 255), -1)
        
        # Log event
        if LOG_DATA:
            timestamp = current_time.strftime("%Y-%m-%d %H:%M:%S")
            event_id = f"EV_{timestamp.replace(':', '').replace(' ', '_')}"
            csv_writer.writerow([timestamp, event_id, cv2.contourArea(contour), centroid])
            
            # Save snapshot
            cv2.imwrite(f"{log_dir}/{event_id}.jpg", frame)
    
    # Update motion history and check persistence
    motion_history.append(current_motion)
    confirmed_motion = all(motion_history) if len(motion_history) == MOTION_PERSISTENCE else False
    
    # Update recorder with confirmed motion
    is_recording = recorder.update(frame, confirmed_motion)
    
    # Add visual feedback
    if is_recording:
        cv2.putText(frame, "REC", (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)


    # Display status info
    cv2.putText(frame, f"Status: {status}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, current_time.strftime("%Y-%m-%d %H:%M:%S"), (10, 90), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    

    # Show debug windows
    if SHOW_DEBUG_WINDOWS:
        cv2.imshow("Foreground Mask", fg_mask)
        cv2.imshow("Processed Mask", eroded)
    
    # Display output
    cv2.imshow("Advanced Motion Detection", frame)
    
    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
if LOG_DATA:
    log_file.close()
cv2.destroyAllWindows()