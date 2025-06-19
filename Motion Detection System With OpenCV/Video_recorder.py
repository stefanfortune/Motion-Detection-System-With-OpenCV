import cv2
import os
import datetime
import numpy as np
from collections import deque


class MotionVideoRecorder:
    def __init__(self, width, height, output_dir="motion_clips", pre_buffer_sec=1, clip_duration=5, fps=30,  ):
        
        self.output_dir = output_dir
        self.pre_buffer_sec = pre_buffer_sec
        self.clip_duration = clip_duration
        self.fps = fps 
        self.frame_buffer = deque(maxlen=int(fps * pre_buffer_sec))
        self.recording = False
        self.frames_remaining = 0
        self.video_writer = None
        self.width = width
        self.height = height 
        self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        os.makedirs(output_dir, exist_ok=True)
        
    def start_recording(self):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.output_dir}/motion_{timestamp}.mp4"
        self.video_writer = cv2.VideoWriter(
            filename,
            self.fourcc,
            self.fps,
            (self.width, self.height)
            
        )
        # Save pre-trigger frames
        for frame in self.frame_buffer:
            self.video_writer.write(frame)
        self.frames_remaining = int(self.fps * self.clip_duration)

    def update(self, frame, motion_detected):
        # Always buffer frames
        self.frame_buffer.append(frame.copy())
        
        # Handle recording state
        if motion_detected:
            if not self.recording:
                self.recording = True
                self.start_recording()
            else:
                self.frames_remaining = int(self.fps * self.clip_duration)  # Extend recording
        
        # Write frames if recording
        if self.recording:
            self.video_writer.write(frame)
            self.frames_remaining -= 1
            if self.frames_remaining <= 0:
                self.stop_recording()
                
        return self.recording, motion_detected

    def stop_recording(self):
        if self.recording:
            self.video_writer.release()
            self.recording = False

    def __del__(self):
        self.stop_recording()
    