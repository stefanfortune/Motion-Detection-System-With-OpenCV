import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading
from motion_detection import MotionDetector
import cv2
from PIL import Image, ImageTk

class MotionDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Advanced Motion Detection System")
        self.root.geometry("600x600")
        
        # Motion detector instance
        self.detector = None
        self.is_running = False
        self.preview_active = False
        
        # GUI variables
        self.threshold_var = tk.IntVar(value=5)
        self.min_area_var = tk.IntVar(value=100)
        self.persistence_var = tk.IntVar(value=5)
        self.log_data_var = tk.BooleanVar(value=False)
        self.show_debug_var = tk.BooleanVar(value=False)
        
        self.setup_ui()
        
    def setup_ui(self):
        # Create main frames
        control_frame = ttk.LabelFrame(self.root, text="Controls", padding=10)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        
        display_frame = ttk.Frame(self.root)
        display_frame.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH, padx=5, pady=5)
        
        # Video display
        self.video_label = ttk.Label(display_frame)
        self.video_label.pack(expand=True, fill=tk.BOTH)
        
        # Control widgets
        ttk.Label(control_frame, text="Sensitivity:").grid(row=0, column=0, sticky=tk.W)
        ttk.Scale(control_frame, from_=1, to=50, variable=self.threshold_var, 
                 command=lambda v: self.update_config('THRESHOLD', int(float(v)))).grid(row=0, column=1)
        
        ttk.Label(control_frame, text="Min Contour Area:").grid(row=1, column=0, sticky=tk.W)
        ttk.Scale(control_frame, from_=50, to=500, variable=self.min_area_var,
                 command=lambda v: self.update_config('MIN_CONTOUR_AREA', int(float(v)))).grid(row=1, column=1)
        
        ttk.Label(control_frame, text="Motion Persistence:").grid(row=2, column=0, sticky=tk.W)
        ttk.Scale(control_frame, from_=1, to=10, variable=self.persistence_var,
                 command=lambda v: self.update_config('MOTION_PERSISTENCE', int(float(v)))).grid(row=2, column=1)
        
        ttk.Checkbutton(control_frame, text="Enable Logging", variable=self.log_data_var,
                       command=lambda: self.update_config('LOG_DATA', self.log_data_var.get())).grid(row=3, column=0, columnspan=2, sticky=tk.W)
        
        ttk.Checkbutton(control_frame, text="Show Debug Windows", variable=self.show_debug_var,
                       command=lambda: self.update_config('SHOW_DEBUG_WINDOWS', self.show_debug_var.get())).grid(row=4, column=0, columnspan=2, sticky=tk.W)
        
        # Buttons
        ttk.Button(control_frame, text="Start Detection", command=self.start_detection).grid(row=5, column=0, columnspan=2, pady=10)
        ttk.Button(control_frame, text="Stop Detection", command=self.stop_detection).grid(row=6, column=0, columnspan=2)
        ttk.Button(control_frame, text="Toggle Preview", command=self.toggle_preview).grid(row=7, column=0, columnspan=2, pady=10)
        ttk.Button(control_frame, text="Exit", command=self.on_close).grid(row=8, column=0, columnspan=2)
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(control_frame, textvariable=self.status_var).grid(row=9, column=0, columnspan=2, pady=10)
        
    def update_config(self, param, value):
        if self.detector:
            setattr(self.detector, param, value)
            self.status_var.set(f"Parameter {param} updated to {value}")
        
    def start_detection(self):
        if not self.is_running:
            try:
                self.detector = MotionDetector()
                # Update detector with current GUI values
                self.detector.THRESHOLD = self.threshold_var.get()
                self.detector.MIN_CONTOUR_AREA = self.min_area_var.get()
                self.detector.MOTION_PERSISTENCE = self.persistence_var.get()
                self.detector.LOG_DATA = self.log_data_var.get()
                self.detector.SHOW_DEBUG_WINDOWS = self.show_debug_var.get()
                
                # Start in a separate thread
                self.thread = threading.Thread(target=self.run_detection, daemon=True)
                self.thread.start()
                self.is_running = True
                self.status_var.set("Detection running...")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to start detection: {str(e)}")
        
    def run_detection(self):
        self.detector.run()
        
    def stop_detection(self):
        if self.is_running and self.detector:
            self.detector.cleanup()
            self.is_running = False
            self.status_var.set("Detection stopped")
            
    def toggle_preview(self):
        self.preview_active = not self.preview_active
        if self.preview_active and self.is_running:
            self.update_preview()
            
    def update_preview(self):
        if self.preview_active and self.is_running and self.detector:
            frame = self.detector.current_frame
            if frame is not None:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame)
                imgtk = ImageTk.PhotoImage(image=img)
                self.video_label.imgtk = imgtk
                self.video_label.configure(image=imgtk)
            self.root.after(30, self.update_preview)
        else:
            # Clear the display
            self.video_label.configure(image='')
            
    def on_close(self):
        if self.is_running:
            self.stop_detection()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = MotionDetectionApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()