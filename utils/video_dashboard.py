import time
from collections import deque
import cv2
import numpy as np
import threading

_dashboard_console = None

def register_dashboard_console(dashboard):
    global _dashboard_console
    _dashboard_console = dashboard

def unregister_dashboard_console():
    global _dashboard_console
    _dashboard_console = None

def dashboard_console_log(message):
    if _dashboard_console is not None:
        _dashboard_console.add_line(message)

class VideoDashboard:
    def __init__(
        self,
        window_name="CPSA Dashboard",
        width=1280,
        height=720,
        fullscreen=True,
        console_max_lines=13,
    ):
        self.window_name = window_name
        self.width = width
        self.height = height
        self.fullscreen = fullscreen

        # Layout: Video a tutto schermo sopra, console in basso
        self.console_height = 250
        self.console_y = self.height - self.console_height
        self.video_panel_w = self.width
        self.video_panel_h = self.console_y

        self.console_lines = deque(maxlen=console_max_lines)
        self.console_lock = threading.Lock()

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)

        if self.fullscreen:
            cv2.setWindowProperty(
                self.window_name,
                cv2.WND_PROP_FULLSCREEN,
                cv2.WINDOW_FULLSCREEN,
            )
        else:
            cv2.resizeWindow(self.window_name, self.width, self.height)

    def add_line(self, message):
        with self.console_lock:
            self.console_lines.append(str(message))

    def log(self, message):
        print(message)
        self.add_line(message)

    def get_phase(self, thread):
        return getattr(thread, "phase", "unknown")

    def fit_frame(self, frame, target_w, target_h):
        if frame is None:
            return np.zeros((target_h, target_w, 3), dtype=np.uint8)
        h, w = frame.shape[:2]
        if h <= 0 or w <= 0:
            return np.zeros((target_h, target_w, 3), dtype=np.uint8)

        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(frame, (new_w, new_h))
        canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        x, y = (target_w - new_w) // 2, (target_h - new_h) // 2
        canvas[y:y + new_h, x:x + new_w] = resized
        return canvas

    def get_thread_frame(self, thread, target_w, target_h):
        if thread is None or not hasattr(thread, "get_latest_frame"):
            return np.zeros((target_h, target_w, 3), dtype=np.uint8)
        return self.fit_frame(thread.get_latest_frame(), target_w, target_h)

    def draw_panel_title(self, canvas, title, active, phase):
        status = "ACTIVE" if active else "IDLE"
        text = f"{title} | {status} | phase={phase}"
        cv2.rectangle(canvas, (0, 0), (self.width, 40), (30, 30, 30), -1)
        cv2.putText(canvas, text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    def draw_console(self, canvas):
        cv2.rectangle(canvas, (0, self.console_y), (self.width, self.height), (10, 10, 10), -1)
        cv2.putText(canvas, "Console", (20, self.console_y + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y = self.console_y + 70
        with self.console_lock:
            lines_snapshot = list(self.console_lines)
        for line in lines_snapshot:
            cv2.putText(canvas, line[:150], (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (210, 210, 210), 1)
            y += 25

    def render(self, yolo_thread=None):
        dashboard = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        # Rendering Video
        yolo_frame = self.get_thread_frame(yolo_thread, self.video_panel_w, self.video_panel_h)
        dashboard[0:self.video_panel_h, 0:self.video_panel_w] = yolo_frame

        # UI Elements
        yolo_active = yolo_thread.is_active() if yolo_thread else False
        self.draw_panel_title(dashboard, "YOLOv3 DPU", yolo_active, self.get_phase(yolo_thread))
        self.draw_console(dashboard)

        # Quit instruction
        cv2.putText(dashboard, "Press 'q' to quit", (self.width - 160, self.height - 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv2.imshow(self.window_name, dashboard)

    def wait_key(self, delay_ms=1):
        return cv2.waitKey(delay_ms) & 0xFF

    def close(self):
        cv2.destroyWindow(self.window_name)
