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

        # Divide lo schermo a metà in altezza: sopra il video unico, sotto la console
        self.top_h = self.height // 2
        self.console_y = self.top_h

        # Pannello video unificato (sfrutta l'intera larghezza disponibile)
        self.video_panel_x = 0
        self.video_panel_y = 0
        self.video_panel_w = self.width
        self.video_panel_h = self.top_h

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

        # Mantiene l'aspect ratio originale centrando l'immagine sul canvas nero
        scale = min(target_w / w, target_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)

        resized = cv2.resize(frame, (new_w, new_h))
        canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)

        x = (target_w - new_w) // 2
        y = (target_h - new_h) // 2

        canvas[y:y + new_h, x:x + new_w] = resized
        return canvas

    def get_thread_frame(self, thread, target_w, target_h):
        if thread is None or not hasattr(thread, "get_latest_frame"):
            return np.zeros((target_h, target_w, 3), dtype=np.uint8)

        frame = thread.get_latest_frame()
        return self.fit_frame(frame, target_w, target_h)

    def draw_panel_title(self, canvas, title, x, y, active, phase):
        status = "ACTIVE" if active else "IDLE"
        text = f"{title} | {status} | phase={phase}"

        # Disegna la barra di stato superiore a larghezza intera
        cv2.rectangle(canvas, (x, y), (x + self.width, y + 32), (30, 30, 30), -1)
        cv2.putText(
            canvas,
            text,
            (x + 12, y + 23),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
        )

    def draw_console(self, canvas):
        # Sfondo nero della console
        cv2.rectangle(
            canvas,
            (0, self.console_y),
            (self.width, self.height),
            (10, 10, 10),
            -1,
        )

        # Intestazione della console
        cv2.putText(
            canvas,
            "Console System Logs",
            (20, self.console_y + 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )

        y = self.console_y + 70

        with self.console_lock:
            lines_snapshot = list(self.console_lines)

        # Stampa le righe memorizzate nel deque
        for line in lines_snapshot:
            cv2.putText(
                canvas,
                line[:150],
                (20, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (210, 210, 210),
                1,
            )
            y += 22

    def render(self, yolo_thread=None):
        # Inizializza la matrice grafica totale
        dashboard = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        # Estrae lo stato di attivazione dei thread
        yolo_active = yolo_thread.is_active() if yolo_thread else False

        # Variabili dinamiche per gestire l'area video unificata
        active_title = "SISTEMA IN ATTESA"
        active_phase = "idle"
        active_frame = None

        # Logica di switch automatico nel pannello: vince chi è ACTIVE nel main loop
        if yolo_active:
            active_title = "LOCALIZZAZIONE SOGGETTO (YOLOv3 DPU)"
            active_phase = self.get_phase(yolo_thread)
            if yolo_thread and hasattr(yolo_thread, "get_latest_frame"):
                active_frame = yolo_thread.get_latest_frame()

        # Adatta l'immagine attiva a tutta larghezza nell'area superiore
        video_panel = self.fit_frame(active_frame, self.video_panel_w, self.video_panel_h)

        # Fallback grafico se non ci sono frame pronti nei thread
        if active_frame is None:
            cv2.putText(
                video_panel,
                "In attesa del flusso video hardware...",
                (self.width // 2 - 200, self.top_h // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
            )

        # Copia il pannello video elaborato nella metà superiore della dashboard
        dashboard[
            self.video_panel_y:self.video_panel_y + self.video_panel_h,
            self.video_panel_x:self.video_panel_x + self.video_panel_w,
        ] = video_panel

        # Disegna l'intestazione dinamica superiore unica
        current_status = "ACTIVE" if yolo_active else "IDLE"
        self.draw_panel_title(
            dashboard,
            active_title,
            self.video_panel_x,
            self.video_panel_y,
            current_status,
            active_phase,
        )

        # Disegna la sezione dei log (metà inferiore)
        self.draw_console(dashboard)

        # Stringa informativa fissa in fondo allo schermo
        cv2.putText(
            dashboard,
            "Premere 'q' all'interno della finestra per uscire",
            (20, self.height - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (150, 150, 150),
            1,
        )

        # Mostra la finestra finale unificata
        cv2.imshow(self.window_name, dashboard)

    def wait_key(self, delay_ms=1):
        return cv2.waitKey(delay_ms) & 0xFF

    def close(self):
        cv2.destroyWindow(self.window_name)import time
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

        self.top_h = self.height // 2
        self.console_y = self.top_h

        self.yolo_panel_x = 0
        self.yolo_panel_y = 0
        self.yolo_panel_w = self.width // 2
        self.yolo_panel_h = self.top_h

        self.resnet_panel_x = self.width // 2
        self.resnet_panel_y = 0
        self.resnet_panel_w = self.width // 2
        self.resnet_panel_h = self.top_h

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
        new_w = int(w * scale)
        new_h = int(h * scale)

        resized = cv2.resize(frame, (new_w, new_h))

        canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)

        x = (target_w - new_w) // 2
        y = (target_h - new_h) // 2

        canvas[y:y + new_h, x:x + new_w] = resized

        return canvas

    def get_thread_frame(self, thread, target_w, target_h):
        if thread is None or not hasattr(thread, "get_latest_frame"):
            return np.zeros((target_h, target_w, 3), dtype=np.uint8)

        frame = thread.get_latest_frame()
        return self.fit_frame(frame, target_w, target_h)

    def draw_panel_title(self, canvas, title, x, y, active, phase):
        status = "ACTIVE" if active else "IDLE"
        text = f"{title} | {status} | phase={phase}"

        cv2.rectangle(canvas, (x, y), (x + 640, y + 32), (30, 30, 30), -1)
        cv2.putText(
            canvas,
            text,
            (x + 12, y + 23),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
        )

    def draw_console(self, canvas):
        cv2.rectangle(
            canvas,
            (0, self.console_y),
            (self.width, self.height),
            (10, 10, 10),
            -1,
        )

        cv2.putText(
            canvas,
            "Console",
            (20, self.console_y + 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )

        y = self.console_y + 70

        with self.console_lock:
            lines_snapshot = list(self.console_lines)

        for line in lines_snapshot:
            cv2.putText(
                canvas,
                line[:150],
                (20, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (210, 210, 210),
                1,
            )
            y += 22

    def render(self, yolo_thread=None, resnet_thread=None):
        dashboard = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        yolo_frame = self.get_thread_frame(
            yolo_thread,
            self.yolo_panel_w,
            self.yolo_panel_h,
        )

        resnet_frame = self.get_thread_frame(
            resnet_thread,
            self.resnet_panel_w,
            self.resnet_panel_h,
        )

        dashboard[
            self.yolo_panel_y:self.yolo_panel_y + self.yolo_panel_h,
            self.yolo_panel_x:self.yolo_panel_x + self.yolo_panel_w,
        ] = yolo_frame

        dashboard[
            self.resnet_panel_y:self.resnet_panel_y + self.resnet_panel_h,
            self.resnet_panel_x:self.resnet_panel_x + self.resnet_panel_w,
        ] = resnet_frame

        yolo_active = yolo_thread.is_active() if yolo_thread else False
        resnet_active = resnet_thread.is_active() if resnet_thread else False

        self.draw_panel_title(
            dashboard,
            "YOLOv3 DPU",
            self.yolo_panel_x,
            self.yolo_panel_y,
            yolo_active,
            self.get_phase(yolo_thread),
        )

        self.draw_panel_title(
            dashboard,
            "ResNet DPU",
            self.resnet_panel_x,
            self.resnet_panel_y,
            resnet_active,
            self.get_phase(resnet_thread),
        )

        cv2.line(
            dashboard,
            (self.width // 2, 0),
            (self.width // 2, self.top_h),
            (80, 80, 80),
            2,
        )

        self.draw_console(dashboard)

        cv2.putText(
            dashboard,
            "Press q to quit",
            (20, self.height - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
        )

        cv2.imshow(self.window_name, dashboard)

    def wait_key(self, delay_ms=1):
        return cv2.waitKey(delay_ms) & 0xFF

    def close(self):
        cv2.destroyWindow(self.window_name)
