# yolo_thread.py

import cv2
import numpy as np
import vart
import xir
import threading
import time

from utils.config import get_yolo_path


# ---------------------- INITIALIZE FACE CASCADE ---------------------- #
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


# ---------------------- CLASS NAMES ---------------------- #
CLASS_NAMES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow", "diningtable",
    "dog", "horse", "motorbike", "person", "pottedplant",
    "sheep", "sofa", "train", "tvmonitor"
]


# ---------------------- PREPROCESS FUNCTION ---------------------- #
def preprocess(frame, input_shape):
    height, width = input_shape[1], input_shape[2]
    image = cv2.resize(frame, (width, height))
    image = image.astype(np.float32) / 255.0
    image = np.ascontiguousarray(image)
    return image


# ---------------------- SIGMOID FUNCTION ---------------------- #
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# ---------------------- POSTPROCESS FUNCTION ---------------------- #
def postprocess(output, frame, conf_threshold=0.6, nms_threshold=0.4):
    H, W = frame.shape[:2]

    anchors = [(116, 90), (156, 198), (373, 326)]
    grid_size = 13
    num_anchors = 3
    num_classes = 20

    output = output.reshape(grid_size, grid_size, num_anchors, 5 + num_classes)

    boxes = []
    confidences = []
    class_ids = []

    for row in range(grid_size):
        for col in range(grid_size):
            for a in range(num_anchors):
                tx, ty, tw, th, obj_score = output[row, col, a, :5]
                obj_score = sigmoid(obj_score)

                if obj_score < conf_threshold:
                    continue

                bx = (sigmoid(tx) + col) / grid_size
                by = (sigmoid(ty) + row) / grid_size
                bw = np.exp(tw) * anchors[a][0] / 416
                bh = np.exp(th) * anchors[a][1] / 416

                x = int((bx - bw / 2) * W)
                y = int((by - bh / 2) * H)
                w = int(bw * W)
                h = int(bh * H)

                class_probs = sigmoid(output[row, col, a, 5:])
                class_id = int(np.argmax(class_probs))
                confidence = float(obj_score * class_probs[class_id])

                if confidence > conf_threshold:
                    boxes.append([x, y, w, h])
                    confidences.append(confidence)
                    class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

    person_detected = False
    best_person_bbox = None
    best_person_conf = 0.0

    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h = boxes[i]
            label = CLASS_NAMES[class_ids[i]]
            conf = float(confidences[i])

            x1 = max(0, int(x))
            y1 = max(0, int(y))
            x2 = min(W - 1, int(x + w))
            y2 = min(H - 1, int(y + h))

            if x2 <= x1 or y2 <= y1:
                continue

            if label == "person":
                person_detected = True
                
                # --- LOGICA IBRIDA ---
                person_roi = frame[y1:y2, x1:x2]
                face_found = False
                
                if person_roi.size > 0:
                    gray_person = cv2.cvtColor(person_roi, cv2.COLOR_BGR2GRAY)
                    faces = face_cascade.detectMultiScale(gray_person, 1.1, 5, minSize=(30, 30))
                    
                    if len(faces) > 0:
                        fx, fy, fw, fh = faces[0]
                        best_person_bbox = (x1 + fx, y1 + fy, x1 + fx + fw, y1 + fy + fh)
                        best_person_conf = conf
                        face_found = True
                        cv2.rectangle(frame, (best_person_bbox[0], best_person_bbox[1]), 
                                      (best_person_bbox[2], best_person_bbox[3]), (0, 255, 0), 2)
                
                # --- FALLBACK PROFILO (Testa e Spalle) ---
                if not face_found:
                    # Prende solo il 40% superiore del corpo, centrato
                    y_head_shoulders = y1 + int((y2 - y1) * 0.4)
                    center_x = (x1 + x2) // 2
                    w_fallback = int((y_head_shoulders - y1) * 1.2) # Rende il box meno rettangolare
                    best_person_bbox = (max(0, center_x - w_fallback//2), y1, 
                                        min(W-1, center_x + w_fallback//2), y_head_shoulders)
                    best_person_conf = conf
                    cv2.rectangle(frame, (best_person_bbox[0], best_person_bbox[1]), 
                                  (best_person_bbox[2], best_person_bbox[3]), (0, 255, 255), 2) # Giallo per fallback

                cv2.putText(frame, f"Person {conf:.2f}", (x1, max(0, y1 - 5)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            else:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, f"{label} {conf:.2f}", (x1, max(0, y1 - 5)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    return frame, person_detected, best_person_bbox, best_person_conf


# ------------------- YOLO DPU THREAD ------------------- #
class YoloDpuThread(threading.Thread):
    def __init__(self, device_id: str = "camera0", camera_index: int = 0, debug_window: bool = False, roi_state=None):
        super().__init__(daemon=True)
        self.device_id = device_id
        self.camera_index = camera_index
        self.debug_window = debug_window
        self.stop_event = threading.Event()
        self.active_event = threading.Event()
        self.cap = None
        self.runner = None
        self.result_lock = threading.Lock()
        self.latest_result = None
        self.latest_result_ts = None
        self.latest_frame = None
        self.latest_person_bbox = None
        self.latest_person_conf = 0.0
        self.latest_person_bbox_ts = None
        self.no_person_timeout_sec = 5.0
        self._last_person_seen_ts = None
        self.roi_state = roi_state
        self.phase = "idle"

    def activate(self):
        self.active_event.set()

    def deactivate(self):
        self.active_event.clear()
        self._last_person_seen_ts = None
        with self.result_lock:
            self.latest_result = None
            self.latest_frame = None
            self.latest_person_bbox = None

    def run(self):
        try:
            self.phase = "loading"
            model_path = get_yolo_path()
            graph = xir.Graph.deserialize(model_path)
            dpu_subgraphs = [sg for sg in graph.get_root_subgraph().toposort_child_subgraph() 
                             if sg.has_attr("device") and sg.get_attr("device").upper() == "DPU"]
            self.runner = vart.Runner.create_runner(dpu_subgraphs[0], "run")
            input_shape = tuple(self.runner.get_input_tensors()[0].dims)
            self.phase = "idle"

            while not self.stop_event.is_set():
                if not self.active_event.wait(timeout=0.5):
                    self.phase = "idle"
                    continue
                
                self.phase = "opening_camera"
                self.cap = cv2.VideoCapture(self.camera_index)
                if not self.cap.isOpened():
                    self.active_event.clear()
                    continue
                
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                
                output_tensors = self.runner.get_output_tensors()
                output_data = [np.empty(tuple(ot.dims), dtype=np.float32) for ot in output_tensors]
                
                self.phase = "running"
                while self.active_event.is_set() and not self.stop_event.is_set():
                    ret, frame = self.cap.read()
                    if not ret: break
                    
                    frame = cv2.flip(frame, 1)
                    img_input = preprocess(frame, input_shape)
                    job_id = self.runner.execute_async([img_input], output_data)
                    self.runner.wait(job_id)

                    frame, person_detected, person_bbox, person_conf = postprocess(output_data[0], frame)

                    if self.roi_state is not None and person_detected and person_bbox is not None:
                        self.roi_state.update_from_yolo(bbox_xyxy=person_bbox, confidence=person_conf)

                    now = time.monotonic()
                    if person_detected: self._last_person_seen_ts = now
                    elif self._last_person_seen_ts and now - self._last_person_seen_ts >= self.no_person_timeout_sec:
                        self.deactivate()
                        break

                    with self.result_lock:
                        self.latest_result = person_detected
                        self.latest_frame = frame.copy()
                        self.latest_person_bbox = person_bbox
                        self.latest_person_conf = person_conf

                    if self.debug_window:
                        cv2.imshow("YOLOv3 DPU", frame)
                        cv2.waitKey(1)
                
                if self.cap: self.cap.release()
        finally:
            if self.cap: self.cap.release()
            self.runner = None
            self.phase = "stopped"

    def stop(self):
        self.stop_event.set()
        self.active_event.set()
        if self.is_alive(): self.join(timeout=3.0)# yolo_thread.py

import cv2
import numpy as np
import vart
import xir
import threading
import time
from utils.config import get_yolo_path

# ---------------------- CONFIG ---------------------- #
# Aumenta questo valore per dare più "aria" attorno alla faccia/testa
PADDING = 20 

# ---------------------- INITIALIZE FACE CASCADE ---------------------- #
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# ---------------------- CLASS NAMES ---------------------- #
CLASS_NAMES = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

# ---------------------- HELPERS ---------------------- #
def preprocess(frame, input_shape):
    height, width = input_shape[1], input_shape[2]
    image = cv2.resize(frame, (width, height)).astype(np.float32) / 255.0
    return np.ascontiguousarray(image)

def sigmoid(x): return 1 / (1 + np.exp(-x))

# ---------------------- POSTPROCESS FUNCTION ---------------------- #
def postprocess(output, frame, conf_threshold=0.6, nms_threshold=0.4):
    H, W = frame.shape[:2]
    anchors = [(116, 90), (156, 198), (373, 326)]
    grid_size, num_anchors, num_classes = 13, 3, 20
    output = output.reshape(grid_size, grid_size, num_anchors, 5 + num_classes)

    boxes, confidences, class_ids = [], [], []
    for row in range(grid_size):
        for col in range(grid_size):
            for a in range(num_anchors):
                tx, ty, tw, th, obj_score = output[row, col, a, :5]
                obj_score = sigmoid(obj_score)
                if obj_score < conf_threshold: continue
                bx, by = (sigmoid(tx) + col) / grid_size, (sigmoid(ty) + row) / grid_size
                bw, bh = np.exp(tw) * anchors[a][0] / 416, np.exp(th) * anchors[a][1] / 416
                boxes.append([int((bx - bw/2)*W), int((by - bh/2)*H), int(bw*W), int(bh*H)])
                class_probs = sigmoid(output[row, col, a, 5:])
                class_id = int(np.argmax(class_probs))
                confidences.append(float(obj_score * class_probs[class_id]))
                class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

    best_person_bbox = None
    best_person_conf = 0.0

    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h = boxes[i]
            label = CLASS_NAMES[class_ids[i]]
            conf = float(confidences[i])
            if label == "person":
                x1, y1, x2, y2 = max(0, int(x)), max(0, int(y)), min(W-1, int(x+w)), min(H-1, int(y+h))
                
                # --- LOGICA IBRIDA ---
                person_roi = frame[y1:y2, x1:x2]
                face_found = False
                
                if person_roi.size > 0:
                    faces = face_cascade.detectMultiScale(cv2.cvtColor(person_roi, cv2.COLOR_BGR2GRAY), 1.1, 5)
                    if len(faces) > 0:
                        fx, fy, fw, fh = faces[0]
                        # Faccia rilevata con padding
                        best_person_bbox = (max(0, x1+fx-PADDING), max(0, y1+fy-PADDING), 
                                            min(W-1, x1+fx+fw+PADDING), min(H-1, y1+fy+fh+PADDING))
                        best_person_conf = conf
                        face_found = True
                
                if not face_found:
                    # Fallback testa e spalle proporzionato
                    y_head_shoulders = y1 + int((y2 - y1) * 0.4)
                    center_x = (x1 + x2) // 2
                    w_fallback = int((y_head_shoulders - y1) * 1.2) # Proporzione ritratto 1.2
                    best_person_bbox = (max(0, center_x - w_fallback//2 - PADDING), y1, 
                                        min(W-1, center_x + w_fallback//2 + PADDING), y_head_shoulders + PADDING)
                    best_person_conf = conf

                # Disegna il box finale
                bx1, by1, bx2, by2 = best_person_bbox
                cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0, 255, 0), 2)
                break 

    return frame, (best_person_bbox is not None), best_person_bbox, best_person_conf

# ------------------- YOLO DPU THREAD ------------------- #
class YoloDpuThread(threading.Thread):
    def __init__(self, camera_index=0, debug_window=False, roi_state=None):
        super().__init__(daemon=True)
        self.camera_index, self.debug_window, self.roi_state = camera_index, debug_window, roi_state
        self.stop_event, self.active_event = threading.Event(), threading.Event()
        self.result_lock = threading.Lock()
        self.latest_result, self.latest_frame = None, None
        self.latest_person_bbox, self.latest_person_conf = None, 0.0

    def activate(self): self.active_event.set()
    def deactivate(self): self.active_event.clear()

    def run(self):
        model_path = get_yolo_path()
        graph = xir.Graph.deserialize(model_path)
        runner = vart.Runner.create_runner([sg for sg in graph.get_root_subgraph().toposort_child_subgraph() 
                                            if sg.has_attr("device") and sg.get_attr("device").upper() == "DPU"][0], "run")
        input_shape = tuple(runner.get_input_tensors()[0].dims)

        while not self.stop_event.is_set():
            if not self.active_event.wait(timeout=0.5): continue
            cap = cv2.VideoCapture(self.camera_index)
            while self.active_event.is_set() and not self.stop_event.is_set():
                ret, frame = cap.read()
                if not ret: break
                frame = cv2.flip(frame, 1)
                
                output_data = [np.empty(tuple(ot.dims), dtype=np.float32) for ot in runner.get_output_tensors()]
                job_id = runner.execute_async([preprocess(frame, input_shape)], output_data)
                runner.wait(job_id)
                
                frame, detected, bbox, conf = postprocess(output_data[0], frame)
                if self.roi_state and detected: self.roi_state.update_from_yolo(bbox, conf)
                
                with self.result_lock:
                    self.latest_result, self.latest_frame, self.latest_person_bbox = detected, frame.copy(), bbox
            cap.release()

    def stop(self):
        self.stop_event.set()
        self.active_event.set()
        if self.is_alive(): self.join(timeout=3.0)
