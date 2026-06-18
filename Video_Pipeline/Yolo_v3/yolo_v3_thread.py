# yolo_thread.py

import cv2
import numpy as np
import vart
import xir
import threading
import time

from utils.config import get_yolo_path


# ---------------------- INITIALIZE FACE CASCADE ---------------------- #
# Carica il mini-modello software leggero per il rilevamento frontale dei volti
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


# ---------------------- CLASS NAMES ---------------------- #
# YOLOv3 trained on VOC dataset can detect these 20 classes.
CLASS_NAMES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow", "diningtable",
    "dog", "horse", "motorbike", "person", "pottedplant",
    "sheep", "sofa", "train", "tvmonitor"
]


# ---------------------- PREPROCESS FUNCTION ---------------------- #
def preprocess(frame, input_shape):
    """
    Resize and normalize a webcam frame for the DPU model.
    """
    height, width = input_shape[1], input_shape[2]
    image = cv2.resize(frame, (width, height))
    image = image.astype(np.float32) / 255.0
    image = np.ascontiguousarray(image)
    return image


# ---------------------- SIGMOID FUNCTION ---------------------- #
def sigmoid(x):
    """
    Apply sigmoid activation function.
    """
    return 1 / (1 + np.exp(-x))


# ---------------------- POSTPROCESS FUNCTION ---------------------- #
def postprocess(output, frame, conf_threshold=0.6, nms_threshold=0.4):
    """
    Convert YOLOv3 DPU output into bounding boxes on the original frame.
    """
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

    indices = cv2.dnn.NMSBoxes(
        boxes,
        confidences,
        conf_threshold,
        nms_threshold
    )

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

                # --- PIPELINE IBRIDA: CERCA LA FACCIA DENTRO LA PERSONA ISOLATA ---
                person_roi = frame[y1:y2, x1:x2]
                
                face_found = False
                if person_roi.size > 0:
                    # Converti in scala di grigi per l'algoritmo di OpenCV
                    gray_person = cv2.cvtColor(person_roi, cv2.COLOR_BGR2GRAY)
                    
                    # Scansiona l'area per trovare lineamenti facciali
                    faces = face_cascade.detectMultiScale(
                        gray_person, 
                        scaleFactor=1.1, 
                        minNeighbors=5, 
                        minSize=(30, 30)
                    )
                    
                    # Se trova la faccia, mappa le coordinate sul frame completo
                    if len(faces) > 0:
                        fx, fy, fw, fh = faces[0] 
                        
                        face_x1 = x1 + fx
                        face_y1 = y1 + fy
                        face_x2 = face_x1 + fw
                        face_y2 = face_y1 + fh
                        
                        face_found = True
                        
                        # Aggiorna il bbox inviato alla dashboard puntando alla FACCIA
                        if conf > best_person_conf:
                            best_person_conf = conf
                            best_person_bbox = (face_x1, face_y1, face_x2, face_y2)
                        
                        # Disegna la scatola e la label specifica sul volto
                        cv2.rectangle(frame, (face_x1, face_y1), (face_x2, face_y2), (0, 255, 0), 2)
                        cv2.putText(
                            frame,
                            f"Face {conf:.2f}",
                            (face_x1, max(0, face_y1 - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 255, 0),
                            1
                        )
                
                # --- FALLBACK: SE NON TROVA FACCIA, USA AREA SPALLE ---
                if not face_found:
                    y_head_shoulders = y1 + int((y2 - y1) * 0.4)
                    center_x = (x1 + x2) // 2
                    w_fallback = int((y_head_shoulders - y1) * 1.2)
                    
                    fb_x1 = max(0, center_x - w_fallback // 2)
                    fb_x2 = min(W - 1, center_x + w_fallback // 2)
                    
                    if conf > best_person_conf:
                        best_person_conf = conf
                        best_person_bbox = (fb_x1, y1, fb_x2, y_head_shoulders)

                    cv2.rectangle(frame, (fb_x1, y1), (fb_x2, y_head_shoulders), (0, 255, 255), 2)
                    cv2.putText(
                        frame,
                        f"{label} {conf:.2f}",
                        (fb_x1, max(0, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 255),
                        1
                    )
            else:
                # Disegna le altre classi non-person rilevate dal modello VOC
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    f"{label} {conf:.2f}",
                    (x1, max(0, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1
                )

    return frame, person_detected, best_person_bbox, best_person_conf


# ------------------- YOLO DPU THREAD ------------------- #
class YoloDpuThread(threading.Thread):
    def __init__(self, device_id="camera0", camera_index=0, debug_window=False, roi_state=None):
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
            self.latest_result_ts = None
            self.latest_frame = None
            self.latest_person_bbox = None
            self.latest_person_conf = 0.0
            self.latest_person_bbox_ts = None

    def get_latest_result(self):
        with self.result_lock:
            return self.latest_result, self.latest_result_ts
        
    def get_latest_person_bbox(self):
        with self.result_lock:
            return (self.latest_person_bbox, self.latest_person_conf, self.latest_person_bbox_ts)

    def get_latest_frame(self):
        with self.result_lock:
            return self.latest_frame.copy() if self.latest_frame is not None else None

    def is_active(self):
        return self.active_event.is_set()

    def run(self):
        try:
            self.phase = "loading"
            model_path = get_yolo_path()
            graph = xir.Graph.deserialize(model_path)
            dpu_subgraphs = [sg for sg in graph.get_root_subgraph().toposort_child_subgraph() if sg.has_attr("device") and sg.get_attr("device").upper() == "DPU"]
            if not dpu_subgraphs: return
            self.runner = vart.Runner.create_runner(dpu_subgraphs[0], "run")
            input_tensors = self.runner.get_input_tensors()
            output_tensors = self.runner.get_output_tensors()
            input_shape = tuple(input_tensors[0].dims)
            self.phase = "idle"

            while not self.stop_event.is_set():
                if not self.active_event.wait(timeout=0.5):
                    self.phase = "idle"
                    continue
                if self.stop_event.is_set(): break
                self.phase = "opening_camera"
                self.cap = cv2.VideoCapture(self.camera_index)
                if not self.cap.isOpened():
                    self.active_event.clear(); self.phase = "idle"; continue
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                output_data = [np.empty(tuple(ot.dims), dtype=np.float32) for ot in output_tensors]
                self._last_person_seen_ts = time.monotonic()
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
                        self.deactivate(); break
                    with self.result_lock:
                        self.latest_result = person_detected
                        self.latest_result_ts = now
                        self.latest_frame = frame.copy()
                        if person_detected and person_bbox is not None:
                            self.latest_person_bbox = person_bbox
                            self.latest_person_conf = person_conf
                            self.latest_person_bbox_ts = now
                        else:
                            self.latest_person_bbox = None
                            self.latest_person_conf = 0.0
                            self.latest_person_bbox_ts = now
                    if self.debug_window:
                        cv2.imshow("YOLOv3 DPU", frame)
                        cv2.waitKey(1)
                if self.cap: self.cap.release(); self.cap = None
                if self.debug_window: cv2.destroyWindow("YOLOv3 DPU")
                with self.result_lock: self.latest_frame = None
                self.phase = "idle"
        finally:
            if self.cap: self.cap.release()
            self.runner = None
            self.phase = "stopped"

    def stop(self):
        self.stop_event.set()
        self.active_event.set()
        if self.is_alive(): self.join(timeout=3.0)
