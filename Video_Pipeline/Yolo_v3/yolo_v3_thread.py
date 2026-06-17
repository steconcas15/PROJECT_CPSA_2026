# yolo_thread.py

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
