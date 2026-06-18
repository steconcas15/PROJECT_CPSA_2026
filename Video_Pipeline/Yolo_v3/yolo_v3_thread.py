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
    H, W = frame.shape[:2]                            # Get the height (H) and width (W) of the original input image

    anchors = [(116, 90), (156, 198), (373, 326)]     # Predefined YOLOv3 anchor box dimensions (width, height)
    grid_size = 13                                    # Feature map size for the specific YOLO head (e.g., 13x13)
    num_anchors = 3                                   # Number of anchor boxes per grid cell
    num_classes = 20 

    # Reshape the flat DPU output into a structured format [grid, grid, anchors, (x, y, w, h, obj, classes)]
    output = output.reshape(grid_size, grid_size, num_anchors, 5 + num_classes)

    boxes = []
    confidences = []
    class_ids = []

    for row in range(grid_size):                                             # Iterate through every cell in the grid
        for col in range(grid_size):                                         
            for a in range(num_anchors):                                     # Iterate through each anchor box in the current grid cell

                # NOTE: The variables tx, ty, tw, th are the raw outputs (offsets) from the neural network.
                # They are not final coordinates, but "adjustment instructions":
                # - tx, ty: Predict the center displacement relative to the grid cell (using sigmoid to keep it inside the cell).
                # - tw, th: Predict the scaling factors (via exponentiation) to stretch/shrink the anchor box.
                # These allow the network to refine the predefined 'anchor' shapes into a precise bounding box.
                
                tx, ty, tw, th, obj_score = output[row, col, a, :5]
                obj_score = sigmoid(obj_score)                               # Apply sigmoid to normalize objectness score to [0, 1]

                if obj_score < conf_threshold:                               # Skip processing if objectness is below threshold
                    continue

                # --- BOUNDING BOX CALCULATION (Refining the Anchor Boxes) ---
                # We calculate the final box by taking a 'base' shape (anchor)
                # and applying the network's 'adjustment instructions' (t variables).
                
                # bx, by: Calculate the center of the object (0.0 to 1.0 relative to the whole frame)
                # We use sigmoid to ensure the center stays within the current grid cell.
                
                bx = (sigmoid(tx) + col) / grid_size
                by = (sigmoid(ty) + row) / grid_size

                # bw, bh: Calculate the final width and height
                # We use np.exp() to stretch/shrink the anchor box (anchors[a]) based on the network's prediction.
                # The '/ 416' factor normalizes the values based on the model's training resolution.
                
                bw = np.exp(tw) * anchors[a][0] / 416
                bh = np.exp(th) * anchors[a][1] / 416

                # Convert relative coordinates (0 to 1) to absolute pixel values on the frame
                # x, y: The top-left corner coordinates of the rectangle in pixels.
                # w, h: The width and height of the rectangle in pixels.
                
                x = int((bx - bw / 2) * W)
                y = int((by - bh / 2) * H)
                w = int(bw * W)
                h = int(bh * H)

                class_probs = sigmoid(output[row, col, a, 5:])            # Apply sigmoid to class probabilities
                class_id = int(np.argmax(class_probs))                    # Determine the class with the highest probability
                confidence = float(obj_score * class_probs[class_id])     # Calculate final confidence score (objectness * class probability)    

                if confidence > conf_threshold:                           # Store if the confidence meets the threshold
                    boxes.append([x, y, w, h])
                    confidences.append(confidence)
                    class_ids.append(class_id)

    # Perform Non-Maximum Suppression (NMS) to eliminate duplicate/overlapping boxes
    
    indices = cv2.dnn.NMSBoxes(
        boxes,
        confidences,
        conf_threshold,
        nms_threshold
    )

    person_detected = False
    best_person_bbox = None
    best_person_conf = 0.0

    if len(indices) > 0:                            # Process if any detections survived NMS
        for i in indices.flatten():
            x, y, w, h = boxes[i]                   # Get current box coordinates and metadata
            label = CLASS_NAMES[class_ids[i]]
            conf = float(confidences[i])

            x1 = max(0, int(x))                     # Ensure box coordinates are within frame boundaries
            y1 = max(0, int(y))
            x2 = min(W - 1, int(x + w))
            y2 = min(H - 1, int(y + h))

            if x2 <= x1 or y2 <= y1:                # Skip if box has zero or negative area
                continue

            if label == "person":                   # Check if current detection is a "person"
                person_detected = True

                # --- HYBRID PIPELINE: DETECT FACE WITHIN THE PERSON ROI ---
                person_roi = frame[y1:y2, x1:x2]
                
                face_found = False
                if person_roi.size > 0:
                    # Convert region of interest (ROI) to grayscale for cascade detector
                    gray_person = cv2.cvtColor(person_roi, cv2.COLOR_BGR2GRAY)
                    
                   # Search for faces using Haar Cascade or similar method
                    faces = face_cascade.detectMultiScale(
                        gray_person, 
                        scaleFactor=1.1, 
                        minNeighbors=5, 
                        minSize=(30, 30)
                    )
                    
                    # If a face is found, update ROI coordinates to the face's location
                    if len(faces) > 0:
                        fx, fy, fw, fh = faces[0] 

                        # Map local face coordinates back to original frame space
                        
                        face_x1 = x1 + fx
                        face_y1 = y1 + fy
                        face_x2 = face_x1 + fw
                        face_y2 = face_y1 + fh
                        
                        face_found = True
                        
                        # Track the best person detection based on confidence
                        if conf > best_person_conf:
                            best_person_conf = conf
                            best_person_bbox = (face_x1, face_y1, face_x2, face_y2)
                        
                        # Draw green rectangle around the detected face
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
                
                --- FALLBACK: IF NO FACE, ESTIMATE HEAD-SHOULDER AREA ---
                if not face_found:
                    
                    # Estimate the vertical limit of the head-shoulder region (top 40% of the person's box)
                    # y1 is the top edge, y2 is the bottom edge. We take 40% of that height.
            
                    y_head_shoulders = y1 + int((y2 - y1) * 0.4)

                    # Calculate the horizontal center of the person's box
                    center_x = (x1 + x2) // 2

                    # Define a width for our fallback box (1.2 times the height of the head region)
                    # This provides a reasonable aspect ratio for a head-shoulder crop.         
                    w_fallback = int((y_head_shoulders - y1) * 1.2)

                    # Calculate new X coordinates, ensuring they stay within frame boundaries (0 to W-1)
                    fb_x1 = max(0, center_x - w_fallback // 2)
                    fb_x2 = min(W - 1, center_x + w_fallback // 2)

                    # Update the best person tracker if this detection has the highest confidence
                    if conf > best_person_conf:
                        best_person_conf = conf
                        best_person_bbox = (fb_x1, y1, fb_x2, y_head_shoulders)

                    # Draw a rectangle for the fallback region (Green: BGR 0, 255, 0)
                    cv2.rectangle(frame, (fb_x1, y1), (fb_x2, y_head_shoulders), (0, 255, 0), 2)

                    # Add a text label to indicate this is a fallback 'Face' estimation
                    cv2.putText(
                        frame,
                        f"Face {conf:.2f}",
                        (fb_x1, max(0, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        1
                    )
            else:
                # --- DEFAULT CASE: DRAW RECTANGLES FOR NON-PERSON CLASSES ---
                # For objects other than people, draw a standard green rectangle
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
