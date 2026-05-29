# CPSA2026/VIDEO_Pipeline/Yolo_v3u/yolo_v3u_thread.py

import cv2
import numpy as np
import vart
import xir
import threading
import time

from utils.config import get_yolo_path


# ---------------------- CLASS NAMES ---------------------- #
# Modello custom ottimizzato solo per il rilevamento di volti
CLASS_NAMES = ["face"]


# ---------------------- PREPROCESS FUNCTION ---------------------- #
def preprocess(frame, input_shape):
    """
    Resizes, normalizes, and prepares a camera frame for DPU model execution.

    Steps:
    1. Extracts target dimensions from the DPU model's expected input shape.
    2. Geometrically resizes the frame to match the network requirements.
    3. Normalizes pixel values from [0, 255] integers to [0.0, 1.0] float32.
    4. Forces memory layout to be contiguous for optimized hardware DMA transfer.

    Args:
        frame (numpy.ndarray): Raw BGR image captured from the webcam.
        input_shape (tuple): Expected model input shape from DPU metadata (Batch, H, W, C).

    Returns:
        numpy.ndarray: Preprocessed float32 image ready for DPU inference.
    """
    
    height, width = input_shape[1], input_shape[2]
    image = cv2.resize(frame, (width, height))
    image = image.astype(np.float32) / 255.0
    image = np.ascontiguousarray(image)
    return image


# ---------------------- SIGMOID FUNCTION ---------------------- #
def sigmoid(x):
    """
    Applies the Sigmoid activation function to map raw DPU output scores.

    The DPU outputs raw unbounded values (logits) ranging from -inf to +inf. 
    This function scales them mathematically into a [0.0, 1.0] probability range.

    Args:
        x (numpy.ndarray or float): Raw logit value(s) from the model output.

    Returns:
        numpy.ndarray or float: Activated values representing confidence probabilities.
    """
    return 1 / (1 + np.exp(-x))


# ---------------------- POSTPROCESS FUNCTION ---------------------- #
def postprocess(output, frame, conf_threshold=0.5, nms_threshold=0.4):
    """
    Converte l'output del DPU Face Detector in bounding box sul frame originale
    applicando il padding del 30% usato in fase di test.

    Returns:
        tuple:
            frame: Frame con la bounding box disegnata.
            face_detected: True se è stato rilevato almeno un volto.
            best_face_bbox: Coordinate (xmin, ymin, xmax, ymax) con il padding del 30%.
            best_face_conf: Confidenza del volto migliore rilevato.
    """
    H, W = frame.shape[:2]

    # The image is divided into a 13x13 macro-cell grid
    grid_size = 13

    # Defines the number of predefined bounding box shapes (anchors) evaluated per grid cell.
    # YOLOv3u tests 3 different aspect ratios at this scale to capture small, medium, and large faces.
    num_anchors = 3
    
    num_classes = 1

   # =========================================================================================
    # RESHAPE THE RAW OUTPUT INTO A STRUCTURED 4D GEOMETRIC MATRIX
    # =========================================================================================
    # The raw model output is a flat 1D array of continuous numbers. To make sense of it,
    # we bend it into a 4D tensor with the following dimensions:
    # 
    # Shape: (GridY, GridX, Anchors, Properties)
    # 
    # 1. GridY & GridX (They are coordinates): The image is divided into a virtual 13x13 chessboard. 
    #    Each (Y, X) coordinate points to a specific geographic cell in the image.
    # 2. Anchors (3): Inside each single cell, the network tests 3 pre-defined bounding box 
    #    shapes (anchor templates) to detect objects of different sizes/aspect ratios.
    # 3. Properties (5 + num_classes): For each anchor in each cell, the model predicts 
    #    a specific vector of data. If num_classes = 20, this vector contains 25 elements.
    # 
    # Properties Array Layout Breakdown:
    # -------------------------------------------------------------------------------------
    # Index [0, 1]   -> tx, ty            : Box center offset relative to the top-left corner of the current grid cell.
    # Index [2, 3]   -> tw, th            : Box width and height scale factors (modifiers for the anchor template).
    # Index [4]      -> objectness_score  : Confidence score (0 to 1) that an actual object exists inside this box.
    # Index [5 to end]-> class_scores (20) : Probability scores for each of the 20 object categories (e.g., car, person).
    # =========================================================================================
    output = output.reshape(grid_size, grid_size, num_anchors, 5 + num_classes)

    boxes = []
    confidences = []

    for row in range(grid_size):
        for col in range(grid_size):
            for a in range(num_anchors):
                tx, ty, tw, th, obj_score = output[row, col, a, :5]
                obj_score = sigmoid(obj_score)

                if obj_score < conf_threshold:
                    continue

                bx = (sigmoid(tx) + col) / grid_size
                by = (sigmoid(ty) + row) / grid_size

                # Ancoraggi normalizzati basati su input size 640
                bw = np.exp(tw) / grid_size
                bh = np.exp(th) / grid_size

                x = int((bx - bw / 2) * W)
                y = int((by - bh / 2) * H)
                w = int(bw * W)
                h = int(bh * H)

                class_probs = sigmoid(output[row, col, a, 5:])
                confidence = float(obj_score * class_probs[0])

                if confidence > conf_threshold:
                    boxes.append([x, y, w, h])
                    confidences.append(confidence)

    indices = cv2.dnn.NMSBoxes(
        boxes,
        confidences,
        conf_threshold,
        nms_threshold
    )

    face_detected = False
    best_face_bbox = None
    best_face_conf = 0.0

    if len(indices) > 0:
        face_detected = True
        
        # Trova il volto con la confidenza più alta
        for i in indices.flatten():
            conf = float(confidences[i])
            if conf > best_face_conf:
                best_face_conf = conf
                x, y, w, h = boxes[i]
                
                # Applica il padding del 30% (Preso dal tuo script di Colab)
                pad_h = int(h * 0.3)
                pad_w = int(w * 0.3)

                xmin = max(0, int(x - pad_w))
                ymin = max(0, int(y - pad_h))
                xmax = min(W - 1, int(x + w + pad_w))
                ymax = min(H - 1, int(y + h + pad_h))
                
                best_face_bbox = (xmin, ymin, xmax, ymax)

        # Se abbiamo un volto valido con padding coerente, disegna il riquadro di base
        if best_face_bbox is not None:
            xmin, ymin, xmax, ymax = best_face_bbox
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255, 255, 255), 2)
            cv2.putText(
                frame,
                f"Face {best_face_conf:.2f}",
                (xmin, max(0, ymin - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )

    return frame, face_detected, best_face_bbox, best_face_conf


# ------------------- YOLO DPU THREAD ------------------- #
class YoloDpuThread(threading.Thread):
    """
    Thread che controlla la telecamera e gestisce l'inferenza di YOLOv3u (Face Detector) sulla DPU.
    Invia i dati grezzi del volto rilevato alla logica principale/dashboard.
    """

    def __init__(
        self,
        device_id: str = "camera0",
        camera_index: int = 0,
        debug_window: bool = False,
        roi_state=None
    ):
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

        # Variabili aggiornate per il tracciamento dei volti
        self.latest_face_bbox = None
        self.latest_face_conf = 0.0
        self.latest_face_bbox_ts = None

        self.no_face_timeout_sec = 5.0
        self._last_face_seen_ts = None

        self.roi_state = roi_state
        self.phase = "idle"

    def activate(self):
        self.active_event.set()

    def deactivate(self):
        self.active_event.clear()
        self._last_face_seen_ts = None

        with self.result_lock:
            self.latest_result = None
            self.latest_result_ts = None
            self.latest_frame = None

            self.latest_face_bbox = None
            self.latest_face_conf = 0.0
            self.latest_face_bbox_ts = None

    def get_latest_result(self):
        with self.result_lock:
            return self.latest_result, self.latest_result_ts
        
    def get_latest_face_bbox(self):
        """
        Restituisce l'ultima bounding box del volto rilevato (comprensiva di padding 30%).
        """
        with self.result_lock:
            return (
                self.latest_face_bbox,
                self.latest_face_conf,
                self.latest_face_bbox_ts
            )

    def get_latest_frame(self):
        with self.result_lock:
            if self.latest_frame is None:
                return None
            return self.latest_frame.copy()

    def is_active(self):
        return self.active_event.is_set()

    def run(self):
        try:
            self.phase = "loading"

            model_path = get_yolo_path()
            graph = xir.Graph.deserialize(model_path)

            dpu_subgraphs = [
                sg for sg in graph.get_root_subgraph().toposort_child_subgraph()
                if sg.has_attr("device") and sg.get_attr("device").upper() == "DPU"
            ]

            if not dpu_subgraphs:
                from utils.logger import log_system
                log_system("[YoloDpuThread] No DPU subgraph found", level="ERROR")
                self.phase = "error"
                return

            dpu_subgraph = dpu_subgraphs[0]
            self.runner = vart.Runner.create_runner(dpu_subgraph, "run")

            input_tensors = self.runner.get_input_tensors()
            output_tensors = self.runner.get_output_tensors()
            input_shape = tuple(input_tensors[0].dims)

            self.phase = "idle"

            while not self.stop_event.is_set():
                if not self.active_event.wait(timeout=0.5):
                    self.phase = "idle"
                    continue

                if self.stop_event.is_set():
                    break

                self.phase = "opening_camera"
                self.cap = cv2.VideoCapture(self.camera_index)

                if not self.cap.isOpened():
                    from utils.logger import log_system
                    log_system("[YoloDpuThread] Webcam not found", level="ERROR")
                    self.active_event.clear()
                    self.phase = "idle"
                    continue

                # Impostiamo a 640x480 come definito nei test su Colab (imgsz=640)
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

                output_data = [
                    np.empty(tuple(ot.dims), dtype=np.float32)
                    for ot in output_tensors
                ]

                self._last_face_seen_ts = time.monotonic()
                self.phase = "running"

                while self.active_event.is_set() and not self.stop_event.is_set():
                    ret, frame = self.cap.read()

                    if not ret:
                        from utils.logger import log_system
                        log_system("[YoloDpuThread] Failed to read frame", level="WARNING")
                        break

                    # Effetto specchio per mantenere la coerenza con i modelli di classificazione
                    frame = cv2.flip(frame, 1)

                    img_input = preprocess(frame, input_shape)

                    job_id = self.runner.execute_async([img_input], output_data)
                    self.runner.wait(job_id)

                    # Postprocess specifico per estrarre la faccia
                    frame, face_detected, face_bbox, face_conf = postprocess(
                        output_data[0],
                        frame
                    )

                    # Se presente un gestore dello stato ROI, aggiornalo con le coordinate del volto
                    if (
                        self.roi_state is not None
                        and face_detected
                        and face_bbox is not None
                    ):
                        self.roi_state.update_from_yolo(
                            bbox_xyxy=face_bbox,
                            confidence=face_conf
                        )

                    now = time.monotonic()

                    if face_detected:
                        self._last_face_seen_ts = now
                    else:
                        if (
                            self._last_face_seen_ts is not None
                            and now - self._last_face_seen_ts >= self.no_face_timeout_sec
                        ):
                            self.deactivate()
                            break

                    # Scrittura thread-safe dei risultati da passare alla Dashboard (che userà ResNet)
                    with self.result_lock:
                        self.latest_result = face_detected
                        self.latest_result_ts = now
                        self.latest_frame = frame.copy()

                        if face_detected and face_bbox is not None:
                            self.latest_face_bbox = face_bbox
                            self.latest_face_conf = face_conf
                            self.latest_face_bbox_ts = now
                        else:
                            self.latest_face_bbox = None
                            self.latest_face_conf = 0.0
                            self.latest_face_bbox_ts = now

                    if self.debug_window:
                        cv2.imshow("YOLOv3u DPU Face Detector", frame)
                        cv2.waitKey(1)

                self.phase = "closing_camera"

                if self.cap is not None:
                    self.cap.release()
                    self.cap = None

                if self.debug_window:
                    cv2.destroyWindow("YOLOv3u DPU Face Detector")

                with self.result_lock:
                    self.latest_frame = None

                self.phase = "idle"

        finally:
            if self.cap is not None:
                self.cap.release()
                self.cap = None

            if self.debug_window:
                try:
                    cv2.destroyWindow("YOLOv3u DPU Face Detector")
                except cv2.error:
                    pass

            self.runner = None
            self.phase = "stopped"

    def stop(self):
        self.stop_event.set()
        self.active_event.set()
        if self.is_alive():
            self.join(timeout=3.0)
