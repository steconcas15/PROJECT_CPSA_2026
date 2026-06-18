import cv2
import numpy as np
import vart
import xir
import threading
import time

ROI_MAX_AGE_SEC = 1.0

# Margini di espansione per il ritaglio (nel tuo script Colab usi il 30%, quindi impostiamo 0.30)
YOLO_ROI_MARGIN_X = 0.30
YOLO_ROI_MARGIN_Y = 0.30

CLASS_NAMES = ['DROWSY', 'NATURAL']

# ---------------------------------------------------------
# UTILITIES DPU
# ---------------------------------------------------------

def get_dpu_subgraph(graph):
    subgraphs = [
        sg for sg in graph.get_root_subgraph().toposort_child_subgraph()
        if sg.has_attr("device") and sg.get_attr("device").upper() == "DPU"
    ]
    if len(subgraphs) != 1:
        raise RuntimeError(f"Expected 1 DPU subgraph, found {len(subgraphs)}")
    return subgraphs[0]


def get_fix_point(tensor):
    return tensor.get_attr("fix_point") if tensor.has_attr("fix_point") else None


def dequantize(output, tensor):
    fix_point = get_fix_point(tensor)
    if fix_point is None:
        return output.astype(np.float32)
    return output.astype(np.float32) / float(2 ** fix_point)


# ---------------------------------------------------------
# IMAGE PROCESSING (Adattato per la scala di grigi a 3 canali)
# ---------------------------------------------------------

def clamp_bbox_xyxy(x1, y1, x2, y2, img_w, img_h):
    x1 = max(0, min(img_w - 1, int(x1)))
    y1 = max(0, min(img_h - 1, int(y1)))
    x2 = max(0, min(img_w, int(x2)))
    y2 = max(0, min(img_h, int(y2)))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def expand_bbox_xyxy(bbox_xyxy, img_w, img_h, margin_x=0.30, margin_y=0.30):
    if bbox_xyxy is None:
        return None
    x1, y1, x2, y2 = bbox_xyxy
    w = x2 - x1
    h = y2 - y1
    if w <= 0 or h <= 0:
        return None
    dx = int(w * margin_x)
    dy = int(h * margin_y)
    return clamp_bbox_xyxy(x1 - dx, y1 - dy, x2 + dx, y2 + dy, img_w, img_h)


def letterbox_image(image, target_w, target_h, color=(0, 0, 0)):
    h, w = image.shape[:2]
    scale = min(target_w / float(w), target_h / float(h))
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    pad_w = target_w - new_w
    pad_h = target_h - new_h
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    out = cv2.copyMakeBorder(resized, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=color)
    return out, scale, pad_left, pad_top


def preprocess_for_resnet(frame, input_shape, input_tensor, roi_bbox=None):
    """
    Preprocessamento con conversione in scala di grigi replicata su 3 canali,
    coerente con l'addestramento del modello nel Colab.
    """
    target_h, target_w = input_shape[1], input_shape[2] 
    img_h, img_w = frame.shape[:2]

    if roi_bbox is not None:
        expanded = expand_bbox_xyxy(roi_bbox, img_w, img_h, margin_x=YOLO_ROI_MARGIN_X, margin_y=YOLO_ROI_MARGIN_Y)
    else:
        expanded = None

    if expanded is None:
        crop_x1, crop_y1, crop_x2, crop_y2 = 0, 0, img_w, img_h
        crop_source = "full_frame"
    else:
        crop_x1, crop_y1, crop_x2, crop_y2 = expanded
        crop_source = "yolo_roi"

    crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]

    # --- INTEGRAZIONE COLAB: TRASFORMAZIONE IN SCALA DI GRIGI A 3 CANALI ---
    if crop.size > 0:
        crop_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        crop_rgb_gray = cv2.cvtColor(crop_gray, cv2.COLOR_GRAY2RGB)
    else:
        crop_rgb_gray = crop

    # Ridimensionamento mantenendo l'aspect ratio (Letterbox)
    letterboxed, scale, pad_left, pad_top = letterbox_image(crop_rgb_gray, target_w, target_h)

    # Quantizzazione hardware per Vitis-AI DPU
    fix_point = get_fix_point(input_tensor)
    if fix_point is not None:
        quant_scale = 2 ** fix_point
        image = letterboxed.astype(np.float32) / 255.0
        image = image * quant_scale
        image = np.clip(image, -128, 127).astype(np.int8)
    else:
        image = letterboxed.astype(np.float32) / 255.0

    transform = {
        "crop_source": crop_source,
        "crop_x1": crop_x1, "crop_y1": crop_y1,
        "crop_x2": crop_x2, "crop_y2": crop_y2,
    }

    return np.ascontiguousarray(image.reshape(input_shape)), crop_rgb_gray, transform


# ---------------------------------------------------------
# THREAD RESNET 18 DPU
# ---------------------------------------------------------

class ResNetDpuThread(threading.Thread):

    def __init__(self, model_path: str, camera_index: int = 0, debug_window: bool = False, roi_state=None):
        super().__init__(daemon=True)
        self.model_path = model_path
        self.camera_index = camera_index
        self.debug_window = debug_window
        self.roi_state = roi_state

        self.stop_event = threading.Event()
        self.active_event = threading.Event()

        self.cap = None
        self.runner = None
        self.result_lock = threading.Lock()

        self.latest_prediction = "UNKNOWN"
        self.latest_confidence = 0.0
        self.latest_result_ts = None
        self.latest_frame = None
        self.phase = "idle"

    def activate(self):
        self.active_event.set()

    def deactivate(self):
        self.active_event.clear()
        with self.result_lock:
            self.latest_prediction = "UNKNOWN"
            self.latest_confidence = 0.0
            self.latest_result_ts = None
            self.latest_frame = None

    def get_latest_status(self):
        with self.result_lock:
            return self.latest_prediction, self.latest_confidence, self.latest_frame, self.latest_result_ts

    def stop(self):
        self.stop_event.set()
        self.active_event.set()
        if self.is_alive():
            self.join(timeout=3.0)

    def run(self):
        try:
            self.phase = "loading"
            
            graph = xir.Graph.deserialize(self.model_path)
            dpu_subgraph = get_dpu_subgraph(graph)
            self.runner = vart.Runner.create_runner(dpu_subgraph, "run")

            input_tensors = self.runner.get_input_tensors()
            output_tensors = self.runner.get_output_tensors()

            input_tensor = input_tensors[0]
            input_shape = tuple(input_tensor.dims)

            display_w, display_h = 640, 480
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
                    print("[ResNetDpuThread] Errore: Webcam non trovata")
                    self.active_event.clear()
                    self.phase = "idle"
                    continue

                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

                if self.debug_window:
                    cv2.namedWindow("ResNet DPU Classification", cv2.WINDOW_NORMAL)
                    cv2.resizeWindow("ResNet DPU Classification", display_w, display_h)

                output_data = [
                    np.empty(tuple(ot.dims), dtype=np.int8 if get_fix_point(ot) is not None else np.float32)
                    for ot in output_tensors
                ]

                self.phase = "running"

                while self.active_event.is_set() and not self.stop_event.is_set():
                    ret, frame = self.cap.read()
                    if not ret:
                        print("[ResNetDpuThread] Errore di lettura del frame")
                        break

                    frame = cv2.flip(frame, 1)

                    # Recupero della ROI passata dall'altro thread (YOLO)
                    roi_bbox = None
                    if self.roi_state is not None:
                        roi_bbox = self.roi_state.get_valid_roi(max_age_sec=ROI_MAX_AGE_SEC)

                    # Preprocessamento con allineamento alla scala di grigi del Colab
                    img_input, crop_view, transform = preprocess_for_resnet(
                        frame, input_shape, input_tensor, roi_bbox=roi_bbox
                    )

                    # Inferenza hardware
                    job_id = self.runner.execute_async([img_input], output_data)
                    self.runner.wait(job_id)

                    # Softmax software per ricavare la confidenza
                    logits = dequantize(output_data[0], output_tensors[0])[0]
                    exp_logits = np.exp(logits - np.max(logits))
                    probabilities = exp_logits / np.sum(exp_logits)
                    
                    pred_class_id = np.argmax(probabilities)
                    prediction_name = CLASS_NAMES[pred_class_id]
                    confidence = probabilities[pred_class_id]

                    # --- AGGIORNAMENTO GRAFICO (DEBUG COERENTE CON COLAB) ---
                    debug_view = frame.copy()
                    
                    # Disegnamo il rettangolo del ritaglio
                    cv2.rectangle(
                        debug_view, 
                        (transform["crop_x1"], transform["crop_y1"]), 
                        (transform["crop_x2"], transform["crop_y2"]), 
                        (0, 255, 0), 2
                    )
                    
                    text_color = (0, 0, 255) if prediction_name == "DROWSY" else (0, 255, 0)
                    
                    cv2.putText(
                        debug_view, f"STATO: {prediction_name} ({confidence*100:.1f}%)", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2
                    )

                    now = time.monotonic()
                    with self.result_lock:
                        self.latest_prediction = prediction_name
                        self.latest_confidence = float(confidence)
                        self.latest_result_ts = now
                        self.latest_frame = debug_view.copy()

                    if self.debug_window:
                        cv2.imshow("ResNet DPU Classification", debug_view)
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            self.deactivate()
                            break

                self.phase = "closing_camera"
                if self.cap:
                    self.cap.release()
                    self.cap = None

                if self.debug_window:
                    try: cv2.destroyWindow("ResNet DPU Classification")
                    except cv2.error: pass

                self.phase = "idle"

        finally:
            if self.cap:
                self.cap.release()
            self.runner = None
            self.phase = "stopped"
