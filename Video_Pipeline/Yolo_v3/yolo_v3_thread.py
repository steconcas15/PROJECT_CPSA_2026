import cv2
import numpy as np
import vart
import xir
import threading
import time

from utils.config import get_yolo_path, get_resnet_path

# Configurazione Margini e Costanti per ResNet integrato
ROI_MAX_AGE_SEC = 1.0
YOLO_ROI_MARGIN_X = 0.30
YOLO_ROI_MARGIN_Y = 0.30
RESNET_CLASS_NAMES = ['DROWSY', 'NATURAL']

# Inizializzazione Cascade per la faccia
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

YOLO_CLASS_NAMES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow", "diningtable",
    "dog", "horse", "motorbike", "person", "pottedplant",
    "sheep", "sofa", "train", "tvmonitor"
]

# ---------------------------------------------------------
# FUNZIONI DI UTILITÀ DPU & PREPROCESS
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


def preprocess_yolo(frame, input_shape):
    height, width = input_shape[1], input_shape[2]
    image = cv2.resize(frame, (width, height))
    image = image.astype(np.float32) / 255.0
    return np.ascontiguousarray(image)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def clamp_bbox_xyxy(x1, y1, x2, y2, img_w, img_h):
    x1 = max(0, min(img_w - 1, int(x1)))
    y1 = max(0, min(img_h - 1, int(y1)))
    x2 = max(0, min(img_w, int(x2)))
    y2 = max(0, min(img_h, int(y2)))
    if x2 <= x1 or y2 <= y1: return None
    return x1, y1, x2, y2


def expand_bbox_xyxy(bbox_xyxy, img_w, img_h, margin_x=0.30, margin_y=0.30):
    if bbox_xyxy is None: return None
    x1, y1, x2, y2 = bbox_xyxy
    w = x2 - x1
    h = y2 - y1
    if w <= 0 or h <= 0: return None
    dx = int(w * margin_x)
    dy = int(h * margin_y)
    return clamp_bbox_xyxy(x1 - dx, y1 - dy, x2 + dx, y2 + dy, img_w, img_h)


def preprocess_for_resnet(frame, input_shape, input_tensor, roi_bbox):
    target_h, target_w = input_shape[1], input_shape[2] 
    img_h, img_w = frame.shape[:2]

    expanded = expand_bbox_xyxy(roi_bbox, img_w, img_h, margin_x=YOLO_ROI_MARGIN_X, margin_y=YOLO_ROI_MARGIN_Y)
    if expanded is None:
        expanded = (0, 0, img_w, img_h)

    crop_x1, crop_y1, crop_x2, crop_y2 = expanded
    crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]

    # Convertiamo in RGB (OpenCV legge BGR, ma ResNet su Colab/PIL è stata addestrata in RGB)
    if crop.size > 0:
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    else:
        crop_rgb = np.zeros((target_h, target_w, 3), dtype=np.uint8)

    # Ridimensionamento diretto identico al Resize((224, 224)) del Colab (senza letterbox)
    resized = cv2.resize(crop_rgb, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

    # Normalizzazione identica al Colab: ToTensor() [0, 1] + Normalize con medie e std di ImageNet
    image = resized.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    image = (image - mean) / std

    # Quantizzazione fissa per DPU Xilinx
    fix_point = get_fix_point(input_tensor)
    if fix_point is not None:
        quant_scale = 2 ** fix_point
        image = image * quant_scale
        image = np.clip(image, -128, 127).astype(np.int8)

    return np.ascontiguousarray(image.reshape(input_shape)), expanded


# ---------------------------------------------------------
# FUNZIONE POSTPROCESS INTEGRATA (Esegue YOLO + RESNET INLINE)
# ---------------------------------------------------------
def postprocess_and_classify(yolo_output, frame, resnet_runner, resnet_in_shape, resnet_in_tensor, resnet_out_tensors, resnet_out_data, conf_threshold=0.6, nms_threshold=0.4):
    H, W = frame.shape[:2]
    anchors = [(116, 90), (156, 198), (373, 326)]
    grid_size = 13
    num_anchors = 3
    num_classes = 20

    yolo_output = yolo_output.reshape(grid_size, grid_size, num_anchors, 5 + num_classes)

    boxes = []
    confidences = []
    class_ids = []
    prediction_name = None

    for row in range(grid_size):
        for col in range(grid_size):
            for a in range(num_anchors):
                tx, ty, tw, th, obj_score = yolo_output[row, col, a, :5]
                obj_score = sigmoid(obj_score)

                if obj_score < conf_threshold: continue

                bx = (sigmoid(tx) + col) / grid_size
                by = (sigmoid(ty) + row) / grid_size
                bw = np.exp(tw) * anchors[a][0] / 416
                bh = np.exp(th) * anchors[a][1] / 416

                x = int((bx - bw / 2) * W)
                y = int((by - bh / 2) * H)
                w = int(bw * W)
                h = int(bh * H)

                class_probs = sigmoid(yolo_output[row, col, a, 5:])
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
            label = YOLO_CLASS_NAMES[class_ids[i]]
            conf = float(confidences[i])

            x1 = max(0, int(x))
            y1 = max(0, int(y))
            x2 = min(W - 1, int(x + w))
            y2 = min(H - 1, int(y + h))

            if x2 <= x1 or y2 <= y1: continue

            if label == "person":
                person_detected = True
                person_roi = frame[y1:y2, x1:x2]
                
                face_found = False
                target_box = None

                if person_roi.size > 0:
                    gray_person = cv2.cvtColor(person_roi, cv2.COLOR_BGR2GRAY)
                    faces = face_cascade.detectMultiScale(gray_person, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
                    
                    if len(faces) > 0:
                        fx, fy, fw, fh = faces[0]
                        target_box = (x1 + fx, y1 + fy, x1 + fx + fw, y1 + fy + fh)
                        face_found = True
                        if conf > best_person_conf:
                            best_person_conf = conf
                            best_person_bbox = target_box
                
                if not face_found:
                    y_head_shoulders = y1 + int((y2 - y1) * 0.4)
                    center_x = (x1 + x2) // 2
                    w_fallback = int((y_head_shoulders - y1) * 1.2)
                    target_box = (max(0, center_x - w_fallback // 2), y1, min(W - 1, center_x + w_fallback // 2), y_head_shoulders)
                    if conf > best_person_conf:
                        best_person_conf = conf
                        best_person_bbox = target_box

                # --- CHIAMATA INLINE A RESNET ALLINEATA AL COLAB ---
                if target_box is not None:
                    resnet_img, expanded_box = preprocess_for_resnet(frame, resnet_in_shape, resnet_in_tensor, target_box)
                    
                    job_id = resnet_runner.execute_async([resnet_img], resnet_out_data)
                    resnet_runner.wait(job_id)
                    
                    raw_logits = dequantize(resnet_out_data[0], resnet_out_tensors[0])
                    logits = raw_logits.flatten()[:2] # Prendiamo esattamente le prime 2 classi (DROWSY, NATURAL)
                    
                    # Calcolo Softmax numericamente stabile in NumPy
                    exp_logits = np.exp(logits - np.max(logits))
                    probabilities = exp_logits / np.sum(exp_logits)
                    
                    pred_class_id = int(np.argmax(probabilities))
                    prediction_name = RESNET_CLASS_NAMES[pred_class_id]
                    resnet_confidence = float(probabilities[pred_class_id])

                    # Configurazione grafica basata sullo stato
                    color = (0, 0, 255) if prediction_name == "DROWSY" else (0, 255, 0)
                    suffix = "" if face_found else " (Est.)"
                    text_label = f"{prediction_name} {resnet_confidence*100:.1f}%{suffix}"
                    
                    cv2.rectangle(frame, (expanded_box[0], expanded_box[1]), (expanded_box[2], expanded_box[3]), color, 3)
                    cv2.putText(frame, text_label, (expanded_box[0], max(20, expanded_box[1] - 10)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            else:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} {conf:.2f}", (x1, max(0, y1 - 5)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    return frame, person_detected, best_person_bbox, best_person_conf, prediction_name


# ---------------------------------------------------------
# UNICO MONOLITHIC YOLO DPU THREAD
# ---------------------------------------------------------
class YoloDpuThread(threading.Thread):
    def __init__(self, device_id="camera0", camera_index=0, debug_window=False, roi_state=None):
        super().__init__(daemon=True)
        self.device_id = device_id
        self.camera_index = camera_index
        self.debug_window = debug_window
        self.stop_event = threading.Event()
        self.active_event = threading.Event()
        
        self.cap = None
        self.yolo_runner = None
        self.resnet_runner = None
        
        self.result_lock = threading.Lock()
        self.latest_result = None
        self.latest_result_ts = None
        self.latest_frame = None
        
        self.latest_person_bbox = None
        self.latest_person_conf = 0.0
        self.latest_person_bbox_ts = None
        self.latest_prediction = None
        
        self.roi_state = roi_state
        self.phase = "idle"

    def activate(self):
        self.active_event.set()

    def deactivate(self):
        self.active_event.clear()
        with self.result_lock:
            self.latest_result = None
            self.latest_result_ts = None
            self.latest_frame = None
            self.latest_person_bbox = None
            self.latest_person_conf = 0.0
            self.latest_person_bbox_ts = None
            self.latest_prediction = None

    def get_latest_result(self):
        with self.result_lock:
            return self.latest_result, self.latest_result_ts

    def get_latest_prediction(self):
        with self.result_lock:
            return self.latest_prediction
        
    def get_latest_person_bbox(self): #
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
            
            # 1. Caricamento Modello YOLO
            yolo_graph = xir.Graph.deserialize(get_yolo_path())
            yolo_subgraph = get_dpu_subgraph(yolo_graph)
            self.yolo_runner = vart.Runner.create_runner(yolo_subgraph, "run")
            
            yolo_input_tensors = self.yolo_runner.get_input_tensors()
            yolo_output_tensors = self.yolo_runner.get_output_tensors()
            yolo_input_shape = tuple(yolo_input_tensors[0].dims)

            # 2. Caricamento Modello RESNET
            resnet_graph = xir.Graph.deserialize(get_resnet_path())
            resnet_subgraph = get_dpu_subgraph(resnet_graph)
            self.resnet_runner = vart.Runner.create_runner(resnet_subgraph, "run")
            
            resnet_input_tensors = self.resnet_runner.get_input_tensors()
            resnet_output_tensors = self.resnet_runner.get_output_tensors()
            resnet_input_shape = tuple(resnet_input_tensors[0].dims)
            resnet_input_tensor = resnet_input_tensors[0]

            self.phase = "idle"

            while not self.stop_event.is_set():
                if not self.active_event.wait(timeout=0.5):
                    self.phase = "idle"
                    continue
                
                if self.stop_event.is_set(): break
                
                self.phase = "opening_camera"

                # Retry con backoff: V4L2 puo' impiegare qualche centinaio di ms
                # a liberare il device dopo un cap.release() precedente. Un fallimento
                # immediato qui faceva active_event.clear() silenziosamente, disallineando
                # lo stato del thread da quello creduto dal dispatcher (la camera restava
                # "spenta per sempre" finche' non arrivava un nuovo evento esterno).
                camera_open_attempts = 5
                camera_open_retry_delay_sec = 0.3
                opened = False

                for attempt in range(camera_open_attempts):
                    if self.stop_event.is_set() or not self.active_event.is_set():
                        break

                    self.cap = cv2.VideoCapture(self.camera_index)
                    if self.cap.isOpened():
                        opened = True
                        break

                    self.cap.release()
                    self.cap = None
                    time.sleep(camera_open_retry_delay_sec)

                if not opened:
                    # Non disattiviamo active_event: il dispatcher resta la fonte
                    # di verita' su quando il video deve essere attivo o no.
                    # Torniamo idle e ritenteremo al prossimo giro del while esterno.
                    self.phase = "idle"
                    time.sleep(0.5)
                    continue

                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                
                # Allocazione buffer DPU dinamici e precisi
                yolo_output_data = [np.empty(tuple(ot.dims), dtype=np.float32) for ot in yolo_output_tensors]
                resnet_output_data = [
                    np.empty(tuple(ot.dims), dtype=np.int8 if get_fix_point(ot) is not None else np.float32)
                    for ot in resnet_output_tensors
                ]
                
                self.phase = "running"
                
                while self.active_event.is_set() and not self.stop_event.is_set():
                    ret, frame = self.cap.read()
                    if not ret: break
                    frame = cv2.flip(frame, 1)
                    
                    # Esecuzione YOLOv3
                    yolo_img = preprocess_yolo(frame, yolo_input_shape)
                    yolo_job_id = self.yolo_runner.execute_async([yolo_img], yolo_output_data)
                    self.yolo_runner.wait(yolo_job_id)
                    
                    # Postprocess combinato con classificazione inline
                    frame, person_detected, person_bbox, person_conf, prediction_name = postprocess_and_classify(
                        yolo_output_data[0], frame, 
                        self.resnet_runner, resnet_input_shape, resnet_input_tensor,
                        resnet_output_tensors, resnet_output_data
                    )
                    
                    if self.roi_state is not None and person_detected and person_bbox is not None:
                        self.roi_state.update_from_yolo(bbox_xyxy=person_bbox, confidence=person_conf)
                        
                    now = time.monotonic()

                    with self.result_lock:
                        self.latest_result = person_detected
                        self.latest_result_ts = now
                        self.latest_frame = frame.copy()
                        self.latest_prediction = prediction_name
                        if person_detected and person_bbox is not None:
                            self.latest_person_bbox = person_bbox
                            self.latest_person_conf = person_conf
                            self.latest_person_bbox_ts = now
                    
                    if self.debug_window:
                        cv2.imshow("CPSA Integrated Stream", frame)
                        cv2.waitKey(1)
                        
                if self.cap: self.cap.release(); self.cap = None
                if self.debug_window: cv2.destroyWindow("CPSA Integrated Stream")
                with self.result_lock: self.latest_frame = None
                self.phase = "idle"
        finally:
            if self.cap: self.cap.release()
            self.yolo_runner = None
            self.resnet_runner = None
            self.phase = "stopped"

    def stop(self):
        self.stop_event.set()
        self.active_event.set()
        if self.is_alive(): self.join(timeout=3.0)
