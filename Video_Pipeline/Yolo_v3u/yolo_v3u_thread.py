# yolo_face_thread.py

import cv2
import numpy as np
import vart
import xir
import threading
import time

from utils.config import get_yolo_path

# ---------------------- CLASS NAMES ---------------------- #
CLASS_NAMES = ["face"]


# ---------------------- PREPROCESS FUNCTION ---------------------- #
def preprocess(frame, input_shape):
    """
    Ridimensiona e normalizza il frame intero per l'input della DPU (YOLOv3u).
    """
    height, width = input_shape[1], input_shape[2]
    image = cv2.resize(frame, (width, height))
    image = image.astype(np.float32) / 255.0
    image = np.ascontiguousarray(image)
    return image


# ---------------------- SIGMOID FUNCTION ---------------------- #
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# ---------------------- POSTPROCESS FUNCTION ---------------------- #
def postprocess(output, frame, conf_threshold=0.5, nms_threshold=0.4):
    """
    Post-elaborazione DPU con estrazione e pre-elaborazione del volto (Padding 30% e B&N).
    """
    H, W = frame.shape[:2]

    # Configurazione griglia YOLO standard su DPU
    anchors = [(116, 90), (156, 198), (373, 326)]
    grid_size = 13
    num_anchors = 3
    num_classes = 1

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

    face_detected = False
    best_face_bbox = None
    best_face_conf = 0.0
    face_crop_gray_3ch = None  # Conterrà il crop pronto per la ResNet18

    if len(indices) > 0:
        # Prendiamo il primo volto rilevato (il più sicuro tramite ordinamento implicito o NMS)
        i = indices.flatten()[0]
        x, y, w, h = boxes[i]
        conf = float(confidences[i])

        # 1. Calcolo del PADDING DEL 30% (esattamente come su Colab)
        pad_h = int(h * 0.3)
        pad_w = int(w * 0.3)

        xmin = max(0, x - pad_w)
        ymin = max(0, y - pad_h)
        xmax = min(W - 1, x + w + pad_w)
        ymax = min(H - 1, y + h + pad_h)

        if xmax > xmin and ymax > ymin:
            face_detected = True
            best_face_conf = conf
            best_face_bbox = (xmin, ymin, xmax, ymax)

            # 2. RITAGLIO DELLA FACCIA
            faccia_ritagliata = frame[ymin:ymax, xmin:xmax]

            if faccia_ritagliata.size > 0:
                # 3. ELABORAZIONE IN SCALA DI GRIGI REPLICATA SUI TRE CANALI
                faccia_gray = cv2.cvtColor(faccia_ritagliata, cv2.COLOR_BGR2GRAY)
                face_crop_gray_3ch = cv2.cvtColor(faccia_gray, cv2.COLOR_GRAY2RGB)

    return frame, face_detected, best_face_bbox, best_face_conf, face_crop_gray_3ch


# ------------------- YOLO FACE DPU THREAD ------------------- #
class YoloFaceDpuThread(threading.Thread):
    """
    Thread che gestisce la telecamera, esegue YOLOv3u sulla DPU e prepara i ritagli
    dei volti in tempo reale per il successivo step di classificazione (ResNet18).
    """

    def __init__(
        self,
        device_id: str = "camera0",
        camera_index: int = 0,
        debug_window: bool = False
    ):
        super().__init__(daemon=True)

        self.device_id = device_id
        self.camera_index = camera_index
        self.debug_window = debug_window

        self.stop_event = threading.Event()
        self.active_event = threading.Event()

        self.cap = None
        self.runner = None

        # Lock per la sincronizzazione dei dati condivisi (Thread-Safe)
        self.result_lock = threading.Lock()
        self.latest_result = None
        self.latest_result_ts = None
        self.latest_frame = None

        # Variabili specifiche per il tracciamento del volto e del ritaglio
        self.latest_face_bbox = None
        self.latest_face_conf = 0.0
        self.latest_face_crop = None  # Immagine B&N a 3 canali pronta per ResNet
        self.latest_face_ts = None

        self.no_face_timeout_sec = 5.0
        self._last_face_seen_ts = None
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
            self.latest_face_crop = None
            self.latest_face_ts = None

    def get_latest_face_data(self):
        """
        Restituisce i dati completi dell'ultimo volto rilevato.
        Utile per la dashboard o direttamente per il modulo ResNet18.
        """
        with self.result_lock:
            crop_copy = self.latest_face_crop.copy() if self.latest_face_crop is not None else None
            return (
                self.latest_face_bbox,
                self.latest_face_conf,
                crop_copy,
                self.latest_face_ts
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
                log_system("[YoloFaceDpuThread] No DPU subgraph found", level="ERROR")
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
                    log_system("[YoloFaceDpuThread] Webcam not found", level="ERROR")
                    self.active_event.clear()
                    self.phase = "idle"
                    continue

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
                        log_system("[YoloFaceDpuThread] Failed to read frame", level="WARNING")
                        break

                    # Specchia il frame per coordinarsi con i sistemi di visualizzazione standard
                    frame = cv2.flip(frame, 1)

                    # Esecuzione del modello Face Detector sulla DPU
                    img_input = preprocess(frame, input_shape)
                    job_id = self.runner.execute_async([img_input], output_data)
                    self.runner.wait(job_id)

                    # Post-elaborazione con estrazione del crop in scala di grigi
                    frame_drawn, face_detected, face_bbox, face_conf, face_crop = postprocess(
                        output_data[0],
                        frame.copy(),  # Passiamo una copia per disegnare i rettangoli senza sporcare il frame pulito
                        conf_threshold=0.5
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

                    # Aggiornamento dello stato (Thread-safe)
                    with self.result_lock:
                        self.latest_result = face_detected
                        self.latest_result_ts = now
                        self.latest_frame = frame_drawn

                        if face_detected:
                            self.latest_face_bbox = face_bbox
                            self.latest_face_conf = face_conf
                            self.latest_face_crop = face_crop
                            self.latest_face_ts = now
                        else:
                            self.latest_face_bbox = None
                            self.latest_face_conf = 0.0
                            self.latest_face_crop = None

                    if self.debug_window:
                        cv2.imshow("YOLO Face DPU (Drawn)", frame_drawn)
                        if face_crop is not None:
                            # Finestra di debug per verificare che il crop sia convertito correttamente in grigi
                            cv2.imshow("ResNet Input Preview (Crop)", face_crop)
                        cv2.waitKey(1)

                self.phase = "closing_camera"
                if self.cap is not None:
                    self.cap.release()
                    self.cap = None

                if self.debug_window:
                    cv2.destroyWindow("YOLO Face DPU (Drawn)")
                    cv2.destroyWindow("ResNet Input Preview (Crop)")

                with self.result_lock:
                    self.latest_frame = None

                self.phase = "idle"

        finally:
            if self.cap is not None:
                self.cap.release()
                self.cap = None
            self.runner = None
            self.phase = "stopped"

    def stop(self):
        self.stop_event.set()
        self.active_event.set()
        if self.is_alive():
            self.join(timeout=3.0)
