# CPSA2026/VIDEO_Pipeline/Yolo_v3u/yolo_v3u_thread.py

import cv2
import numpy as np
import vart
import xir
import threading
import time

# Nel tuo progetto reale, assicurati che questo import punti alla cartella corretta dei file di configurazione
from utils.config import get_yolo_path

# Funzione matematica di supporto per l'output grezzo della DPU
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class YoloV3uThread(threading.Thread):
    """
    Thread DPU dedicato al rilevamento del volto in tempo reale.
    Prende i frame dalla webcam, esegue l'inferenza hardware e aggiorna 
    lo stato ROI condiviso (PersonRoiState) in modo thread-safe.
    """
    def __init__(self, roi_state, camera_index=0, debug_window=False):
        super().__init__(daemon=True)
        
        # Aggancio alla "lavagna" condivisa passata dal main
        self.roi_state = roi_state
        self.camera_index = camera_index
        self.debug_window = debug_window
        
        # Eventi di controllo del ciclo del Thread
        self.stop_event = threading.Event()
        self.active_event = threading.Event()
        
        self.cap = None
        self.runner = None
        
        # Lock di sicurezza per estrarre l'ultimo frame disegnato (es. per la Dashboard)
        self.result_lock = threading.Lock()
        self.latest_frame = None
        
        # Stringa di stato monitorabile dall'applicazione principale
        self.phase = "idle"

    def activate(self):
        """Attiva il processing della telecamera e della DPU."""
        self.active_event.set()

    def deactivate(self):
        """Disattiva il processing, spegne la telecamera e pulisce la ROI."""
        self.active_event.clear()
        self.roi_state.clear()
        with self.result_lock:
            self.latest_frame = None

    def get_latest_frame(self):
        """Restituisce una copia del frame corrente (con box disegnata) in modo sicuro."""
        with self.result_lock:
            return self.latest_frame.copy() if self.latest_frame is not None else None

    def run(self):
        try:
            self.phase = "loading"
            # 1. Caricamento del modello hardware (.xmodel) sulla DPU
            model_path = get_yolo_path()
            graph = xir.Graph.deserialize(model_path)
            
            dpu_subgraphs = [
                sg for sg in graph.get_root_subgraph().toposort_child_subgraph()
                if sg.has_attr("device") and sg.get_attr("device").upper() == "DPU"
            ]
            
            if not dpu_subgraphs:
                self.phase = "error"
                return

            # Inizializzazione del Vitis AI Runner
            self.runner = vart.Runner.create_runner(dpu_subgraphs[0], "run")
            input_tensors = self.runner.get_input_tensors()
            output_tensors = self.runner.get_output_tensors()
            input_shape = tuple(input_tensors[0].dims) # Formato richiesto dalla DPU (es. 1, 416, 416, 3)

            self.phase = "idle"

            while not self.stop_event.is_set():
                # Rimane in attesa finché non viene chiamato .activate()
                if not self.active_event.wait(timeout=0.5):
                    self.phase = "idle"
                    continue

                self.phase = "opening_camera"
                self.cap = cv2.VideoCapture(self.camera_index)
                if not self.cap.isOpened():
                    self.active_event.clear()
                    self.phase = "idle"
                    continue

                # Configurazione risoluzione standard della webcam
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                
                # Alloca lo spazio di memoria per l'output della DPU
                output_data = [np.empty(tuple(ot.dims), dtype=np.float32) for ot in output_tensors]
                self.phase = "running"

                while self.active_event.is_set() and not self.stop_event.is_set():
                    ret, frame = self.cap.read()
                    if not ret:
                        break

                    # Effetto specchio per l'interfaccia utente
                    frame = cv2.flip(frame, 1)
                    H, W = frame.shape[:2]

                    # 2. PRE-ELABORAZIONE: Ridimensionamento e normalizzazione per la DPU
                    img_resized = cv2.resize(frame, (input_shape[2], input_shape[1]))
                    img_input = img_resized.astype(np.float32) / 255.0
                    img_input = np.ascontiguousarray(img_input)

                    # 3. INFERENZA HARDWARE: Esecuzione asincrona sulla DPU
                    job_id = self.runner.execute_async([img_input], output_data)
                    self.runner.wait(job_id)

                    # 4. POST-PROCESSING: Lettura delle coordinate (Griglia 13x13, 3 Ancore, 1 Classe)
                    # Struttura tensore: [row, col, anchor, [x, y, w, h, obj, class_prob]]
                    output = output_data[0].reshape(13, 13, 3, 6) 
                    anchors = [(116, 90), (156, 198), (373, 326)]
                    
                    best_conf = 0.0
                    best_box = None

                    for row in range(13):
                        for col in range(13):
                            for a in range(3):
                                tx, ty, tw, th, obj_score = output[row, col, a, :5]
                                obj_score = sigmoid(obj_score)

                                if obj_score < 0.5:
                                    continue

                                class_prob = sigmoid(output[row, col, a, 5])
                                confidence = float(obj_score * class_prob)

                                # Cerca la bounding box con la confidenza più alta nel frame
                                if confidence > 0.5 and confidence > best_conf:
                                    bx = (sigmoid(tx) + col) / 13
                                    by = (sigmoid(ty) + row) / 13
                                    bw = np.exp(tw) * anchors[a][0] / 416
                                    bh = np.exp(th) * anchors[a][1] / 416

                                    # Coordinate assolute denormalizzate rispetto alle dimensioni del frame originale
                                    x1 = int((bx - bw / 2) * W)
                                    y1 = int((by - bh / 2) * H)
                                    x2 = int((bx + bw / 2) * W)
                                    y2 = int((by + bh / 2) * H)

                                    best_conf = confidence
                                    best_box = (max(0, x1), max(0, y1), min(W - 1, x2), min(H - 1, y2))

                    # 5. SCRITTURA SULLA LAVAGNA CONDIVISA
                    if best_box is not None:
                        # Comunica le nuove coordinate grezze al PersonRoiState
                        self.roi_state.update_from_yolo(bbox_xyxy=best_box, confidence=best_conf)
                        
                        # Disegna la bounding box sul frame se la finestra di debug è attiva
                        if self.debug_window:
                            cv2.rectangle(frame, (best_box[0], best_box[1]), (best_box[2], best_box[3]), (0, 255, 0), 2)
                    else:
                        # Se il volto non viene rilevato, notifica lo stato che è necessario riacquisire il target
                        self.roi_state.mark_reacquire()

                    # Aggiornamento del frame per l'interfaccia grafica
                    with self.result_lock:
                        self.latest_frame = frame.copy()

                    if self.debug_window:
                        cv2.imshow("YOLOv3u DPU Debug", frame)
                        cv2.waitKey(1)

                if self.cap is not None:
                    self.cap.release()
                    self.cap = None
                if self.debug_window:
                    cv2.destroyAllWindows()
                
                self.phase = "idle"

        finally:
            if self.cap is not None:
                self.cap.release()
                self.cap = None
            self.runner = None
            self.phase = "stopped"

    def stop(self):
        """Ferma definitivamente l'esecuzione del thread (Shutdown)."""
        self.stop_event.set()
        self.active_event.set()
        if self.is_alive():
            self.join(timeout=3.0)
