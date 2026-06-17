# CPSA2026/VIDEO_Pipeline/Yolo_v3/yolo_v3_thread.py

import cv2
import numpy as np
import vart
import xir
import threading
import time

from utils.config import get_yolo_path


# ---------------------- CLASS NAMES ---------------------- #
# Custom model optimized exclusively for face detection
CLASS_NAMES = ["face"]


# ---------------------- PREPROCESS FUNCTION ---------------------- #
def preprocess(frame, input_shape):
    """
    Resizes, normalizes, and prepares a camera frame for DPU model execution.

    Steps:
    1. Extracts target dimensions from the DPU model's expected input shape.
    2. Geometrically resizes the frame to match the network requirements (640x640).
    3. Normalizes pixel values from [0, 255] integers to [0.0, 1.0] float32.
    4. Forces memory layout to be contiguous for optimized hardware DMA transfer.

    Args:
        frame (numpy.ndarray): Raw BGR image captured from the webcam.
        input_shape (tuple): Expected model input shape from DPU metadata (Batch, H, W, C).

    Returns:
        numpy.ndarray: Preprocessed float32 image ready for DPU inference.
    """
    height, width = input_shape[1], input_shape[2]  # Extracts 640, 640 from model specs
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
def postprocess(output, frame, conf_threshold=0.6, nms_threshold=0.4):
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

    # YOLOv3 standard anchors for the macro-objects detection scale
    anchors = [(116, 90), (156, 198), (373, 326)]

    # The image is divided into a 20x20 macro-cell grid (640 input / 32 stride = 20)
    grid_size = 20

    # Defines the number of predefined bounding box shapes (anchors) evaluated per grid cell.
    # YOLOv3 tests 3 different aspect ratios at this scale to capture small, medium, and large faces.
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
    # 1. GridY & GridX (They are coordinates): The image is divided into a virtual 20x20 chessboard. 
    #    Each (Y, X) coordinate points to a specific geographic cell in the image.
    # 2. Anchors (3): Inside each single cell, the network tests 3 pre-defined bounding box 
    #    shapes (anchor templates) to detect objects of different sizes/aspect ratios.
    # 3. Properties (5 + num_classes): For each anchor in each cell, the model predicts 
    #    a specific vector of data. If num_classes = 1, this vector contains 6 elements.
    # 
    # Properties Array Layout Breakdown:
    # -------------------------------------------------------------------------------------
    # Index [0, 1]   -> tx, ty            : Box center offset relative to the top-left corner of the current grid cell.
    # Index [2, 3]   -> tw, th            : Box width and height scale factors (modifiers for the anchor template).
    # Index [4]      -> objectness_score  : Confidence score (0 to 1) that an actual object exists inside this box.
    # Index [5]      -> class_scores (1)  : Probability scores for the face.
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
                    
                # =========================================================================================
                # STEP-BY-STEP NUMERICAL EXAMPLE FOR YOLO COORDINATE DECODING (640x640 MODEL)
                # =========================================================================================
                # Let's trace how raw DPU numbers turn into real camera pixels using a concrete example:
                #
                # ASSUMPTIONS (Camera & Grid Setup):
                # - Camera Resolution (W, H) = 640 x 480 pixels
                # - Grid Size = 20x20 cells. Each cell manages a chunk of 32.0 x 32.0 pixels on the 640 input map.
                # - Current Loop Position: row (GridY) = 5, col (GridX) = 10 (Upper-center section of the screen)
                # - Active Anchor Template: a = 0 (anchors[0] = width: 116, height: 90)
                #
                # RAW DPU OUTPUT VALUES (Generated by the model for this specific cell):
                # - tx = 0.5   | ty = -0.2  (Raw center offsets)
                # - tw = 0.1   | th = -0.3  (Raw size dimensions)
                # =========================================================================================

                # 1. APPLY STANDARD YOLO DECODING FORMULAS (Convert to relative 0.0 to 1.0 ratios)
                # -----------------------------------------------------------------------------------------
                # bx = (sigmoid(0.5) + 10) / 20  -> (0.621 + 10) / 20  -> 10.621 / 20 -> 0.531 (53.1% of total width)
                # by = (sigmoid(-0.2) + 5) / 20  -> (0.450 + 5) / 20   -> 5.450 / 20  -> 0.272 (27.2% of total height)
                # bw = (exp(0.1) * 116) / 640.0  -> (1.105 * 116) / 640 -> 128.18 / 640 -> 0.200 (Face is 20.0% wide as the image)
                # bh = (exp(-0.3) * 90) / 640.0  -> (0.741 * 90) / 640  -> 66.69 / 640  -> 0.104 (Face is 10.4% tall as the image)
                # -----------------------------------------------------------------------------------------
                bx = (sigmoid(tx) + col) / grid_size
                by = (sigmoid(ty) + row) / grid_size
                bw = np.exp(tw) * anchors[a][0] / 640.0
                bh = np.exp(th) * anchors[a][1] / 640.0

                # 2. CONVERT RELATIVE RATIOS INTO ABSOLUTE CAMERA PIXEL COORDINATES
                # -----------------------------------------------------------------------------------------
                # w = int(0.200 * 640) = 128 pixels (Face width)
                # h = int(0.104 * 480) = 49 pixels (Face height)
                #
                # To find the top-left corner (x, y) required by OpenCV, we shift back from the center (bx, by):
                # x = int((0.531 - (0.200 / 2)) * 640) -> int((0.531 - 0.100) * 640) -> int(0.431 * 640) = 275
                # y = int((0.272 - (0.104 / 2)) * 480) -> int((0.272 - 0.052) * 480) -> int(0.220 * 480) = 105
                # 
                # FINAL BOUNDING BOX DELIVERED TO COOPERATING FUNCTIONS: [x=275, y=105, w=128, h=49]
                # =========================================================================================
                x = int((bx - bw / 2) * W)
                y = int((by - bh / 2) * H)
                w = int(bw * W)
                h = int(bh * H)

                # =========================================================================================
                # 1. CLASS PROBABILITIES & COMBINED CONFIDENCE FILTERING
                # =========================================================================================
                # Extract all class scores starting from index 5 to the end. 
                # Apply sigmoid to squash values into clean probability percentages (0.0 to 1.0).
                class_probs = sigmoid(output[row, col, a, 5:])

                # Calculate the absolute final confidence score.
                # Combined Confidence = Probability of ANY object existing (obj_score) 
                #                       MULTIPLIED BY the probability that it is specifically a face (class_probs[0]).
                confidence = float(obj_score * class_probs[0])

                # Hard Threshold Filter: If the final confidence doesn't clear our minimal barrier,
                # immediately drop this candidate box and move to the next anchor/grid cell.
                if confidence > conf_threshold:
                    boxes.append([x, y, w, h])
                    confidences.append(confidence)

    # =========================================================================================
    # 2. NON-MAXIMUM SUPPRESSION (NMS) - REDUNDANCY CLEANUP
    # =========================================================================================
    # Because neighboring grid cells often detect the exact same face, we get multiple overlapping boxes.
    # cv2.dnn.NMSBoxes evaluates overlapping areas (IoU) and keeps ONLY the highest-scoring box per object,
    # effectively suppressing all weaker duplicate boxes. It returns a list of surviving indices.
    indices = cv2.dnn.NMSBoxes(
        boxes,
        confidences,
        conf_threshold,
        nms_threshold
    )

    # Initialize tracking variables for the single best target in the current frame
    face_detected = False
    best_face_bbox = None
    best_face_conf = 0.0

    # =========================================================================================
    # 3. SELECTING THE BEST FACE & APPLYING 30% VISUAL PADDING
    # =========================================================================================
    if len(indices) > 0:
        face_detected = True
        
        # Flatten the NMS index array to easily loop through all surviving boxes
        valid_indices = indices.flatten() if hasattr(indices, 'flatten') else [i[0] for i in indices]
        
        for i in valid_indices:
            conf = float(confidences[i])

            # Continuous comparison to find the absolute champion box (highest confidence score)
            if conf > best_face_conf:
                best_face_conf = conf
                x, y, w, h = boxes[i]
                
                # Target Bounding Boxes are often tightly cropped around facial features (eyes/nose/mouth).
                # To prevent clipping ears, hair, or jawlines, we compute a 30% margin buffer based on size.
                pad_h = int(h * 0.3)
                pad_w = int(w * 0.3)

                # Expand box boundaries while strictly keeping coordinates within physical camera frame limits.
                # max(0, ...) prevents coordinates from jumping off the left/top edges into negative space.
                # min(W-1, ...) prevents coordinates from overflowing past the right/bottom resolution edges.
                xmin = max(0, int(x - pad_w))
                ymin = max(0, int(y - pad_h))
                xmax = min(W - 1, int(x + w + pad_w))
                ymax = min(H - 1, int(y + h + pad_h))

                # Save the final optimized, padded box coordinates
                best_face_bbox = (xmin, ymin, xmax, ymax)

        # =========================================================================================
        # 4. RENDERING VISUALS ON SCREEN
        # =========================================================================================
        if best_face_bbox is not None:
            xmin, ymin, xmax, ymax = best_face_bbox

            # Draw a solid white bounding box rectangle over the original frame (Thickness = 2 pixels)
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255, 255, 255), 2)

            # Overlay text showing confidence percentage (e.g., "Face 0.94").
            # The text is dynamically anchored 5 pixels ABOVE the top line (ymin - 5), 
            # using max(0, ...) to ensure the text stays visible even if the face hits the top border.
            cv2.putText(
                frame,
                f"Face {best_face_conf:.2f}",
                (xmin, max(0, ymin - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )
            
    # Output processed frame alongside metadata to be used by downstream pipeline tasks
    return frame, face_detected, best_face_bbox, best_face_conf


# ------------------- YOLO DPU THREAD ------------------- #
class YoloDpuThread(threading.Thread):
    """
    Thread that controls the USB camera and runs YOLOv3 inference on the DPU.

    Important GUI rule:
        This thread does not create OpenCV windows unless debug_window=True.
        The dashboard should be the normal owner of cv2.imshow() and cv2.waitKey().
    """

    def __init__(
        self,
        device_id: str = "camera0",
        camera_index: int = 0,
        debug_window: bool = False,
        roi_state=None
    ):
        """
        Initialize the YOLO DPU camera thread.

        Args:
            device_id: Logical identifier for the camera sensor.
            camera_index: OpenCV camera index.
            debug_window: If True, show YOLO's own debug window.
                          Keep False when using VideoDashboard.
        """
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
                log_system("[YoloDpuThread] No DPU subgraph found inside model binary", level="ERROR")
                self.phase = "error"
                return

            dpu_subgraph = dpu_subgraphs[0]

            # Instantiate the Vitis-AI execution runner
            self.runner = vart.Runner.create_runner(dpu_subgraph, "run")

            # Extract hardware input/output structural dimension shapes
            input_tensors = self.runner.get_input_tensors()
            output_tensors = self.runner.get_output_tensors()
            input_shape = tuple(input_tensors[0].dims)  # Configured for [1, 640, 640, 3]

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
                    log_system("[YoloDpuThread] Failed to open targeted webcam source device", level="ERROR")

                    self.active_event.clear()
                    self.phase = "idle"
                    continue

                # Set native resolution frame boundaries for initial acquisition
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

                # Dynamically allocate empty NumPy arrays with shape profiles fitting output tensors
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
                        log_system("[YoloDpuThread] Failed to read incoming video frame hardware stream", level="WARNING")
                        break

                    # Mirror the camera frame to match uniform dashboard and coordinate system expectations
                    frame = cv2.flip(frame, 1)

                    # Scale and normalize frame to match model input configurations ([1, 640, 640, 3])
                    img_input = preprocess(frame, input_shape)

                    # Submit asymmetric inference execution request to the Xilinx DPU core
                    job_id = self.runner.execute_async([img_input], output_data)
                    self.runner.wait(job_id)

                    # Extract the first tensor array element containing the relevant spatial detection grids
                    frame, face_detected, face_bbox, face_conf = postprocess(
                        output_data[0],
                        frame
                    )

                    # Pipe valid coordinates into the state-tracking interface if available
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
                        # Deactivate processing threads if no object is observed for the set duration threshold
                        if (
                            self._last_face_seen_ts is not None
                            and now - self._last_face_seen_ts >= self.no_face_timeout_sec
                        ):
                            self.deactivate()
                            break

                    # Establish global variable memory synchronization with atomic threading locks
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

                    # Manage UI frames locally if isolated standalone testing parameters are selected
                    if self.debug_window:
                        cv2.imshow("YOLOv3 DPU", frame)
                        cv2.waitKey(1)

                self.phase = "closing_camera"

                if self.cap is not None:
                    self.cap.release()
                    self.cap = None

                if self.debug_window:
                    cv2.destroyWindow("YOLOv3 DPU")

                with self.result_lock:
                    self.latest_frame = None

                self.phase = "idle"

        finally:
            # Emergency hardware resource cleanup routines
            if self.cap is not None:
                self.cap.release()
                self.cap = None

            if self.debug_window:
                try:
                    cv2.destroyWindow("YOLOv3 DPU")
                except cv2.error:
                    pass

            self.runner = None
            self.phase = "stopped"

    def stop(self):
        """
        Request cancellation of thread operations and unblock active event listeners.
        """
        self.stop_event.set()
        self.active_event.set()

        if self.is_alive():
            self.join(timeout=3.0)

```
