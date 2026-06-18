import threading
import time

# YOLO alpha high: trust detector more, correct quickly.
YOLO_SMOOTH_ALPHA = 0.85


class PersonRoiState:
    """
    Thread-safe shared person ROI (Region of Interest) state.

    Coordinates are always full-frame xyxy:
        (x1, y1, x2, y2)

    Internally, the ROI is smoothed in cx, cy, w, h space.

    The state is intended to be:
        - written by YOLO when a person is detected
    """

    def __init__(self):
        #Prevents any other thread from writing while we are reading information from YOLO
        self._lock = threading.Lock()

        # Latest raw bbox received from YOLO.
        self._raw_bbox_xyxy = None

        # Smoothed bbox returned to consumers.
        self._bbox_xyxy = None

        # Internal smoothed representation: cx, cy, w, h.
        self._smoothed_cxcywh = None

        self._confidence = 0.0
        self._timestamp = None

        self._need_reacquire = True

    # -----------------------------------------------------
    # PUBLIC UPDATE API
    # -----------------------------------------------------

    def update_from_yolo(self, bbox_xyxy, confidence):
        """
        Update ROI using a YOLO person detection.

        YOLO is treated as a stronger correction source, so it uses
        a higher smoothing alpha.
        """
        self._update(
            bbox_xyxy=bbox_xyxy,
            confidence=confidence,
            alpha=YOLO_SMOOTH_ALPHA
        )

    # -----------------------------------------------------
    # INTERNAL HELPERS
    # -----------------------------------------------------

    @staticmethod
    def _xyxy_to_cxcywh(bbox_xyxy):
        x1, y1, x2, y2 = bbox_xyxy  # 1. Unpack coordinates

        w = float(x2 - x1)          # 2. Calculate width
        h = float(y2 - y1)          # 3. Calculate height
        cx = float(x1) + 0.5 * w    # 4. Calculate center X
        cy = float(y1) + 0.5 * h    # 5. Calculate center Y

        return cx, cy, w, h         # 6. Return in the new format

    @staticmethod
    def _cxcywh_to_xyxy(cxcywh):
        cx, cy, w, h = cxcywh       # 1. Unpack coordinates

        x1 = cx - 0.5 * w           # 2. Calculate x1
        y1 = cy - 0.5 * h           # 3. Calculate y1
        x2 = cx + 0.5 * w           # 4. Calculate x2
        y2 = cy + 0.5 * h           # 5. Calculate y2

        return (                    # 6. Return as integer pixel coordinates
            int(round(x1)),
            int(round(y1)),
            int(round(x2)),
            int(round(y2))
        )

    
    @staticmethod
    def _is_valid_bbox(bbox_xyxy):
        # Returns True only if the bounding box has positive area and isn't None
        if bbox_xyxy is None:
            return False

        x1, y1, x2, y2 = bbox_xyxy
        
        # Ensure width and height are greater than zero to avoid invalid geometries
        return x2 > x1 and y2 > y1


    
    def _update(self, bbox_xyxy, confidence, alpha):
        # 1. Validation check: ignore bad data immediately
        if not self._is_valid_bbox(bbox_xyxy):
            return
        # 2. Extract raw coordinates    
        x1, y1, x2, y2 = bbox_xyxy
        # 3. Cast coordinates to integers and pack into a tuple
        raw_bbox = (
            int(x1),
            int(y1),
            int(x2),
            int(y2)
        )

        # 4. Convert xyxy format to center-based cxcywh for easier math
        new_cxcywh = self._xyxy_to_cxcywh(raw_bbox)

        # 5. Capture the current timestamp for age tracking
        now = time.monotonic()

        # 6. Thread-safe block: prevent concurrent reading/writing
        with self._lock:
            # 7. Initialize if it's the first detection
            if self._smoothed_cxcywh is None:
                smoothed = new_cxcywh
                
            else:
                # 8. Unpack current and previous state
                old_cx, old_cy, old_w, old_h = self._smoothed_cxcywh
                new_cx, new_cy, new_w, new_h = new_cxcywh

                # 9. Apply Exponential Moving Average (EMA) formula
                smoothed = (
                    alpha * new_cx + (1.0 - alpha) * old_cx,
                    alpha * new_cy + (1.0 - alpha) * old_cy,
                    alpha * new_w + (1.0 - alpha) * old_w,
                    alpha * new_h + (1.0 - alpha) * old_h,
                )

            # 10. Convert the smoothed result back to pixel coordinates
            smoothed_bbox = self._cxcywh_to_xyxy(smoothed)

            # 11. Final geometry check after smoothing
            if not self._is_valid_bbox(smoothed_bbox):
                return

            # 12. Update the internal state with new values
            self._raw_bbox_xyxy = raw_bbox
            self._bbox_xyxy = smoothed_bbox
            self._smoothed_cxcywh = smoothed
            self._confidence = float(confidence)
            self._timestamp = now
            self._need_reacquire = False

    # -----------------------------------------------------
    # PUBLIC READ API
    # -----------------------------------------------------

    def get_snapshot(self):
        """
        Returns a complete, thread-safe snapshot of the current ROI state.
        
        This method does not perform any age filtering or validation, making it 
        ideal for debugging, logging, or monitoring the system's performance. 
        It retrieves a consistent view of the raw and smoothed data, the 
        confidence score, and the internal status flags.
        
        """
        with self._lock:
            return {
                "bbox_xyxy": self._bbox_xyxy,
                "raw_bbox_xyxy": self._raw_bbox_xyxy,
                "confidence": self._confidence,
                "timestamp": self._timestamp,
                "need_reacquire": self._need_reacquire,
            }

    
    def get_valid_roi(self, max_age_sec=None):
        """
        Return smoothed bbox if available and not stale.

        Args:
            max_age_sec:
                If None, no age filtering is applied.
                Otherwise, bbox is returned only if recent enough.

        Returns:
            Smoothed (x1, y1, x2, y2) or None.
        """

        # 1. Thread-safe lock: protect state during validation
        with self._lock:
            # 2. Check if data exists
            if self._bbox_xyxy is None or self._timestamp is None:
                return None
            # 3. Check if we are in a 'reacquire' state (lost tracking)
            if self._need_reacquire:
                return None
            # 4. Age filtering: check if the data is too old (stale)
            if max_age_sec is not None:
                age = time.monotonic() - self._timestamp
                if age > max_age_sec:
                    return None
            
            return self._bbox_xyxy

    def mark_reacquire(self):
        """
        Mark current ROI as unreliable.

        The smoothed bbox is intentionally kept in memory, but get_valid_roi()
        will stop returning it until a new update clears need_reacquire.
        """
        with self._lock:
            self._need_reacquire = True

    def needs_reacquire(self):
        """
        Return True if YOLO should reacquire the person ROI.
        """
        with self._lock:
            return self._need_reacquire

    def clear(self):
        """
        Clear the ROI completely.
        """
        with self._lock:
            self._raw_bbox_xyxy = None
            self._bbox_xyxy = None
            self._smoothed_cxcywh = None
            self._confidence = 0.0
            self._timestamp = None
            self._need_reacquire = True
