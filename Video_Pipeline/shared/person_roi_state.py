"""
Module: person_roi_state.py

This class manages the "bounding box" that surrounds a 
person detected by a YOLO model.

Scope:
    1. Smoother movement: Without this code, the box on the screen would 
       "jump" or shake around. We use the Exponential Moving Average (EMA) 
       technique to make the movement look natural and smooth.
    2. Time management: If the camera stops seeing the person, the system 
       won't keep showing old, wrong information.
    3. Data protection: Because the video analysis runs in the background 
       at the same time as other parts of the program, we use a "Lock." 
       This prevents errors that happen when two parts of the program 
       try to change the same data at the same time.
"""

import threading
import time

# YOLO alpha high: trust detector more, correct quickly.
YOLO_SMOOTH_ALPHA = 0.85


class PersonRoiState:
    """
    Manages the state and smoothing of a person's Bounding Box (ROI) 
    in a thread-safe manner.
    
    Coordinates are always full-frame xyxy:
        (x1, y1, x2, y2)

    Internally, the ROI is smoothed in cx, cy, w, h space

    The state is intended to be:
        - written by YOLO when a person is detected
    """

    def __init__(self):
        # Lock to ensure thread safety between detection (writer) and usage (reader)
        self._lock = threading.Lock()

        # Latest raw bbox received from the YOLO detector
        self._raw_bbox_xyxy = None

        # Smoothed bbox returned to consumers
        self._bbox_xyxy = None

        # Internal representation for smoothing: center coordinates (cx, cy) and size (w, h)
        self._smoothed_cxcywh = None

        self._confidence = 0.0
        self._timestamp = None

        # Flag to indicate if the object is lost or needs re-detection
        self._need_reacquire = True

    # -----------------------------------------------------
    # PUBLIC UPDATE API
    # -----------------------------------------------------

    def update_from_yolo(self, bbox_xyxy, confidence):
        """
        Public method to inject new YOLO detection data into the state.
        Uses the predefined alpha constant for consistency.
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
        """Converts coordinate format [x1, y1, x2, y2] to [center_x, center_y, width, height]."""
        x1, y1, x2, y2 = bbox_xyxy

        w = float(x2 - x1)
        h = float(y2 - y1)
        cx = float(x1) + 0.5 * w
        cy = float(y1) + 0.5 * h

        return cx, cy, w, h

    @staticmethod
    def _cxcywh_to_xyxy(cxcywh):
        """Converts [center_x, center_y, width, height] back to [x1, y1, x2, y2]."""
        cx, cy, w, h = cxcywh

        x1 = cx - 0.5 * w
        y1 = cy - 0.5 * h
        x2 = cx + 0.5 * w
        y2 = cy + 0.5 * h

        return (
            int(round(x1)),
            int(round(y1)),
            int(round(x2)),
            int(round(y2))
        )

    @staticmethod
    def _is_valid_bbox(bbox_xyxy):
        """Validates that the bbox exists and has positive dimensions."""
        if bbox_xyxy is None:
            return False

        x1, y1, x2, y2 = bbox_xyxy

        return x2 > x1 and y2 > y1

    def _update(self, bbox_xyxy, confidence, alpha):
        """Performs the internal EMA math and updates the state under a thread lock."""
        if not self._is_valid_bbox(bbox_xyxy):
            return

        x1, y1, x2, y2 = bbox_xyxy

        raw_bbox = (
            int(x1),
            int(y1),
            int(x2),
            int(y2)
        )

        new_cxcywh = self._xyxy_to_cxcywh(raw_bbox)

        now = time.monotonic()

        with self._lock:
            if self._smoothed_cxcywh is None:
                # Initialize with the first valid detection
                smoothed = new_cxcywh
            else:
                # Calculate EMA: New State = (Alpha * New) + ((1 - Alpha) * Old)
                old_cx, old_cy, old_w, old_h = self._smoothed_cxcywh
                new_cx, new_cy, new_w, new_h = new_cxcywh

                smoothed = (
                    alpha * new_cx + (1.0 - alpha) * old_cx,
                    alpha * new_cy + (1.0 - alpha) * old_cy,
                    alpha * new_w + (1.0 - alpha) * old_w,
                    alpha * new_h + (1.0 - alpha) * old_h,
                )

            smoothed_bbox = self._cxcywh_to_xyxy(smoothed)

            if not self._is_valid_bbox(smoothed_bbox):
                return

            # Commit the update to state
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
        """Returns the complete current state for diagnostics or logging."""
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
        Retrieves the smoothed bbox only if it is currently valid and not expired.
        'max_age_sec' acts as a TTL (Time-To-Live) for the detection.
        """
        with self._lock:
            if self._bbox_xyxy is None or self._timestamp is None:
                return None

            if self._need_reacquire:
                return None

            if max_age_sec is not None:
                age = time.monotonic() - self._timestamp
                if age > max_age_sec:
                    return None

            return self._bbox_xyxy

    def mark_reacquire(self):
        """Signals that the current track is lost/unreliable."""
        with self._lock:
            self._need_reacquire = True

    def needs_reacquire(self):
        """Checks if the system is currently looking for a new detection."""
        with self._lock:
            return self._need_reacquire

    def clear(self):
        """Resets the state to default values."""
        with self._lock:
            self._raw_bbox_xyxy = None
            self._bbox_xyxy = None
            self._smoothed_cxcywh = None
            self._confidence = 0.0
            self._timestamp = None
            self._need_reacquire = True
