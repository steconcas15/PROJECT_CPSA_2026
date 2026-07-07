# lock.py

import threading

device_scan_lock = threading.Lock()
device_connection_lock = threading.Lock()
device_reconnection_lock = threading.RLock()
dpu_lock = threading.Lock()

logging_lock = threading.Lock()
