# lock.py
#
# Author: Stefano Concas, Matteo Matta, Sara Cadedo, Fabio Piras
# Repository: https://github.com/steconcas15/PROJECT_CPSA_2026

import threading

device_scan_lock = threading.Lock()
device_connection_lock = threading.Lock()
device_reconnection_lock = threading.RLock()
dpu_lock = threading.Lock()

logging_lock = threading.Lock()
