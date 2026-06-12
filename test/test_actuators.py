# test_metamotion_real.py
# Testa ActuatorManager + DrowsinessActivationPolicy + EventDispatcher + MetaMotionThread
# su HARDWARE REALE. Eseguire dalla root del progetto con: sudo python3 test_metamotion_real.py
#
# Prerequisiti:
#   - MetaMotionRL acceso e nelle vicinanze (LED che lampeggia)
#   - config.yaml con metamotion.enable: true

import time

from actuators.BLE.metamotion import scan_metamotion_devices, MetaMotionThread
from actuation_policy import DrowsinessActivationPolicy
from core.actuator_manager import ActuatorManager
from utils.event_queue import get_event_queue, enqueue_drop_oldest
from core.event_dispatcher import EventDispatcher
from utils.logger import log_system

# ---- 1. Scansione dispositivo reale ----
print("Scansione MetaMotion in corso (assicurati che sia acceso)...")
mac_list = scan_metamotion_devices(timeout=5)

if not mac_list:
    print("Nessun MetaMotion trovato. Controlla che sia acceso e nel raggio BLE.")
    exit(1)

MAC = mac_list[0]
print(f"Trovato dispositivo: {MAC}")

# ---- 2. Setup ActuatorManager ----
manager = ActuatorManager()
thread = MetaMotionThread(MAC)
manager.actuators[f"meta_{MAC}"] = thread
thread.start()

print("Attesa connessione (la doppia vibrazione confirma la connessione)...")
time.sleep(5)

# ---- 3. Setup policy e dispatcher ----
policy = DrowsinessActivationPolicy(actuator_ids=manager.get_actuators_ids())
policy.max_vibration_attempts = 2

dispatcher = EventDispatcher(actuator_manager=manager, policy=policy)
dispatcher.start()

# ---- 4. Scenario eventi finti ----
scenario = [
    (0, "Sveglio — nessuna azione"),
    (1, "Sospetto — nessuna azione"),
    (2, "Confermato — vibrazione #1 (60%, 800ms)"),
    (2, "Confermato — vibrazione #2 (80%, 1200ms)"),
    (2, "Confermato — passa a speaker (nessuno speaker configurato, skip)"),
    (0, "Sveglio — reset"),
    (2, "Confermato dopo reset — vibrazione #1 di nuovo"),
]

q = get_event_queue()
for tag, desc in scenario:
    print(f"\n→ tag={tag} | {desc}")
    enqueue_drop_oldest(q, {
        "drowsiness_tag": tag,
        "source": "test",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    })
    time.sleep(2.5)  # tempo per sentire/vedere la vibrazione

# ---- 5. Stop ----
print("\nArresto in corso...")
dispatcher.stop()
manager.stop_all()
print("Test completato.")
