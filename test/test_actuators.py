# test_metamotion_simple.py
# Testa ActuatorManager + DrowsinessActivationPolicy + EventDispatcher + MetaMotionThread
# usando eventi finti. Eseguire dalla root del progetto con: sudo python3 test_metamotion_simple.py

import time
import threading
from unittest.mock import MagicMock, patch

# ---- Mock BLE/MetaWear (rimuovere se si usa hardware reale) ----
fake_device = MagicMock()
fake_device.is_connected = True
fake_device.board = MagicMock()

vibration_log = []
def fake_haptic(board, duty, duration):
    vibration_log.append({"duty": duty, "duration": duration})
    print(f"  🔴 [HW] vibrazione duty={duty}% duration={duration}ms")

with patch("bluepy.btle.Scanner") as mock_scanner, \
     patch("mbientlab.metawear.MetaWear", return_value=fake_device), \
     patch("mbientlab.metawear.libmetawear") as mock_lib, \
     patch("mbientlab.warble.WarbleException", Exception):

    mock_scanner.return_value.scan.return_value = [
        MagicMock(addr="AA:BB:CC:DD:EE:FF", getValueText=lambda x: "MetaWear")
    ]
    mock_lib.mbl_mw_haptic_start_motor.side_effect = fake_haptic


    from actuators.BLE.metamotion import MetaMotionThread
    from core.actuation_policy import DrowsinessActivationPolicy
    from actuators.actuator_manager import ActuatorManager
    from utils.event_queue import get_event_queue, enqueue_drop_oldest
    from core.event_dispatcher import EventDispatcher

    # ---- Setup ----
    MAC = "AA:BB:CC:DD:EE:FF"

    manager = ActuatorManager()
    manager.actuators[f"meta_{MAC}"] = MetaMotionThread(MAC)
    manager.actuators[f"meta_{MAC}"].start()
    time.sleep(1)  # attesa connessione

    policy = DrowsinessActivationPolicy(actuator_ids=manager.get_actuators_ids())
    policy.max_vibration_attempts = 2

    dispatcher = EventDispatcher(actuator_manager=manager, policy=policy)
    dispatcher.start()

    # ---- Scenario eventi finti ----
    scenario = [
        (0, "Sveglio — nessuna azione"),
        (1, "Sospetto — nessuna azione"),
        (2, "Confermato — vibrazione #1"),
        (2, "Confermato — vibrazione #2"),
        (2, "Confermato — passa a speaker (nessuno speaker, skip)"),
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
        time.sleep(1.5)

    # ---- Stop ----
    dispatcher.stop()
    manager.stop_all()

    # ---- Riepilogo ----
    print(f"\n{'─'*50}")
    print(f"Vibrazioni inviate all'HW: {len(vibration_log)}")
    for i, v in enumerate(vibration_log, 1):
        print(f"  [{i}] duty={v['duty']}%, duration={v['duration']}ms")
    print("Test completato.")
