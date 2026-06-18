import time

from Video_Pipeline.Yolo_v3.yolo_v3_thread import YoloDpuThread
from Video_Pipeline.shared.person_roi_state import PersonRoiState

# 1. IMPORTA I MODULI REALI CORRETTI
from IMU_pipeline.classifiers.drowsiness_classifier import DrowsinessClassifier
from IMU_pipeline.policies.actuation_policy import StereotipyActivationPolicy
from core.event_dispatcher import EventDispatcher

from sensors.sensor_manager import SensorManager
from utils.logger import log_system
from utils.config import get_bluecoin_config

from utils.video_dashboard import (
    VideoDashboard,
    register_dashboard_console,
    unregister_dashboard_console,
)

# Rimane momentaneo solo il manager degli attuatori fisici se non lo hai ancora centralizzato
class MockActuatorManager:
    def trigger(self, actuator_id, action_type, **kwargs):
        log_system(f"[MOCK ACTUATOR] 🚨 ATTIVAZIONE FISICA: {actuator_id} -> {kwargs}")

def main():
    sensor_manager = None
    dashboard = None
    yolo_thread = None
    dispatcher = None

    try:
        # Inizializza la Dashboard grafica (la finestra OpenCV)
        dashboard = VideoDashboard(
            window_name="CPSA Dashboard - YOLO TRIGGER TEST",
            fullscreen=False
        )
        register_dashboard_console(dashboard)

        log_system("[TEST] Initializing sensors and YOLO test system...")
        sensor_manager = SensorManager()

        # ── EVENTO CORREZIONE 1: AGGANCIA IL VERO CLASSIFICATORE AL MANAGER ──
        # Se non fai questo, il SensorManager userà il vecchio classificatore di stereotipie!
        sensor_manager.classifier = DrowsinessClassifier()
        sensor_manager.synchronizer.buffer.set_features_sink(sensor_manager.classifier.recognize)

        sensor_manager.scan_sensors()

        expected_names = {
            entry.get("name")
            for entry in get_bluecoin_config()
            if entry.get("name")
        }

        if expected_names:
            max_sensor_retries = 5
            retry_delay_sec = 5
            attempt = 0

            def actual_sensors():
                return set(sensor_manager.get_sensors_names())

            while not expected_names.issubset(actual_sensors()) and attempt < max_sensor_retries:
                attempt += 1
                log_system(f"[MAIN] Missing expected sensors. Retry {attempt}/{max_sensor_retries} in {retry_delay_sec}s...")
                time.sleep(retry_delay_sec)
                sensor_manager.scan_sensors()

            if not expected_names.issubset(actual_sensors()):
                log_system("[MAIN] Critical: Target BlueCoin sensors not found. Exiting.", level="ERROR")
                return

        # Avvo dei thread di ricezione dati dei sensori BlueCoin
        sensor_manager.initialize_sensors()

        log_system("[MAIN] IMU Pipeline and Drowsiness Classifier are now fully running.")
        
        # Crea la struttura ROI di cui YOLO ha bisogno per memorizzare i dati
        roi_state = PersonRoiState()

        # Istanzia il thread di YOLO passandogli la ROI
        yolo_thread = YoloDpuThread(roi_state=roi_state)
        yolo_thread.start()

        log_system("[TEST] YOLO System is running.")
        
        # ── EVENTO CORREZIONE 2: ISTANZIA LA VERA POLICY CON IL METODO HANDLE ──
        # Specifichiamo gli attuatori da usare. Questa classe ha il metodo .handle() richiesto!
        attuatori_selezionati = ["led_cruscotto", "speaker_allarme"]
        drowsiness_policy = StereotipyActivationPolicy(actuator_ids=attuatori_selezionati)

        # Istanzia l'EventDispatcher passando i moduli corretti
        dispatcher = EventDispatcher(
            actuator_manager=MockActuatorManager(),
            policy=drowsiness_policy, # <── Adesso passiamo l'oggetto reale e funzionante
            yolo_thread=yolo_thread,
            movenet_thread=None,
            roi_state=roi_state
        )
        dispatcher.start()

        log_system("[TEST] System connected via EventDispatcher. Waiting for IMU triggers...")

        # Loop principale: rendering grafico continuo
        while True:
            dashboard.render(yolo_thread, None)

            key = dashboard.wait_key(1)
            if key == ord("q"):
                log_system("[TEST] Quit requested via keyboard.")
                break

            time.sleep(0.01)

    except KeyboardInterrupt:
        log_system("[TEST] Termination signal received.")

    except Exception as e:
        import traceback
        log_system(f"[TEST] Unhandled error:", level="ERROR")
        log_system(traceback.format_exc(), level="ERROR")

    finally:
        log_system("[TEST] Shutting down YOLO test...")
        
        if dispatcher:
            dispatcher.stop()
        
        if sensor_manager:
            sensor_manager.stop_all()

        if yolo_thread:
            yolo_thread.stop()

        if dashboard:
            unregister_dashboard_console()
            dashboard.close()
            
        log_system("[TEST] Shutdown complete.")

if __name__ == "__main__":
    main()
