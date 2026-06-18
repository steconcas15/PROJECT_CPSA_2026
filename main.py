# test_yolo.py
import time

from Video_Pipeline.Yolo_v3.yolo_v3_thread import YoloDpuThread
from Video_Pipeline.shared.person_roi_state import PersonRoiState

from IMU_pipeline.policies.actuation_policy import StereotipyActivationPolicy
from IMU_pipeline.dispatchers.event_dispatcher import EventDispatcher

from sensors import sensor_manager
from sensors.sensor_manager import SensorManager
from utils.logger import log_system
from utils.config import get_bluecoin_config

from utils.logger import log_system
# Nota: verifica che all'interno di utils ci sia effettivamente il file video_dashboard.py. 
# Se nello screenshot non si vede perché la cartella è contratta, l'import corretto è questo:
from utils.video_dashboard import (
    VideoDashboard,
    register_dashboard_console,
    unregister_dashboard_console,
)

# Classi momentanee per testare il funzionamento della fotocamera
class MockPolicy:
    def handle(self, event): return None
class MockActuatorManager:
    def trigger(self, *args, **kwargs): pass

def main():
    sensor_manager = None
    dashboard = None
    yolo_thread = None

    try:
        # Inizializza la Dashboard grafica (la finestra OpenCV)
        dashboard = VideoDashboard(
            window_name="CPSA Dashboard - SOLO YOLO TEST",
            fullscreen=False
        )
        register_dashboard_console(dashboard)

        log_system("[TEST] Initializing sensors and minimal YOLO test system...")
        sensor_manager = SensorManager()

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

        # 3. Avvio dei thread di ricezione dati (Sensing Layer)
        sensor_manager.initialize_sensors()

        log_system("[MAIN] IMU Pipeline and Drowsiness Classifier are now fully running.")
        log_system("[MAIN] Monitoring streaming data... Press Ctrl+C to stop.")
        
        # Crea la struttura ROI di cui YOLO ha bisogno per memorizzare i dati
        roi_state = PersonRoiState()

        # Istanzia il thread di YOLO passandogli la ROI
        yolo_thread = YoloDpuThread(roi_state=roi_state)

        # Avvia il thread (il motore asincrono si accende in background)
        yolo_thread.start()

        log_system("[TEST] YOLO System is running. Press 'q' inside the window to exit.")
        
        dispatcher = EventDispatcher(
            actuator_manager=MockActuatorManager(),
            policy=MockPolicy(),
            yolo_thread=yolo_thread,
            movenet_thread=None, # In questo test usiamo solo YOLO
            roi_state=roi_state
        )
        dispatcher.start()

        log_system("[TEST] System connected via EventDispatcher. Waiting for IMU triggers...")

        

        # Loop principale: rendering grafico continuo
        while True:
            # Passiamo 'None' al posto di movenet_thread così la dashboard disegna solo YOLO
            dashboard.render(yolo_thread, None)

            # Ascolta la tastiera per catturare la chiusura
            key = dashboard.wait_key(1)
            if key == ord("q"):
                log_system("[TEST] Quit requested via keyboard.")
                break

            time.sleep(0.01)

    except KeyboardInterrupt:
        log_system("[TEST] Termination signal received.")

    except Exception as e:
        log_system(f"[TEST] Unhandled error: {e}", level="ERROR")

    finally:
        log_system("[TEST] Shutting down YOLO test...")
        
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
