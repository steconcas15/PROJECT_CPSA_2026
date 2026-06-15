# test_yolo.py
import time

from Video_Pipeline.Yolo_v3.yolo_v3_thread import YoloDpuThread
from Video_Pipeline.shared.person_roi_state import PersonRoiState

from utils.logger import log_system
# Nota: verifica che all'interno di utils ci sia effettivamente il file video_dashboard.py. 
# Se nello screenshot non si vede perché la cartella è contratta, l'import corretto è questo:
from utils.video_dashboard import (
    VideoDashboard,
    register_dashboard_console,
    unregister_dashboard_console,
)

def main():
    dashboard = None
    yolo_thread = None

    try:
        # Inizializza la Dashboard grafica (la finestra OpenCV)
        dashboard = VideoDashboard(
            window_name="CPSA Dashboard - SOLO YOLO TEST",
            fullscreen=False
        )
        register_dashboard_console(dashboard)

        log_system("[TEST] Initializing minimal YOLO test system...")

        # Crea la struttura ROI di cui YOLO ha bisogno per memorizzare i dati
        roi_state = PersonRoiState()

        # Istanzia il thread di YOLO passandogli la ROI
        yolo_thread = YoloDpuThread(roi_state=roi_state)

        # Avvia il thread (il motore asincrono si accende in background)
        yolo_thread.start()
        
        # Sveglia il thread dallo stato 'idle' per forzare l'apertura della cam
        yolo_thread.activate()

        log_system("[TEST] YOLO System is running. Press 'q' inside the window to exit.")

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
        
        if yolo_thread:
            yolo_thread.stop()

        if dashboard:
            unregister_dashboard_console()
            dashboard.close()
            
        log_system("[TEST] Shutdown complete.")

if __name__ == "__main__":
    main()
