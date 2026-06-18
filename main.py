# main.py
# Realtime Core Executive Layer for Drowsiness Detection Pipeline
#
# Author: Francesco Urru (Adapted for Drowsiness Core)
# Repository: https://github.com/frarvo/CPSA_2026
# License: MIT

import time

# Core hardware video (YOLOv3 DPU) e gestione ROI
from Video_Pipeline.Yolo_v3.yolo_v3_thread import YoloDpuThread
from Video_Pipeline.shared.person_roi_state import PersonRoiState

# Core logico (Dispatcher e Policy reale degli stimoli)
from core.event_dispatcher import EventDispatcher
from core.actuation_policy import DrowsinessActivationPolicy

# Core classificazione algoritmica e gestione sensori IMU BlueCoin
from IMU_pipeline.classifiers.drowsiness_classifier import DrowsinessClassifier
from sensors.sensor_manager import SensorManager

# Moduli di sistema centralizzati del Framework CPSA
from utils.actuator_manager import get_actuator_manager  
from utils.logger import log_system
from utils.config import get_bluecoin_config
from utils.video_dashboard import (
    VideoDashboard,
    register_dashboard_console,
    unregister_dashboard_console,
)


def main():
    sensor_manager = None
    dashboard = None
    yolo_thread = None
    dispatcher = None

    try:
        # 1. Inizializzazione Interfaccia Grafica (Dashboard OpenCV su schermo)
        dashboard = VideoDashboard(
            window_name="CPSA 2026 - Drowsiness Detection System",
            fullscreen=False
        )
        register_dashboard_console(dashboard)

        log_system("[MAIN] Inizializzazione del SensorManager...")
        sensor_manager = SensorManager()

        # 2. Configurazione del VERO DrowsinessClassifier all'interno del manager delle IMU
        sensor_manager.classifier = DrowsinessClassifier()
        sensor_manager.synchronizer.buffer.set_features_sink(sensor_manager.classifier.recognize)

        # 3. Scansione BLE e controllo presenza dei sensori BlueCoin configurati
        sensor_manager.scan_sensors()
        expected_names = {
            entry.get("name") for entry in get_bluecoin_config() if entry.get("name")
        }

        if expected_names:
            max_sensor_retries = 5
            retry_delay_sec = 3
            attempt = 0

            while not expected_names.issubset(set(sensor_manager.get_sensors_names())) and attempt < max_sensor_retries:
                attempt += 1
                log_system(f"[MAIN] Sensori BlueCoin attesi non trovati. Tentativo {attempt}/{max_sensor_retries}...")
                time.sleep(retry_delay_sec)
                sensor_manager.scan_sensors()

            if not expected_names.issubset(set(sensor_manager.get_sensors_names())):
                log_system("[MAIN] Errore Critico: Hardware BlueCoin non rilevato. Uscita dal sistema.", level="ERROR")
                return

        # 4. Avvio del Sensing Layer (I sensori iniziano a trasmettere pacchetti via BLE)
        sensor_manager.initialize_sensors()
        log_system("[MAIN] Pipeline IMU e modulo di filtraggio complementare online.")
        
        # 5. Inizializzazione dello stato ROI e avvio del thread hardware YOLOv3 sulla DPU
        roi_state = PersonRoiState()
        yolo_thread = YoloDpuThread(roi_state=roi_state, camera_index=0)
        yolo_thread.start() # Il thread parte in standby ('idle'), la telecamera fisica rimane spenta

        # 6. Configurazione della Policy reale per l'attivazione dei dispositivi di allarme
        # Inserisci gli ID precisi dei tuoi attuatori hardware (es. led cruscotto, buzzer, speaker)
        attuatori_sistema = ["led_cruscotto", "speaker_allarme"]
        drowsiness_policy = DrowsinessActivationPolicy(actuator_ids=attuatori_sistema)

        # 7. Avvio del Direttore d'Orchestra (EventDispatcher)
        # Sincronizza i dati estratti dalle IMU con l'accensione della telecamera e degli stimoli hardware
        dispatcher = EventDispatcher(
            actuator_manager=get_actuator_manager(),
            policy=drowsiness_policy,
            yolo_thread=yolo_thread,
            movenet_thread=None,  # Configurato a None in questa fase di test YOLO
            roi_state=roi_state
        )
        dispatcher.start()

        log_system("[MAIN] Sistema di controllo ad eventi attivo. In attesa di segnali dai sensori...")

        # 8. Loop principale di esecuzione: Aggiornamento Dashboard grafica a schermo
        while True:
            # Renderizza lo stato corrente (Mostra lo standby se il guidatore è sveglio, 
            # mostra il feed video in real-time se si attiva YOLO in seguito a un evento)
            dashboard.render(yolo_thread, None)

            # Cattura la chiusura del software tramite la pressione del tasto 'q' sulla finestra
            key = dashboard.wait_key(1)
            if key == ord("q"):
                log_system("[MAIN] Richiesta di chiusura intercettata da tastiera.")
                break

            time.sleep(0.01)

    except KeyboardInterrupt:
        log_system("[MAIN] Interruzione manuale del sistema rilevata (Ctrl+C).")
    except Exception as e:
        import traceback
        log_system("="*50, level="ERROR")
        log_system(f"[CRITICAL ERROR] Fallimento bloccante nel ciclo executive principale:", level="ERROR")
        log_system(traceback.format_exc(), level="ERROR")
        log_system("="*50, level="ERROR")
    finally:
        # 9. Pipeline di spegnimento sicuro e rilascio controllato di tutte le risorse
        log_system("[MAIN] Avvio procedure di arresto hardware e software in corso...")
        
        if dispatcher:
            dispatcher.stop()

        if sensor_manager:
            sensor_manager.stop_all()

        if yolo_thread:
            yolo_thread.stop()

        if dashboard:
            unregister_dashboard_console()
            dashboard.close()
            
        log_system("[MAIN] Tutti i moduli sono stati spenti correttamente. Risorse hardware rilasciate.")


if __name__ == "__main__":
    main()
