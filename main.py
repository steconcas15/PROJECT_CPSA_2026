# main.py
# Realtime Core Executive Layer for Drowsiness Detection Pipeline
#
# Author: Francesco Urru (Adapted for Drowsiness Core)
# Repository: https://github.com/frarvo/CPSA_2026
# License: MIT

import time

# Core video hardware execution (YOLOv3 running on DPU hardware accelerator) and ROI management
from Video_Pipeline.Yolo_v3.yolo_v3_thread import YoloDpuThread
from Video_Pipeline.shared.person_roi_state import PersonRoiState

# Logic orchestration core (Event routing engine and actual actuation policies)
from core.event_dispatcher import EventDispatcher
from core.actuation_policy import DrowsyAlertPolicy

# Classification and data collection from STMicroelectronics BlueCoin IMU nodes
from IMU_pipeline.classifiers.drowsiness_classifier import DrowsinessClassifier
from sensors.sensor_manager import SensorManager

# Centralized system managers
from actuators.actuator_manager import ActuatorManager  
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
    actuator_manager = None

    try:
        # 1. Initialize UI (OpenCV Dashboard)
        dashboard = VideoDashboard(
            window_name="CPSA 2026 - Drowsiness Detection System",
            fullscreen=False
        )
        register_dashboard_console(dashboard)

        # 2. Initialize hardware and system managers
        log_system("[MAIN] Initializing SensorManager...")
        sensor_manager = SensorManager()
        
        actuator_manager = ActuatorManager()

        # Inject the Drowsiness Classifier into the IMU pipeline
        sensor_manager.classifier = DrowsinessClassifier()
        sensor_manager.synchronizer.buffer.set_features_sink(sensor_manager.classifier.recognize)

        # 3. BLE Scan & Retry loop for BlueCoin sensors
        sensor_manager.scan_sensors()
        expected_names = {
            entry.get("name") for entry in get_bluecoin_config() if entry.get("name")
        }

        if expected_names:
            max_sensor_retries = 5
            retry_delay_sec = 3
            attempt = 0

            # Wait until all configured BlueCoin sensors are found or max retries reached
            while not expected_names.issubset(set(sensor_manager.get_sensors_names())) and attempt < max_sensor_retries:
                attempt += 1
                log_system(f"[MAIN] Expected BlueCoin sensors missing. Retrying scan {attempt}/{max_sensor_retries}...")
                time.sleep(retry_delay_sec)
                sensor_manager.scan_sensors()
                
            # Exit if required sensors are not found
            if not expected_names.issubset(set(sensor_manager.get_sensors_names())):
                log_system("[MAIN] Critical Error: BlueCoin hardware nodes not detected. Exiting.", level="ERROR")
                return

        # Scan and initialize output actuators (speaker)
        actuator_manager.scan_actuators()

        actuator_manager.initialize_actuators()
        
        actuators_list = actuator_manager.get_actuators_ids()  
        
        # 4. Start Sensing Layer (Sensors start streaming IMU data over BLE)
        sensor_manager.initialize_sensors()
        log_system("[MAIN] IMU data pipeline online.")
        
        if not actuators_list:
            log_system("[MAIN] No actuators discovered. Event detection and logging still executing")
            
        # Initialize the alert policy with available actuators    
        policy = DrowsyAlertPolicy(actuator_ids=actuators_list)
        
        # 5. Initialize ROI state and start the YOLOv3 DPU thread
        roi_state = PersonRoiState()
        yolo_thread = YoloDpuThread(roi_state=roi_state)
        yolo_thread.start() # Starts in 'idle' mode, physical camera stays off for now

        # 6. Start the Orchestrator (EventDispatcher)
        # Connects IMU events with camera activation and hardware triggers
        dispatcher = EventDispatcher(
            actuator_manager=actuator_manager,
            policy=policy,
            yolo_thread=yolo_thread,
            roi_state=roi_state
        )
        dispatcher.start()

        log_system("[MAIN] Event loop active. Monitoring incoming sensor data...")

        # 7. Main execution loop (UI refresh and keyboard input polling)o
        while True:
            # Renders standby screen when alert, switches to camera feed if YOLO triggers
            dashboard.render(yolo_thread)

            # Exit software if 'q' is pressed on the dashboard window
            key = dashboard.wait_key(1)
            if key == ord("q"):
                log_system("[MAIN] Shutdown requested via keyboard.")
                break

            time.sleep(0.01)

    except KeyboardInterrupt:
        log_system("[MAIN] Manual interruption detected (Ctrl+C).")
    except Exception as e:
        import traceback
        log_system("="*50, level="ERROR")
        log_system(f"[CRITICAL ERROR] Pipeline failure in main loop:", level="ERROR")
        log_system(traceback.format_exc(), level="ERROR")
        log_system("="*50, level="ERROR")
    finally:
        # 8. Safe hardware teardown and resource release
        log_system("[MAIN] Initiating clean software and hardware shutdown...")
        
        if dispatcher:
            dispatcher.stop()

        if sensor_manager:
            sensor_manager.stop_all()

        if actuator_manager:
            actuator_manager.stop_all()

        if yolo_thread:
            yolo_thread.stop()

        if dashboard:
            unregister_dashboard_console()
            dashboard.close()
            
        log_system("[MAIN] All modules stopped. Hardware resources released.")


if __name__ == "__main__":
    main()
