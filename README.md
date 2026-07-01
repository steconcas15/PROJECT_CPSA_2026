# Cyber-Physical-Systems for Real-Time Driver Drowsiness Detection and Road Safety

### System View

* BLE BlueCoin acquisition for driver monitoring.
* Accelerometer and gyroscope used to detect involuntary head falls.
* IMU synchronization and timestamp alignment via dedicated synchronizer module.
* Sliding-window buffering with configurable overlap to stream sensor features.
* Drowsiness classifier integrated directly into the IMU pipeline.
* Event queue with dispatching operations.
* The event dispatcher manages synchronization between IMU events, YOLO activation and hardware triggers.
* Staged DPU video pipeline:
  * Hardware-accelerated YOLOv3 person/driver detection on the Xilinx DPU.
  * ResNet18 classification developed for drowsiness and driver state estimation.
  * Dynamically updated bounding box status managed via a PersonRoiState.
* The dashboard GUI renders runtime states (standby/active feed) and handles user termination.
* Modular actuator manager supporting:
  * Bluetooth (BT) speaker feedback orchestration.
* Drowsiness alert policy with feedback deployment based on active actuator ID.
* Centralized shutdown of dispatcher, video threads, sensors, actuators, and dashboard resources

### Runtime Flow
```text
1. Create the VideoDashboard and register the dashboard console.
2. Create SensorManager and ActuatorManager.
3. Instantiate DrowsinessClassifier.
4. Scan BLE sensor and read expected BlueCoin name from config.yaml.
5. Retry BlueCoin discovery up to 5 times if expected device is missing (exit on failure).
6. Scan for speaker actuator and initialize it.
7. Read discovered actuator ID from the ActuatorManager.
8. Initialize and start sensor thread.
9. Create PersonRoiState.
10. Create YoloDpuThread (passing the roi_state).
11. Create DrowsyAlertPolicy using the discovered actuator ID.
12. Create EventDispatcher with actuator manager, policy, YOLO thread, and roi_state.
13. Start the EventDispatcher thread.
14. Enter the main dashboard render loop.
15. Loop until 'q' is pressed in the GUI or a KeyboardInterrupt (Ctrl+C) is received.
16. Cleanly stop dispatcher, sensor manager, actuator manager, video thread, and unregister dashboard resources.
```

### Important Runtime Details

* The DPU overlay must be loaded before `main.py` starts. Use `bash xmutil_load_dpu.sh` at system startup to program the FPGA fabric on the KV260 board.
* `main.py` is the direct runtime entry point.
* The `YoloDpuThread` is started at boot but stays idle, keeping the physical camera interface suspended until the `EventDispatcher` activates it upon event detection.
* The `VideoDashboard` is responsible for the main execution loop, rendering frames and reading GUI key inputs (e.g., pressing `'q'` to gracefully terminate).
* The system aborts startup completely if the expected BlueCoin devices configured in `config.yaml` are not discovered after the 5-retry loop window.
* If no hardware actuator is discovered, the core event detection, IMU classification pipelines, and logging systems will still execute normally.

### Project Structure 
```text
CPSA_2026/
├── DPU_FIRMWARE/
│   └── kv260-benchmark-b4096.xclbin
│
├── IMU_pipeline/
│   ├── classifiers/
│   │   └── drowsiness_classifier.py
│   └── data_stream/
│       ├── data_buffer.py
│       ├── data_processing_wrapper_quat.py
│       └── synchronizer.py
│
├── Video_Pipeline/
│   ├── Resnet18/
│   │   └── kv260_train_resnet18_drowsy.xmodel
│   ├── Yolo_v3/
│   │   ├── pynqdpu.tf_yolov3_voc.DPUCZDX8G_ISA1_B4096.2.5.0.xmodel
│   │   └── yolo_v3_thread.py
│   └── shared/
│       └── person_roi_state.py
│
├── actuators/
│   ├── BT/
│   │   └── speaker.py
│   └── actuator_manager.py
│
├── assets/
│   └── audio/
│       ├── beep_beep.mp3
│       └── speaker_connected.mp3
│
├── core/
│   ├── actuation_policy.py
│   └── event_dispatcher.py
│
├── sensors/
│   ├── BLE/
│   │   ├── bluecoin.py
│   │   ├── feature_listeners.py
│   │   └── feature_mems_sensor_fusion_compact.py
│   └── sensor_manager.py
│
├── utils/
│   ├── audio_paths.py
│   ├── config.py
│   ├── event_queue.py
│   ├── lock.py
│   ├── logger.py
│   └── video_dashboard.py
│
├── LICENSE
├── README.md
├── config.yaml
├── main.py
├── test_actuators.py
├── trash.py
└── xmutil_load_dpu.sh
```
