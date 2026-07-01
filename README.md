# Cyber-Physical-Systems for Real-Time Driver Drowsiness Detection and Road Safety

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
