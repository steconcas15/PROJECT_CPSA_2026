# Cyber-Physical-Systems for Real-Time Driver Drowsiness Detection and Road Safety

### Project Structure

```text
📂 CPSA_2026/
 ┃
 ┣ 📂 DPU_FIRMWARE/
 ┃ ┗ 📜 kv260-benchmark-b4096.xclbin
 ┃
 ┣ 📂 IMU_pipeline/
 ┃ ┣ 📂 classifiers/
 ┃ ┃ ┗ 📜 drowsiness_classifier.py
 ┃ ┗ 📂 data_stream/
 ┃   ┣ 📜 data_buffer.py
 ┃   ┣ 📜 data_processing_wrapper_quat.py
 ┃   ┗ 📜 synchronizer.py
 ┃
 ┣ 📂 Video_Pipeline/
 ┃ ┣ 📂 Resnet18/
 ┃ ┃ ┗  📜 kv260_train_resnet18_drowsy.xmodel 
 ┃ ┣ 📂 Yolo_v3/
 ┃ ┃ ┣ 📜 pynqdpu.tf_yolov3_voc.DPUCZDX8G_ISA1_B4096.2.5.0.xmodel
 ┃ ┃ ┗ 📜 yolo_v3_thread.py
 ┃ ┗ 📂 shared/
 ┃   ┗ 📜 person_roi_state.py
 ┃
 ┣ 📂 actuators/
 ┃ ┣ 📂 BT/
 ┃ ┃ ┗ 📜 speaker.py
 ┃ ┗ 📜 actuator_manager.py
 ┃
 ┣ 📂 core/
 ┃ ┣ 📜 actuation_policy.py
 ┃ ┗ 📜 event_dispatcher.py
 ┃
 ┣ 📂 sensors/
 ┃ ┣ 📂 BLE/
 ┃ ┃ ┣ 📜 bluecoin.py
 ┃ ┃ ┣ 📜 feature_listeners.py
 ┃ ┃ ┗ 📜 feature_mems_sensor_fusion_compact.py
 ┃ ┗ 📜 sensor_manager.py
 ┃
 ┣ 📂 test/
 ┃ ┗ 📜 test_actuators.py
 ┃
 ┣ 📂 utils/
 ┃ ┣ 📂 __pycache__/
 ┃ ┣ 📜 config.py
 ┃ ┣ 📜 event_queue.py
 ┃ ┣ 📜 lock.py
 ┃ ┣ 📜 logger.py
 ┃ ┗ 📜 video_dashboard.py
 ┃
 ┣ 📜 LICENSE
 ┣ 📜 README.md
 ┣ 📜 config.yaml
 ┣ 📜 main.py
 ┣ 📜 test_actuators.py
 ┣ 📜 trash.py
 ┗ 📜 xmutil_load_dpu.sh
