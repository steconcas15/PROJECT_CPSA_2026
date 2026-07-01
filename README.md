# Cyber-Physical-Systems for Real-Time Driver Drowsiness Detection and Road Safety

### 📂 Struttura del Progetto

```text
📂 CPSA_2026/
 ┣ 📂 actuators/
 ┃ ┣ 📜 actuator_manager.py
 ┃ ┣ 📂 BLE/
 ┃ ┃ ┗ 📜 metamotion.py
 ┃ ┣ 📂 BT/
 ┃ ┃ ┗ 📜 speaker.py
 ┃ ┗ 📂 WIFI/
 ┃   ┗ 📜 led_strip.py
 ┣ 📂 assets/
 ┃ ┗ 📂 audio/
 ┃   ┗ 🎵 *.mp3
 ┣ 📂 core/
 ┃ ┣ 📜 actuation_policy.py
 ┃ ┗ 📜 event_dispatcher.py
 ┣ 📂 IMU_pipeline/
 ┃ ┣ 📂 classifiers/
 ┃ ┃ ┗ 📂 stereotipy_classifier/
 ┃ ┃   ┣ 📜 stereotipy_classifier.py
 ┃ ┃   ┣ 📜 predict_models_wrapper_quat.py
 ┃ ┃   ┣ ⚙️ libPredictPericolosaWristsQuat.so
 ┃ ┃   ┗ 📂 Predict_Pericolosa_Wrists_Quat/
 ┃ ┗ 📂 data_stream/
 ┃   ┣ 📜 synchronizer.py
 ┃   ┣ 📜 data_buffer.py
 ┃   ┣ 📜 data_processing_wrapper_quat.py
 ┃   ┣ ⚙️ libProcessDataWristsQuat.so
 ┃   ┗ 📂 ProcessDataWristsQuat/
 ┗ 📂 sensors/
   ┣ 📜 sensor_manager.py
   ┗ 📂 BLE/
     ┣ 📜 bluecoin.py
     ┗ 📜 feature_listeners.py
