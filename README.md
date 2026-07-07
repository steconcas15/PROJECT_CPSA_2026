# Cyber-Physical-System for Real-Time Driver Drowsiness Detection and Road Safety

## Table of Contents

* [System View](#system-view)
* [Architecture and Working Principle](#architecture-and-working-principle)
  * [System Architecture](#system-architecture)
  * [Runtime Flow](#runtime-flow)
  * [Important Runtime Details](#important-runtime-details)
  * [Project Structure](#project-structure)
* [Python Dependencies and Environment Setup](#python-dependencies-and-environment-setup)
  * [Core Python Modules Used](#core-python-modules-used)
  * [Installation Commands](#installation-commands)
  * [DPU / Vitis-AI Runtime Setup](#dpu--vitis-ai-runtime-setup)
  * [Configuration](#configuration)
  * [Run Procedure](#run-procedure)
* [IMU Pipeline](#imu-pipeline)
  * [Architecture](#architecture)
  * [Data Flow](#data-flow)
  * [Configuration and tuning parameters](#configuration-and-tuning-parameters)
* [Video Pipeline](#video-pipeline)
  * [YOLO & ResNet18](#yolo--resnet18)
* [Actuation Policy](#actuation-policy)
* [Event System and Dispatcher](#event-system-and-dispatcher)
  * [Anti-Blink Filter Mechanism](#anti-blink-filter-mechanism)
* [Dashboard](#dashboard)
* [Logging](#logging)

---

## System View

* BLE BlueCoin acquisition for real-time driver state monitoring.
* Acceleration and gyroscope data stream processing to identify involuntary head falls.
* IMU synchronization and timestamp alignment via a dedicated synchronizer module.
* Sliding-window buffering with configurable overlap to continuously stream sensor features.
* Drowsiness classifier integrated directly into the IMU data pipeline.
* Centralized event queue with dispatching operations to manage synchronization between IMU events, YOLO activation, and hardware triggers.
* Multi-stage DPU video pipeline featuring hardware-accelerated YOLOv3 person detection and ResNet18 drowsiness classification.
* Dynamic bounding box and state estimation management via the `PersonRoiState` module.
* Dashboard GUI to render runtime execution states (standby/active feed) and capture user termination inputs.
* Alert policy for Bluetooth (BT) speaker feedback.
* Centralized, clean shutdown sequence for the dispatcher, video threads, sensors, actuators, and dashboard resources.

---

## Architecture and Working Principle

The system is organized around an IMU-first event loop. A BlueCoin device provides motion data to monitor driver state. The classifier produces drowsiness tags. The dispatcher consumes the latest tag, turns on video stage if needed, and asks the actuation policy to select feedback behavior. The dashboard owns video rendering and termination input.

---

### System Architecture

```text
BlueCoin BLE Sensors
       │
       ▼
 SensorManager
       │
       ▼
Feature Listeners
       │
       ▼
  Synchronizer
       │
       ▼
   DataBuffer
       │
       ▼
DrowsinessClassifier
       │
       ▼
  Event Queue
       │
       ▼
EventDispatcher <─────────────────────────────────┐
       │                                          │
       ├───> YOLO DPU Thread (Standby / Active)   │ (Video Prediction:
       │                   │                      │  "DROWSY" / "ALERT")
       ▼                   ▼                      │
DrowsyAlertPolicy   ResNet18 Inference ───────────┘
       │
       ▼
ActuatorManager
       │
       └─► Bluetooth Speaker
```

---

### Runtime Flow
1. **Initialize the UI:** Create the `VideoDashboard` and register the dashboard console.
2. **Initialize Managers:** Create the `SensorManager` and `ActuatorManager`.
3. **Configure Classifier:** Instantiate the `DrowsinessClassifier` and link its recognition method as the data sink for the IMU synchronizer buffer.
4. **Scan Sensors:** Scan for BLE sensors and read the expected BlueCoin names from `config.yaml`.
5. **Sensor Retry Mechanism:** Retry BlueCoin discovery up to 5 times (with a 3-second delay) if the expected device is missing. Exit the pipeline if the hardware is not found.
6. **Scan Actuators:** Scan for the speaker actuator and initialize it.
7. **Retrieve Actuator IDs:** Read the discovered actuator IDs from the `ActuatorManager`.
8. **Start Sensing Layer:** Initialize and start the sensor threads to begin streaming IMU data over BLE.
9. **Configure Alert Policy:** Create the `DrowsyAlertPolicy` using the discovered actuator IDs.
10. **Initialize Computer Vision:** Create the `PersonRoiState` and instantiate the `YoloDpuThread` (passing the ROI state), starting it in standby/idle mode.
11. **Configure Orchestrator:** Create the `EventDispatcher` with the actuator manager, alert policy, YOLO thread, and ROI state.
12. **Start Orchestrator:** Start the `EventDispatcher` background thread.
13. **Hardware Synchronization (Polling):** Enter a connection polling loop (up to 60 seconds), waiting to verify that both the BlueCoin sensor and Speaker actuator report a `CONNECTED` status.
14. **Enter UI Loop:** Enter the main dashboard rendering loop to display system status and video feeds.
15. **Listen for Exit Commands:** Continue the execution loop until `q` is pressed in the GUI or a `KeyboardInterrupt` (Ctrl+C) is received in the terminal.
16. **Safe Shutdown:** In the `finally` block, cleanly stop the dispatcher, sensor manager, actuator manager, video thread, and unregister all dashboard resources to free up hardware.
---

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

---

## Python Dependencies and Environment Setup
The system relies on a mix of standard Python utilities, specialized hardware acceleration bindings, and wireless communication packages.

### Core Python Modules Used
The following libraries are imported across the system architecture and must be available in the execution environment:
*   System & Utilities: `os`, `sys`, `time`, `math`, `uuid`, `typing`, `datetime`, `pathlib`, `yaml` (via `pyyaml`), `collections` (such as `deque`).
*   Concurrency & Communication: `threading`, `queue`, `asyncio`.
*   Wireless & Hardware Interfaces: `bluepy`, `blue_st_sdk`, `dbus_fast`.
*   Data Processing, Vision & Audio: `numpy`, `cv2` (OpenCV), `playsound`.
*   The file sensors/BLE/feature_mems_sensor_fusion_compact.py must be present in the repository even though quaternions are not used by the IMU pipeline. The version of blue_st_sdk installed on the Kria board imports that module at load time from sensors.BLE; removing it causes an ImportError that propagates through the entire import chain starting from bluecoin.py.

---

### Installation Commands
Install the primary application dependencies in the same Python environment used to run `main.py`:

```bash
pip install numpy pyyaml playsound bluepy blue-st-sdk dbus_fast opencv-python
```

### DPU / Vitis-AI Runtime Setup
The machine learning video pipeline requires specialized hardware acceleration bindings to interface with the Xilinx DPU on the KV260 board:

*   Proprietary Imports: The modules `import xir` and `import vart` are mandatory components of the Xilinx Vitis-AI infrastructure. These packages cannot be installed via standard PyPI and must be provided directly by the Xilinx Linux image runtime.
*   FPGA Overlay Initialization: The DPU fabric must be programmed before starting the main Python application. Run the deployment helper script once at system startup:

```bash
bash xmutil_load_dpu.sh
```

### Configuration
 The main configuration file is:
 
```text
config.yaml
```

Current configuration areas:

```yaml

log_base_path: ~/Desktop/CPSA_logs 

enable_system_log: true  
enable_actuation_detail: false  

debug_system_console: true  
debug_event_console: true   

yolo_model_name: "/home/ubuntu/Desktop/PROJECT_CPSA_2026/Video_Pipeline/Yolo_v3/yolo_v3.xmodel"
resnet_model_name: "/home/ubuntu/Desktop/PROJECT_CPSA_2026/Video_Pipeline/Resnet18/resnet18.xmodel"

bluecoins:
  - id: bc_left
    name: "CPSA_L2"

speaker:
  mac: D4:8C:49:C9:DC:9A   
  enable: true             
  scan_timeout: 5          
  fast_retry_attempts: 5      
  retry_interval: 10          
  retry_sleep: 60

sync:
  max_skew_ms: 30        
  stale_ms: 50          

buffer:
  window_size: 150                        
  overlap: 75                            
  debug_print_buffer: false               
  debug_print_features: false              

event_queue_size: 1
```

### Run Procedure
Load the DPU firmware/overlay first, then run the Python entry point directly:
```bash
bash xmutil_load_dpu.sh
python3 main.py
```

---

## IMU Pipeline

Real-time drowsiness detection module based on a single head-mounted STM BlueCoin sensor.

The module receives raw accelerometer and gyroscope data over BLE, estimates the head pitch angle through a complementary filter and classifies two drowsiness patterns: sudden drop and slow drift. Detected events are sent to the central `EventDispatcher`, which manages alert actuator activation and the video pipeline trigger.

---

### Architecture

```
[ STM BlueCoin CPSA_L2 (bc_left) ]
                 │
                 ▼ (Asynchronous BLE Notifications: acc [mg], gyr [dps])
 1. Feature Listeners (Accel/Gyro Listeners)
                 │
                 ▼ synchronizer.update("bc_left", kind, values, ts)
 2. IMUSynchronizer
                 │
                 ├──► [ Non-matching Timestamps ] ──► (Wait / Drop Async Data)
                 │
                 └──► [ Timestamps Aligned ]
                           │
                           ▼ buffer.add_buffer_row(Lacc, Lgyr, ts_emit)
                   3. DataBuffer (Sliding Window Processing)
                           │
                           ├──► [ Window Size < 150 ] ──► (Accumulating Samples)
                           │
                           └──► [ Window Size = 150 (Hop: 75) ]
                                     │
                                     ├──► [ First Window ] ──► (Discard / Warmup Phase)
                                     │
                                     └──► [ Valid Window ]
                                               │
                                               ▼ (Scale: mg → g & Pack into structured dict)
                                       4. DrowsinessClassifier
                                          (Complementary Filter + State Machine)
                                               │
                                               ▼ classifier.recognize(window_payload, window_end_ts)
                                               │
                                               ▼ Genereate Event: drowsiness_tag ("0" | "1" | "2")
                                       5. Event Queue (FIFO)
                                               │
                                               ├──► [ Queue Full ] ──► (enqueue_drop_oldest)
                                               │
                                               └──► [ Push Event ]
                                                         │
                                                         ▼
                                               [ EventDispatcher ] ──► (Camera / Actuator Activation)
```

---

## Data Flow

### 1. BLE acquisition — `feature_listeners.py`

`AccelerometerFeatureListener` and `GyroscopeFeatureListener` are registered on the BlueCoin BLE features by `BlueCoinThread`. Every time the sensor firmware sends a BLE notification, `on_update()` is invoked, converts the raw data to a float tuple and calls `synchronizer.update()`.

### 2. Pairing — `synchronizer.py`

`IMUSynchronizer` maintains a `DeviceState` object for `bc_left`. Each `update()` call stores the incoming acc or gyr measurement. When both fields are populated (`is_ready() = acc is not None and gyr is not None`), the pair is emitted to the buffer and the state is cleared, ready for the next sample pair.

This guarantees that accelerometer and gyroscope data are always forwarded as coherent pairs — the buffer never receives an acc sample without its corresponding gyr, or vice versa. Any packet that arrives with an unrecognised `device_id` is silently dropped via the early-return guard.

### 3. Sliding window — `data_buffer.py`

`DataBuffer` accumulates incoming rows in an internal list. When the list reaches `window_size` samples (150 by default), it emits a window and advances the cursor by `hop_size` samples (75), retaining the last 75 samples as overlap with the next window.

The first window is always silently discarded (warmup), to prevent potentially noisy BLE startup data from influencing the filter calibration.

Before passing data to the classifier, the buffer scales the accelerometer from mg to g (division by 1000) and packs everything into a structured dict:

```python
window_payload = {
    'accX':     np.array([...], dtype=float32) / 1000.0,  # pitch axis   [g]
    'accZ':     np.array([...], dtype=float32) / 1000.0,  # vertical     [g]
    'gyrX':     np.array([...], dtype=float32),            # pitch ω      [dps]
    'ts_array': np.array([...], dtype=float64),            # timestamps   [s]
    'hop_size': 75                                         # new samples in this window
}
```

Note: Only `accX`, `accZ`, `gyrX` are extracted and forwarded as these are the exclusive inputs required by the classifier.

### 4. Classification — `drowsiness_classifier.py`

`DrowsinessClassifier` is the algorithmic core. It receives each window from the buffer, processes only the new samples (anti-overlap computation), applies the complementary filter sample by sample, updates the baseline and runs the state machine.

#### Anti-overlap computation

Since windows overlap by 50%, each window contains samples already processed by the previous one. The classifier processes only the last `hop_size` samples:

```python
if self.last_processed_ts == 0.0:
    start_idx = 0                      # first window: process everything
else:
    start_idx = len(accX) - hop_size   # subsequent windows: new tail only
```

#### Complementary filter

For each new sample, the filter fuses two sources to estimate head pitch:

```
θ_accel = atan2(accX, |accZ|)                        — accurate long-term, noisy
θ_gyro  = θ_prev + ω · dt                            — accurate short-term, drifts
θ       = α · θ_gyro + (1-α) · θ_accel   if static
θ       = θ_gyro                           if gated (vehicle accelerating)
```

The accelerometer gate (`|a_total - 1g| > gate_thresh_g`) disables the accelerometer correction when the vehicle is accelerating or braking, preventing parasitic accelerations from distorting the pitch estimate.

The `dt` is computed from real BLE timestamps rather than a fixed nominal value, with a fallback to `1/target_fs` (1/100Hz in this case) on gaps or out-of-order packets.

#### Dynamic baseline

The driver's neutral posture is estimated and updated once per second over a 30-second rolling history. Only samples within ±5° of the current baseline contribute to the recalculation, preventing drowsiness episodes from pulling the reference point along with them.

#### State machine — detected events

All patterns are evaluated on the deviation `Δθ = θ − baseline`.

**Sudden drop — tag `2`** (highest priority)

Analyses the last 0.40 s of history (40 samples at 100 Hz). Fires when all four conditions are met simultaneously: current deviation > 10°, current angular velocity > 10 dps, peak angular velocity in the window > 10 dps, total excursion of the deviation in the window > 10°. The initial two conditions provide an instantaneous assessment, whereas the subsequent two introduce a temporal analysis dimension.

**Slow drift — tag `1`**

Fires when the deviation exceeds 10° and remains sustained for at least 1.5 seconds with angular velocity < 12 dps (the movement must be slow — a fast movement would already be captured by the sudden drop). If the condition breaks at any point before the timer expires, the episode is discarded and the timer resets.

**Awake — tag `0`**

Default state. Emitted every window when no pattern is detected, allowing `EventDispatcher` to detect the return-to-normal transition.

After any event fires, a **5-second refractory period** blocks new triggers. During this window the state machine returns the last detected tag rather than `0`, keeping the dispatcher informed of the sustained alert state.

### Event output

One event is produced per window regardless of detected state. The event is inserted into the shared queue via `enqueue_drop_oldest()`: if the queue is full, the oldest item is popped and logged with reason `"queue_full"` before the new event is inserted.

### 5. Hardware Orchestration & Pipeline Wiring — sensor_manager.py

The `SensorManager` is responsible for coordinating the hardware lifecycle and connecting the streaming data to the drowsiness classifier:

* **Pipeline Wiring (`__init__`):** Loads the BlueCoin configuration, instantiates the `IMUSynchronizer` and registers the `DrowsinessClassifier` as the target feature sink for the data buffer.
* **Device Discovery (`scan_sensors`):** Performs a BLE scan for BlueCoin devices and stores the results internally.
* **Thread Initialization (`initialize_sensors`):** Initializes a single sensor thread specifically for the left BlueCoin device. It extracts the accelerometer and gyroscope features from the node and pairs them with their respective listeners.
* **Resource Cleanup (`stop_all`):** Stops all active sensor threads and clears the internal tracking list to ensure a clean shutdown.
---

### Configuration and tuning parameters

#### `DrowsinessClassifier` — algorithmic parameters

| Configurable parameters | Default | Effect |
|---|---|---|
| `alpha` | `0.96` | Gyroscope weight in the filter. Raise → faster response, more long-term drift. Lower → less drift, more sensitive to vehicle vibrations |
| `gate_thresh_g` | `0.15 g` | Gating threshold. Raise → correction active even during mild vehicle motion. Lower → more conservative, ignores accel more often |
| `sudden_drop_gyro_thresh` | `10.0 dps` | Minimum angular velocity to qualify a sudden drop. Lower for higher sensitivity; raise to reduce false positives from road bumps |
| `sudden_drop_angle_thresh` | `10.0 °` | Minimum angular excursion over the 0.40 s analysis window |
| `slow_drift_angle_thresh` | `10.0 °` | Deviation from baseline beyond which the drift timer starts |
| `slow_drift_max_gyro` | `12.0 dps` | Maximum angular velocity for a movement to be classified as a slow drift |
| `target_fs` | `100.0 Hz` | Nominal sampling rate — used to size sample-count windows and as dt fallback |

| Hard-coded parameters | Default | Effect |
|---|---|---|
| `SLOW_DRIFT_DURATION_SEC` | `1.5 s` | Time the head must remain above threshold to fire the alert |
| `SUDDEN_DROP_WINDOW_SEC` | `0.40 s` | Duration of the analysis window for sudden drop detection |
| `REFRACTORY_SEC` | `5.0 s` | Minimum time between any two consecutive events |
| `HISTORY_WINDOW_SEC` | `30.0 s` | Length of the circular history deques used for baseline estimation |

---

### Video Pipeline
### Architecture

```text
[ main.py (Main Thread) ]
         │
         ▼ (Spawns and initializes background execution)
 1. YoloDpuThread Initialization
         │
         ▼ (Sequential Model Loading & XIR Graph / VART Runner Configuration)
 2. Hardware Acceleration Setup (DPU)
         │
         ▼ camera.read() [Smart Reconnection Loop: max 5 retries]
 3. Frame Acquisition & Preprocessing
         │
         ▼ dpu_runner.execute_async(input_frame)
 4. Person Detection (YOLOv3 on DPU)
         │
         ├──► [ Person NOT Detected ] ──► (Skip / Next Frame Loop)
         │
         └──► [ Person Detected ]
                 │
                 ▼ nms_boxes(raw_detections, confidence_thresh)
           Non-Maximum Suppression (Duplicate Box Cleanup)
                 │
                 ▼ crop_roi(person_bbox)
           5. Face Localization (Haar Cascade on CPU)
                 │
                 ├──► [ Face Found ] ───────► [ Target: Face ROI ] ──┐
                 │                                                     │
                 └──► [ Face NOT Found ]                               ▼
                      (Profile / Occlusion)                     7. State Classification
                             │                                   (ResNet18 on DPU)
                             ▼ Stage 1: Top-40% Box Estimation         │
                       6. Geometric Fallback Strategy                  ▼
                             │                          resnet_runner.execute_async()
                             ▼ Stage 2: +30% Context Margin            │
                             │                                         ▼ Generate Event:
                       [ Target: Fallback Crop ] ──────────────────────┘ state_tag ("NATURAL" | "DROWSY")
                                                                               │
                                                                               ▼ lock.acquire()
                                                                        8. Shared State Update
                                                                           (Thread-Safe Mutex)
```

---

## Data Flow

### 1. Camera Acquisition & Initialization — `main.py` / `YoloDpuThread`

The `YoloDpuThread` class is instantiated and started by the main thread (`main.py`). Upon startup, the thread sequentially deserializes the `.xmodel` files to load the YOLOv3 and ResNet18 models into the system memory.

### 2. Hardware Acceleration Setup (DPU)

During the preliminary model setup phase, the code analyzes the XIR (Xilinx Intermediate Representation) graphs and instantiates the corresponding VART (Vitis AI Runtime) runners for both YOLOv3 and ResNet18. This configuration maps heavy mathematical operations directly onto the hardware registers and ALUs of the physical Xilinx DPU (Deep Learning Processing Unit) chip.

### 3. Person Detection & NMS — `YoloDpuThread`

The acquired frame is normalized and forwarded to the DPU hardware module configured for YOLOv3. The model performs real-time spatial localization to identify human figures in the field of view. 

Since the algorithm generates multiple overlapping predictive boxes for the same subject, a Non-Maximum Suppression (NMS) algorithm is applied. The cleanup process filters overlaps based on Intersection over Union (IoU) and keeps only the single bounding box associated with the highest confidence score. If no person is detected, the flow interrupts the cascade and immediately skips to the next frame to optimize resources.

### 4. Face Localization — Haar Cascade (CPU)

Once the person's isolated bounding box is obtained, processing temporarily shifts to the main CPU. The system crops the region of interest (ROI) corresponding to the person and applies a traditional Haar Cascade algorithm to locate frontal facial features (eyes, nose, mouth). A margin expansion is operated to ensure that all the features (like hair, ears) are included.

### 5. Geometric Fallback Strategy

If the Haar Cascade fails to detect a face (e.g., subject in profile, turned away, or under harsh lighting variations), a geometric fallback strategy is automatically triggered based on the standard proportions of the human body. The calculation is divided into two sequential phases:

*   Stage 1 (Reconstruction): The top 40% height of the original YOLO bounding box (the head-shoulders area) is isolated. After calculating the horizontal center of this portion, a width equal to 120% of the isolated height is forced.
*   Stage 2 (Margin Expansion): The newly calculated bounding box is symmetrically expanded outwards by 30% on all sides. This zoom-out operation ensures the inclusion of the surrounding context (hair, ears, contour elements).

> Visual Breakdown of the Fallback Strategy: 

  > ```text

  >    Original YOLO Box             Stage 1: Estimated ROI       Stage 2: Final DPU Input

  > ┌─────────────────────┐        ┌───────────────────────┐     ┌────────────────────────────┐

  > │                     │ ▲      │      [Head Area]      │ ▲   │  ........................  │

  > │  (Top 40% Height)   │ │0.4H  │ ◄──── 120% Width ────►│ │   │  :  +30% Context Margin :  │

  > ├─────────────────────┤ ▼      └───────────────────────┘ ▼   │  :   (Wider Framing)    :  │

  > │                     │                                      │  :......................:  │

  > │  (Bottom 60% Body)  │                                      └────────────────────────────┘

  > └─────────────────────┘

  > ```



### 6. State Classification — ResNet18 (DPU)

The second deep learning model (ResNet18) is executed on the DPU conditionally: if the pipeline at step 3 detected no people, ResNet18 inference is completely skipped to reduce the SoC's power consumption.

When a valid ROI (`crop_payload`) is available — whether extracted by Haar Cascade or geometrically reconstructed — the ResNet18 DPU runner processes the crop to classify the subject's state in real-time, assigning one of the system's predefined labels:

| State Tag | Classification Condition | Pipeline Action |
| :--- | :--- | :--- |
| **`NATURAL`** | Subject alert, clear facial dynamics | Normal monitoring continued |
| **`DROWSY`** | Persistent biometric indicators of fatigue | Trigger structural alert events |

---

### Actuation Policy

`DrowsyAlertPolicy` decides when to trigger audio alarms based on drowsiness detection and speaker cooldown limits.

The policy tracks:
* actuator_ids: list of actuator IDs, filtering only the ones that start with `speaker_`.
* _audio_file: the actual file path checked on disk using `AudioLibrary.DROWSINESS_ALERT`.
* _spk_cooldown_sec: a cooldown period set to 5 seconds.
* _spk_last_fire_time: a dictionary mapping each speaker ID to its last activation timestamp using monotonic time.

Configuration:
```yaml
speaker:
  enable: true
  mac: "XX:XX:XX:XX:XX:XX"
  scan_timeout: 5
  fast_retry_attempts: 5
  retry_interval: 5
  retry_sleep: 60
```

---

### Event System and Dispatcher
The event queue is shared across runtime components and consumed by the dispatcher.

#### Queue Configuration
*   `event_queue_size`: Managed asynchronously via a shared `get_event_queue()` mechanism. The thread uses a 0.5-second polling timeout to handle idle states, check internal resource timers, and process continuous video-only policies when no new sensor data arrives.

#### Classification Mapping
The dispatcher translates numerical sensor tags (e.g., from an IMU) into human-readable labels:
*   `0` -> `AWAKE`
*   `1` -> `SLOW_DRIFT`
*   `2` -> `SUDDEN_DROP`

#### Video-Stage Behavior
The dispatcher dynamically controls the execution of the video pipeline (YOLO/ResNet) to optimize processing resources and maintain battery life:
*   Trigger Rules: If an anomalous IMU sensor tag is encountered (`1: SLOW_DRIFT` or `2: SUDDEN_DROP`), the video thread is immediately initialized/activated.
*   `NATURAL` State Auto-Off Countdown: If the active video model pipeline predicts a stable `NATURAL` state continuously for `AWAKE_OFF_DELAY_SEC` (set to 5.0 seconds), the video pipeline is turned off automatically to conserve resources.
*   Anti-Blink Suppression Filter: When a `DROWSY` state is predicted by the video module, the system ensures a continuous duration threshold of less than 1.0 second is treated as a blink. This prevents brief eye blinks from accidentally resetting the camera shutdown timer and suppresses short false positives.

#### Actuation & Cooldown Behavior
*   Trigger Priorities:
    *   *Video Off*: The system relies on sensor anomalies (`1` or `2`) to trigger the policy.
    *   *Video On*: The dispatcher ignores raw sensor tags entirely. Policy evaluations are dictated exclusively by active AI video prediction states.
*   Audio-Alarm Anti-Blink Filter: If the video module yields a `DROWSY` prediction but the condition has persisted for less than 1.0 continuous second, the state is temporarily overridden to `None` to prevent an accidental speaker alarm from playing.
*   Rate Limiting: Consecutive policies are restricted by an `ACTUATION_COOLDOWN` window (currently 5 seconds) before another physical trigger can occur.

---

### Anti-Blink Filter Mechanism
 
The system implements an Anti-Blink Filter within the `EventDispatcher` pipeline to distinguish between a normal human eye blink and a sleep event. This prevents false positives and unnecessary audio flooding.
 
#### Timing Thresholds
1. State Interception: When the computer vision model (YOLO/ResNet) outputs a `DROWSY` prediction, the `EventDispatcher` immediately captures the event inside `_process_policy_for_event`.
2. Monotonic Tracking: The dispatcher initiates a duration window tracking mechanism:
   * If `self._drowsy_since_ts` is empty (`None`), it locks the current time using `time.monotonic()`.
   * On subsequent frames, it calculates the delta: $\Delta t = \text{now} -$ `self._drowsy_since_ts`.
3. Suppression Phase ($\Delta t < 1.0\text{s}$): If the continuous duration of the `DROWSY` state is less than 1.0 second, the system flags the behavior as a standard eye blink.
4. Trigger Phase ($\Delta t \ge 1.0\text{s}$): If the user's eyes remain closed and the `DROWSY` status persists continuously for 1.0 second or longer, the state is validated as an actual microsleep anomaly. The dispatcher passes the raw `DROWSY` status to `DrowsyAlertPolicy.handle()`, which checks the 5-second per-speaker cooldown and activates the speaker sounds.
 
#### State Reset Conditions
* The moment the video pipeline returns a `NATURAL` prediction, the `self._drowsy_since_ts` timestamp is immediately reset to `None`, clearing the window for the next event.

---

### Dashboard
The dashboard manages the main graphical user interface using OpenCV, combining the live camera stream and system logs into a single window.

#### Layout and Features
*   Split Display: The interface features a top video panel that shows the camera feed and a bottom text console that scrolls through system logs.
*   Thread Status: The top header bar reads variables directly from the `YoloDpuThread` to show whether the vision pipeline is currently `ACTIVE` or `IDLE`, along with its current processing phase.

#### Crucial GUI Rules
*   Centralized Rendering: The dashboard owns the window context and handles all rendering operations.
*   System Exit: Pressing `q` inside the window terminates the application cleanly.

---

### Logging
The system includes a centralized logging infrastructure to track the application's runtime behavior and sensor transitions.

#### Features and Data Tracked
*   Monitored Operations: The pipeline records system operations, event queue updates, physical actuation details, and raw console outputs.
*   File Export: Log archives are automatically saved to disk using an organized folder structure located under the directory path defined by `log_base_path`.

#### Configuration Variables
All main logging behaviors are controlled directly inside `config.yaml` using the following properties:

```yaml
enable_system_log: true
enable_actuation_detail: false
debug_system_console: true
debug_event_console: true
log_base_path: "~/Desktop/CPSA_logs"
```
