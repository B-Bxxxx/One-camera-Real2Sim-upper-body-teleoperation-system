```md
```text
███████╗████████╗██╗         ██╗   
██╔════╝╚══██╔══╝██║         ██║   
███████╗   ██║   ██║    ████████████╗
╚════██║   ██║   ██║    ╚════██╔════╝
███████║   ██║   ███████╗    ██║   
╚══════╝   ╚═╝   ╚══════╝    ╚═╝
```

# Real2Sim G1 Upper-Body Teleoperation with RTMPose, OpenClaw Interface and MuJoCo

This repository contains the main source code of a one-camera Real2Sim upper-body teleoperation prototype.

The system tracks a human user's upper-body arm motion from a single webcam using **RTMPose / MMPose**, converts the detected 2D keypoints into robot-control proxy signals, exposes the current state through an **OpenClaw-compatible HTTP interface**, and drives a **Unitree G1 humanoid robot model** inside a **MuJoCo** simulation.

---

## Short Description

Real2Sim upper-body teleoperation prototype using RTMPose/MMPose, an OpenClaw-compatible HTTP interface, and MuJoCo Unitree G1 simulation.

The system tracks human arm motion from a single webcam and maps 2D pose features to shoulder and elbow movements in the simulated robot.

---

## Repository Content

This repository contains only the main implementation file:

```text
main.py
README.md
LICENSE
requirements.txt
```

The following external components are required but are **not included** in this repository:

```text
unitree_g1/
OpenClaw

```

The demo video is provided separately by email.

---

## Required External Folders and Files

### 1. Unitree G1 MuJoCo Model

The Unitree G1 MuJoCo model is not included in this repository due to licensing uncertainty.

To run the project, place the Unitree G1 MuJoCo model folder in the project root directory.

Expected structure:

```text
project_root/
│
├── main.py
├── README.md
├── requirements.txt
├── LICENSE
│
└── unitree_g1/
    └── scene.xml
```

The script expects the MuJoCo model at:

```text
unitree_g1/scene.xml
```

If your model file has a different path or filename, modify this line in `main.py`:

```python
MUJOCO_XML = "unitree_g1/scene.xml"
```

---

### 2. OpenClaw

OpenClaw is not included in this repository.

This project provides an **OpenClaw-compatible HTTP interface**, which means OpenClaw or another external controller can query and control the Real2Sim system through HTTP endpoints.

The Real2Sim program exposes:

```text
GET /health
GET /state
GET /config
GET /stop
```

Default address:

```text
http://127.0.0.1:8765
```

OpenClaw can be connected to this system by creating a tool or agent action that queries these endpoints.

---

## System Architecture

```text
Webcam
  ↓
RTMPose / MMPose
  ↓
2D Human Keypoints + Confidence Scores
  ↓
Real2Sim Feature Extraction
  ↓
Joint Proxy Estimation
  ↓
OpenClaw-compatible HTTP State Interface
  ↓
MuJoCo Unitree G1 Simulation
```

---

## Main Features

- Single-camera human upper-body tracking
- RTMPose / MMPose keypoint detection
- MuJoCo Unitree G1 simulation
- OpenClaw-compatible HTTP state interface
- Shoulder pitch proxy based on visible arm-length shortening
- Shoulder roll estimation from 2D shoulder-elbow geometry
- Shoulder yaw / upper-arm twist proxy
- Elbow flexion tracking
- Runtime parameter tuning through HTTP endpoints
- Manual neutral arm-length calibration
- Real-time OpenCV visualization overlay

---

## One-Camera Method

This project intentionally uses a single RGB camera for stability and simplicity.

Because a single camera cannot reliably reconstruct true 3D depth, the system uses 2D proxy features instead of true anatomical 3D joint angles.

---

## Shoulder Pitch Proxy

Shoulder pitch is estimated from the apparent visible arm length in the image.

```text
arm_length = distance(shoulder, elbow) + distance(elbow, wrist)
```

A neutral arm length is calibrated when the user is in a relaxed/default pose.

If the visible arm length becomes shorter than the calibrated neutral length, the system interprets this as forward arm movement.

```text
shorter visible arm length → forward shoulder pitch proxy
```

This is not a true 3D pitch angle, but it provides a stable control signal for real-time imitation.

Important parameters:

```python
PITCH_ROBOT_SIGN = -1.0
PITCH_GAIN = 2.0
PITCH_MIN = -1.5
PITCH_MAX = 0.0
```

If the simulated robot moves in the wrong pitch direction, change:

```python
PITCH_ROBOT_SIGN = 1.0
```

or:

```python
PITCH_ROBOT_SIGN = -1.0
```

---

## Shoulder Roll

Shoulder roll is estimated from the 2D angle between the shoulder and elbow.

```text
arm hanging down       → roll ≈ 0
arm lifted sideways    → roll increases
```

This works well in a one-camera setup because side arm elevation is visible in the image plane.

---

## Shoulder Yaw / Upper-Arm Twist Proxy

Shoulder yaw is approximated from the signed 2D angle between the upper arm and forearm:

```text
upper arm vector = shoulder → elbow
forearm vector   = elbow → wrist
```

This is not true anatomical humerus rotation, but it provides a visually useful proxy for shoulder yaw / arm twisting in the simulation.

Because this proxy is based on the same shoulder-elbow-wrist keypoints as the elbow flexion calculation, the two signals are not fully independent in a one-camera setup.

For this reason, the twist/yaw proxy can be enabled or disabled during runtime with the `t` key.

```text
t = toggle shoulder yaw / upper-arm twist proxy
```

When twist mode is disabled, the system keeps the shoulder yaw control inactive and prioritizes the elbow flexion signal.

When twist mode is enabled, the same 2D arm geometry is additionally used to drive the shoulder yaw joint.

Important parameters:

```python
TWIST_ENABLED = True
TWIST_GAIN = 2.0
TWIST_MIN = -1.57
TWIST_MAX = 1.57
TWIST_CONF_LIMIT = 0.2

TWIST_ROBOT_SIGN_LEFT = 1.0
TWIST_ROBOT_SIGN_RIGHT = -1.0
```

---

## Elbow Flexion

Elbow flexion is calculated from the 2D angle formed by:

```text
shoulder → elbow → wrist
```

The elbow signal is used to control the simulated robot elbow joints.

---

## Camera and Parameter Calibration

The default proxy gains and thresholds were tuned for the author's hardware setup using a standard webcam with an approximately 70-80 degree field of view.

Using a camera with a significantly different field of view, such as a wide-angle camera or a narrow laptop webcam, may change the apparent arm length and 2D joint geometry.

In that case, the proxy parameters should be recalibrated.

The most important tunable parameters are:

```python
PITCH_GAIN
TWIST_GAIN
CONF_LIMIT
SIMITAS
```

These can also be adjusted at runtime through the OpenClaw-compatible `/config` endpoint.

Examples:

```text
http://127.0.0.1:8765/config?pitch_gain=2.5
http://127.0.0.1:8765/config?twist_gain=1.5
http://127.0.0.1:8765/config?smoothing=0.2
http://127.0.0.1:8765/config?conf_limit=0.45
```

---

## Controls

| Key | Function |
|---|---|
| `c` | Calibrate neutral arm length |
| `t` | Toggle shoulder yaw / upper-arm twist proxy |
| `q` | Quit |

Before using the shoulder pitch proxy, stand in a neutral pose with relaxed arms and press:

```text
c
```

This stores the current visible arm length as the neutral reference.

---

## OpenClaw-Compatible HTTP Interface

The system exposes an HTTP API at:

```text
http://127.0.0.1:8765
```

This interface can be used by OpenClaw or any external controller.

---

### Health Check

```text
GET /health
```

Example:

```text
http://127.0.0.1:8765/health
```

Expected response:

```json
{
  "ok": true,
  "service": "real2sim-openclaw"
}
```

---

### Current State

```text
GET /state
```

Example:

```text
http://127.0.0.1:8765/state
```

The returned state contains:

- current FPS
- keypoint coordinates
- keypoint confidence values
- extracted motion features
- robot joint commands
- neutral calibration values
- runtime configuration

Example response structure:

```json
{
  "fps": 14.2,
  "status": "OK",
  "features": {
    "left_pitch_proxy": -0.23,
    "right_pitch_proxy": -0.18,
    "left_roll": 0.72,
    "right_roll": -0.65,
    "left_elbow_flex": 1.24,
    "right_elbow_flex": 1.10,
    "left_twist_cmd": 0.31,
    "right_twist_cmd": -0.28
  },
  "joints": {
    "left_shoulder_pitch": -0.23,
    "right_shoulder_pitch": -0.18,
    "left_shoulder_roll": 0.72,
    "right_shoulder_roll": -0.65,
    "left_shoulder_yaw": 0.31,
    "right_shoulder_yaw": -0.28,
    "left_elbow": 0.35,
    "right_elbow": 0.47
  }
}
```

---

### Runtime Configuration

Parameters can be changed while the program is running.

Examples:

```text
GET /config?pitch_gain=2.0
GET /config?twist_gain=2.0
GET /config?conf_limit=0.45
GET /config?smoothing=0.2
GET /config?twist_enabled=true
GET /config?twist_enabled=false
GET /config?twist_sign_left=1&twist_sign_right=-1
```

---

### Stop the System

```text
GET /stop
```

Example:

```text
http://127.0.0.1:8765/stop
```

---

## MuJoCo Joint Mapping

The script attempts to control the following Unitree G1 joints:

```text
right_shoulder_pitch_joint
left_shoulder_pitch_joint
right_shoulder_roll_joint
left_shoulder_roll_joint
right_elbow_joint
left_elbow_joint
```

For shoulder yaw / twist, the script tries multiple possible joint names:

```text
right_shoulder_yaw_joint
right_shoulder_yaw
right_shoulder_rotation_joint
right_shoulder_twist_joint

left_shoulder_yaw_joint
left_shoulder_yaw
left_shoulder_rotation_joint
left_shoulder_twist_joint
```

If no shoulder yaw joint is found, the system continues running and simply skips yaw control.

---

## Tested Environment

The project was tested with the following setup:

| Component | Version |
|---|---|
| OS | Windows |
| GPU | NVIDIA RTX 3050 Laptop GPU, 4 GB VRAM |
| CPU | AMD Ryzen 5 5600H |
| Python | 3.10 |
| MMPose | 1.3.2 |
| PyTorch | 2.1.0+cu121 |
| CUDA | 12.1 |
| MMCV | 2.1.0 |
| OpenCV | 4.9.0.80 |
| MuJoCo | 3.8.1 |
| NumPy | 1.26.4 |

---

## Installation

### 1. Create a Python virtual environment

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\venv\Scripts\Activate.ps1
```

---

### 2. Upgrade pip tools

```powershell
python -m pip install --upgrade pip setuptools wheel
```

---

### 3. Install PyTorch with CUDA 12.1

```powershell
pip install torch==2.1.0+cu121 torchvision==0.16.0+cu121 torchaudio==2.1.0+cu121 --index-url https://download.pytorch.org/whl/cu121
```

---

### 4. Install MMCV

```powershell
pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.1.0/index.html
```

---

### 5. Install remaining dependencies

```powershell
pip install mmpose==1.3.2
pip install opencv-python==4.9.0.80
pip install mujoco==3.8.1
pip install numpy==1.26.4
pip install json-tricks
pip install munkres
pip install xtcocotools
pip install chumpy
```

Alternatively:

```powershell
pip install -r requirements.txt
```

If dependency issues occur, install PyTorch and MMCV manually first, as shown above.

---

## Example requirements.txt

```txt
mmpose==1.3.2
torch==2.1.0+cu121
torchvision==0.16.0+cu121
torchaudio==2.1.0+cu121
mmcv==2.1.0
opencv-python==4.9.0.80
mujoco==3.8.1
numpy==1.26.4
json-tricks
munkres
xtcocotools
chumpy
```

Note: CUDA-specific PyTorch wheels may require installing PyTorch separately using the official PyTorch CUDA wheel index.

---

## Running

After placing the required `unitree_g1/` folder next to `main.py`, run:

```powershell
python main.py
```

The program starts:

1. OpenClaw-compatible HTTP server
2. RTMPose / MMPose inference
3. Camera capture
4. MuJoCo Unitree G1 simulation
5. Real-time motion retargeting loop

---

## Notes About One-Camera Tracking

Advantages:

- simple setup
- no stereo calibration
- no camera synchronization problem
- stable 2D keypoints from RTMPose
- easier real-time performance

Limitations:

- no true depth estimation
- pitch is a proxy, not a real 3D joint angle
- twist/yaw is a proxy, not true humerus rotation
- movements depend on camera viewpoint
- calibration pose affects pitch behavior
- twist and elbow flexion are not fully independent because both use shoulder-elbow-wrist geometry

---

## Why RTMPose / MMPose?

RTMPose provides stable and accurate keypoints suitable for real-time robot-control experiments.

Main benefits:

- shoulder, elbow and wrist localization
- keypoint confidence scores
- GPU acceleration
- robust tracking under partial occlusion
- better suited for robot-control pipelines than the initial MediaPipe prototype

In this project, RTMPose is used as the perception module, while the Real2Sim mapper converts detected keypoints into robot-control signals.

---

## Why OpenClaw?

OpenClaw is used as an agent/interface layer.

In this prototype, OpenClaw does not replace the pose estimator.

Instead, the system exposes the current tracking and robot-control state through a simple HTTP API that can be queried or controlled by an OpenClaw tool or agent.

The role of OpenClaw in the architecture:

```text
RTMPose = visual perception
Real2Sim mapper = motion feature extraction
OpenClaw = agent interface / runtime control
MuJoCo = robot simulation
```

---

## Known Limitations

- Single-camera system cannot recover true 3D arm pose.
- Shoulder pitch is based on visible arm-length shortening.
- Shoulder yaw/twist is based on a 2D upper-arm/forearm direction proxy.
- The system is sensitive to camera placement and calibration pose.
- The Unitree G1 model is not included in this repository.
- OpenClaw itself is not included in this repository.
- Very fast hand motion may reduce tracking quality.
- The twist proxy can overlap with elbow motion because both use shoulder, elbow and wrist keypoints.

---

## Future Work

Possible improvements:

- two-camera RTMPose-based triangulation
- stereo reprojection-error filtering
- body-frame based 3D shoulder angle estimation
- dedicated OpenClaw tool definition
- One Euro filter or Kalman filtering
- improved inverse kinematics mapping for Unitree G1
- ROS2 bridge
- recording and replaying motion sequences
- WebSocket-based live OpenClaw communication

---

## Acknowledgements

This project uses or interfaces with:

- RTMPose / MMPose by OpenMMLab
- MuJoCo by Google DeepMind
- Unitree G1 MuJoCo model
- OpenClaw-compatible interface concept

---

## License

This repository is published under a Creative Commons license.

The submitted repository uses:

```text
Creative Commons Zero v1.0 Universal
```

Note: external assets, such as the Unitree G1 MuJoCo model or OpenClaw itself, are not included in this repository and may have their own licenses.

---

## Author

Bence Bodnár

GitHub: [B-Bxxxx]
