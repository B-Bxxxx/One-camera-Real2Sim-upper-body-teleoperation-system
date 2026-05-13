import cv2
import time
import math
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse, parse_qs

import numpy as np
import mujoco
import mujoco.viewer
from mmpose.apis import MMPoseInferencer


# ============================================================
# CONFIG
# ============================================================

CAM_ID = 0
FRAME_W = 640
FRAME_H = 480

DEVICE = "cuda:0"
MODEL = "human"
# MODEL = "rtmpose-s"
# MODEL = "rtmpose-t"

MUJOCO_XML = "unitree_g1/scene.xml"

OPENCLAW_HTTP_PORT = 8765

CONF_LIMIT = 0.35
SIMITAS = 0.15

PITCH_ROBOT_SIGN = -1.0

# Pitch proxy: karhossz rövidülés -> előre emelés proxy.
PITCH_GAIN = 2.0
PITCH_MIN = -1.5
PITCH_MAX = 0.0

# Felkarcsavarás / váll yaw proxy.
TWIST_ENABLED = True
TWIST_GAIN = 2.0
TWIST_MIN = -1.57
TWIST_MAX = 1.57
TWIST_CONF_LIMIT = 0.2

# Ha rossz irányba fordul a robot váll-yaw, ezeket fordítsd +/-1-re.
TWIST_ROBOT_SIGN_LEFT = 1.0
TWIST_ROBOT_SIGN_RIGHT = -1.0

# Roll és könyök limitek
ROLL_MIN = -1.57
ROLL_MAX = 1.2
ELBOW_MIN = -1.57
ELBOW_MAX = 1.2

# C gomb: aktuális karhossz elmentése neutralnak.
AUTO_CALIBRATE_ON_FIRST_GOOD_FRAME = True


# ============================================================
# COCO 17 KEYPOINT INDEXEK
# ============================================================

NOSE = 0
LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6
LEFT_ELBOW = 7
RIGHT_ELBOW = 8
LEFT_WRIST = 9
RIGHT_WRIST = 10
LEFT_HIP = 11
RIGHT_HIP = 12

KP_NAMES = {
    "left_shoulder": LEFT_SHOULDER,
    "right_shoulder": RIGHT_SHOULDER,
    "left_elbow": LEFT_ELBOW,
    "right_elbow": RIGHT_ELBOW,
    "left_wrist": LEFT_WRIST,
    "right_wrist": RIGHT_WRIST,
    "left_hip": LEFT_HIP,
    "right_hip": RIGHT_HIP,
}


# ============================================================
# GLOBAL STATE OPENCLAW-HOZ
# ============================================================

state_lock = threading.Lock()

STATE = {
    "running": True,
    "timestamp": None,
    "fps": 0.0,
    "status": "init",
    "model": MODEL,
    "device": DEVICE,
    "config": {
        "pitch_gain": PITCH_GAIN,
        "pitch_min": PITCH_MIN,
        "pitch_max": PITCH_MAX,
        "pitch_robot_sign": PITCH_ROBOT_SIGN,
        "twist_enabled": TWIST_ENABLED,
        "twist_gain": TWIST_GAIN,
        "twist_min": TWIST_MIN,
        "twist_max": TWIST_MAX,
        "twist_conf_limit": TWIST_CONF_LIMIT,
        "twist_robot_sign_left": TWIST_ROBOT_SIGN_LEFT,
        "twist_robot_sign_right": TWIST_ROBOT_SIGN_RIGHT,
        "conf_limit": CONF_LIMIT,
        "smoothing": SIMITAS,
    },
    "neutral": {
        "left_arm_len": None,
        "right_arm_len": None,
    },
    "keypoints": {},
    "features": {},
    "joints": {},
}


# ============================================================
# UTIL
# ============================================================

def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def dist2d(a, b):
    dx = float(a[0] - b[0])
    dy = float(a[1] - b[1])
    return math.sqrt(dx * dx + dy * dy)


def angle_2d(a, b, c):
    """ABC szög 2D-ben, b a középpont."""
    a = np.array(a[:2], dtype=np.float64)
    b = np.array(b[:2], dtype=np.float64)
    c = np.array(c[:2], dtype=np.float64)

    ba = a - b
    bc = c - b

    cosang = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-9)
    return math.acos(np.clip(cosang, -1.0, 1.0))


def signed_angle_2d(v1, v2):
    """Két 2D vektor előjeles szöge radiánban."""
    v1 = np.asarray(v1, dtype=np.float64)
    v2 = np.asarray(v2, dtype=np.float64)

    v1 = v1 / (np.linalg.norm(v1) + 1e-9)
    v2 = v2 / (np.linalg.norm(v2) + 1e-9)

    cross = v1[0] * v2[1] - v1[1] * v2[0]
    dot = float(np.dot(v1, v2))
    return math.atan2(cross, dot)


def get_first_person(result):
    preds = result.get("predictions", [])
    if not preds or len(preds[0]) == 0:
        return None, None

    person = preds[0][0]
    kpts = np.array(person["keypoints"], dtype=np.float64)
    scores = np.array(person["keypoint_scores"], dtype=np.float64)
    return kpts, scores


def valid(scores, idx):
    return scores is not None and idx < len(scores) and scores[idx] >= CONF_LIMIT


def shoulder_roll_2d(shoulder, elbow, side):
    """
    Oldalemelés 2D képsík alapján.
    0: kar lefelé
    kb. pi/2: vízszintes oldalra
    """
    dx = elbow[0] - shoulder[0]
    dy = elbow[1] - shoulder[1]

    val = math.atan2(abs(dx), dy + 1e-6)

    if side == "right":
        return -val
    return val


def upper_arm_twist_proxy_2d(shoulder, elbow, wrist, side):
    """
    Felkarcsavarás / váll-yaw proxy 1 kamerából.

    Ez NEM valódi anatómiai felkarcsavarás.
    Azt méri, hogy az alkar 2D iránya mennyire fordul el a felkar 2D irányához képest.

    shoulder->elbow = felkar irány
    elbow->wrist    = alkar irány
    """
    upper = np.array([elbow[0] - shoulder[0], elbow[1] - shoulder[1]], dtype=np.float64)
    forearm = np.array([wrist[0] - elbow[0], wrist[1] - elbow[1]], dtype=np.float64)

    if np.linalg.norm(upper) < 1e-6 or np.linalg.norm(forearm) < 1e-6:
        return 0.0

    angle = signed_angle_2d(upper, forearm)

    # Bal kar tükörkorrekció.
    if side == "left":
        angle = -angle

    return angle


def arm_length_2d(shoulder, elbow, wrist):
    upper = dist2d(shoulder, elbow)
    lower = dist2d(elbow, wrist)
    return upper + lower


def pitch_from_arm_length(current_len, neutral_len):
    if neutral_len is None or neutral_len <= 1e-6:
        return 0.0, 1.0

    ratio = current_len / neutral_len

    # Előre mozgás proxy: a kar látszó hossza rövidül.
    raw = max(0.0, 1.0 - ratio)

    # A detektálás mindig pozitív, csak a robot joint előjelét fordítjuk.
    pitch = PITCH_ROBOT_SIGN * raw * PITCH_GAIN
    pitch = clamp(pitch, PITCH_MIN, PITCH_MAX)

    return pitch, ratio


def draw_keypoints(frame, kpts, scores):
    if kpts is None or scores is None:
        return

    for name, idx in KP_NAMES.items():
        if valid(scores, idx):
            x, y = kpts[idx][:2]
            cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 255), -1)
            cv2.putText(
                frame,
                name.replace("_", " "),
                (int(x) + 5, int(y) - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 255, 255),
                1,
            )


def keypoints_for_state(kpts, scores):
    out = {}

    if kpts is None or scores is None:
        return out

    for name, idx in KP_NAMES.items():
        if idx < len(scores):
            out[name] = {
                "x": float(kpts[idx][0]),
                "y": float(kpts[idx][1]),
                "conf": float(scores[idx]),
            }

    return out


def find_joint_id(model, possible_names):
    """Több lehetséges joint név közül visszaadja az első létező MuJoCo joint id-t."""
    for name in possible_names:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid != -1:
            return jid, name
    return -1, None


# ============================================================
# OPENCLAW HTTP SERVER
# ============================================================

class OpenClawHandler(BaseHTTPRequestHandler):
    def _send_json(self, obj, code=200):
        data = json.dumps(obj, indent=2).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        qs = parse_qs(parsed.query)

        if path == "/state":
            with state_lock:
                snapshot = json.loads(json.dumps(STATE))
            self._send_json(snapshot)
            return

        if path == "/health":
            self._send_json({"ok": True, "service": "real2sim-openclaw"})
            return

        if path == "/config":
            global PITCH_GAIN, CONF_LIMIT, SIMITAS
            global TWIST_GAIN, TWIST_ENABLED, TWIST_ROBOT_SIGN_LEFT, TWIST_ROBOT_SIGN_RIGHT

            changed = {}

            if "pitch_gain" in qs:
                PITCH_GAIN = float(qs["pitch_gain"][0])
                changed["pitch_gain"] = PITCH_GAIN

            if "conf_limit" in qs:
                CONF_LIMIT = float(qs["conf_limit"][0])
                changed["conf_limit"] = CONF_LIMIT

            if "smoothing" in qs:
                SIMITAS = float(qs["smoothing"][0])
                changed["smoothing"] = SIMITAS

            if "twist_gain" in qs:
                TWIST_GAIN = float(qs["twist_gain"][0])
                changed["twist_gain"] = TWIST_GAIN

            if "twist_enabled" in qs:
                val = qs["twist_enabled"][0].lower()
                TWIST_ENABLED = val in ("1", "true", "yes", "on")
                changed["twist_enabled"] = TWIST_ENABLED

            if "twist_sign_left" in qs:
                TWIST_ROBOT_SIGN_LEFT = float(qs["twist_sign_left"][0])
                changed["twist_robot_sign_left"] = TWIST_ROBOT_SIGN_LEFT

            if "twist_sign_right" in qs:
                TWIST_ROBOT_SIGN_RIGHT = float(qs["twist_sign_right"][0])
                changed["twist_robot_sign_right"] = TWIST_ROBOT_SIGN_RIGHT

            with state_lock:
                STATE["config"]["pitch_gain"] = PITCH_GAIN
                STATE["config"]["conf_limit"] = CONF_LIMIT
                STATE["config"]["smoothing"] = SIMITAS
                STATE["config"]["twist_gain"] = TWIST_GAIN
                STATE["config"]["twist_enabled"] = TWIST_ENABLED
                STATE["config"]["twist_robot_sign_left"] = TWIST_ROBOT_SIGN_LEFT
                STATE["config"]["twist_robot_sign_right"] = TWIST_ROBOT_SIGN_RIGHT

            self._send_json({"ok": True, "changed": changed})
            return

        if path == "/stop":
            with state_lock:
                STATE["running"] = False
            self._send_json({"ok": True, "running": False})
            return

        self._send_json({
            "ok": False,
            "error": "unknown endpoint",
            "endpoints": [
                "/health",
                "/state",
                "/config?pitch_gain=2.0&conf_limit=0.35&smoothing=0.15",
                "/config?twist_gain=0.3&twist_enabled=true",
                "/config?twist_sign_left=-1&twist_sign_right=1",
                "/stop",
            ],
        }, code=404)

    def log_message(self, format, *args):
        return


def start_openclaw_server():
    server = HTTPServer(("127.0.0.1", OPENCLAW_HTTP_PORT), OpenClawHandler)
    print(f"OpenClaw HTTP interface: http://127.0.0.1:{OPENCLAW_HTTP_PORT}/state")
    server.serve_forever()


# ============================================================
# MAIN
# ============================================================

print("OpenClaw HTTP szerver indítása...")
server_thread = threading.Thread(target=start_openclaw_server, daemon=True)
server_thread.start()

print("RTMPose inicializálás...")
inferencer = MMPoseInferencer(pose2d=MODEL, device=DEVICE)
print("RTMPose kész.")

print("MuJoCo G1 betöltés...")
modell = mujoco.MjModel.from_xml_path(MUJOCO_XML)
adat = mujoco.MjData(modell)
modell.opt.gravity[:] = [0, 0, 0]

# Váll-yaw joint neve modellverziótól függhet, ezért több lehetőséget próbálunk.
right_shoulder_yaw_id, right_shoulder_yaw_name = find_joint_id(modell, [
    "right_shoulder_yaw_joint",
    "right_shoulder_yaw",
    "right_shoulder_rotation_joint",
    "right_shoulder_twist_joint",
])
left_shoulder_yaw_id, left_shoulder_yaw_name = find_joint_id(modell, [
    "left_shoulder_yaw_joint",
    "left_shoulder_yaw",
    "left_shoulder_rotation_joint",
    "left_shoulder_twist_joint",
])

motorok = {
    "j_vall": mujoco.mj_name2id(modell, mujoco.mjtObj.mjOBJ_JOINT, "right_shoulder_pitch_joint"),
    "b_vall": mujoco.mj_name2id(modell, mujoco.mjtObj.mjOBJ_JOINT, "left_shoulder_pitch_joint"),
    "j_konyok": mujoco.mj_name2id(modell, mujoco.mjtObj.mjOBJ_JOINT, "right_elbow_joint"),
    "b_konyok": mujoco.mj_name2id(modell, mujoco.mjtObj.mjOBJ_JOINT, "left_elbow_joint"),
    "j_vall_roll": mujoco.mj_name2id(modell, mujoco.mjtObj.mjOBJ_JOINT, "right_shoulder_roll_joint"),
    "b_vall_roll": mujoco.mj_name2id(modell, mujoco.mjtObj.mjOBJ_JOINT, "left_shoulder_roll_joint"),
    "j_vall_yaw": right_shoulder_yaw_id,
    "b_vall_yaw": left_shoulder_yaw_id,
}

if right_shoulder_yaw_id == -1:
    print("FIGYELEM: jobb vall yaw/twist joint nem talalhato, j_vall_yaw kihagyva.")
else:
    print(f"Jobb vall yaw joint: {right_shoulder_yaw_name}")

if left_shoulder_yaw_id == -1:
    print("FIGYELEM: bal vall yaw/twist joint nem talalhato, b_vall_yaw kihagyva.")
else:
    print(f"Bal vall yaw joint: {left_shoulder_yaw_name}")

e = {k: 0.0 for k in motorok.keys() if motorok[k] != -1}

cap = cv2.VideoCapture(CAM_ID, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

if not cap.isOpened():
    raise RuntimeError("Nem sikerült megnyitni a kamerát.")

neutral_left_len = None
neutral_right_len = None

prev_time = time.time()

print("Real2Sim indul.")
print("Billentyűk:")
print("  q = kilépés")
print("  c = neutral karhossz kalibrálás")
print("  t = twist/yaw proxy ki-be kapcsolás")
print(f"OpenClaw lekérdezés: http://127.0.0.1:{OPENCLAW_HTTP_PORT}/state")

with mujoco.viewer.launch_passive(modell, adat) as viewer:
    while viewer.is_running() and cap.isOpened():
        with state_lock:
            if not STATE["running"]:
                break

        adat.qpos[0:3] = [0.0, 0.0, 0.75]
        adat.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]

        ok, frame = cap.read()
        if not ok:
            break

        result = next(inferencer(frame, show=False, return_vis=False))
        vis = frame.copy()

        kpts, scores = get_first_person(result)
        draw_keypoints(vis, kpts, scores)

        cel = {}
        features = {}
        status = "Nincs eleg stabil keypoint"

        needed = [
            LEFT_SHOULDER, RIGHT_SHOULDER,
            LEFT_ELBOW, RIGHT_ELBOW,
            LEFT_WRIST, RIGHT_WRIST,
        ]

        if kpts is not None and scores is not None and all(valid(scores, i) for i in needed):
            ls = kpts[LEFT_SHOULDER]
            rs = kpts[RIGHT_SHOULDER]
            le = kpts[LEFT_ELBOW]
            re = kpts[RIGHT_ELBOW]
            lw = kpts[LEFT_WRIST]
            rw = kpts[RIGHT_WRIST]

            left_len = arm_length_2d(ls, le, lw)
            right_len = arm_length_2d(rs, re, rw)

            if AUTO_CALIBRATE_ON_FIRST_GOOD_FRAME:
                if neutral_left_len is None:
                    neutral_left_len = left_len
                if neutral_right_len is None:
                    neutral_right_len = right_len

            # 1. Könyök hajlítás
            l_elbow_angle = angle_2d(ls, le, lw)
            r_elbow_angle = angle_2d(rs, re, rw)

            l_elbow_flex = math.pi - l_elbow_angle
            r_elbow_flex = math.pi - r_elbow_angle

            cel["b_konyok"] = -(l_elbow_flex - (math.pi / 2))
            cel["j_konyok"] = -(r_elbow_flex - (math.pi / 2))

            # 2. Váll roll / oldalemelés
            cel["b_vall_roll"] = shoulder_roll_2d(ls, le, "left")
            cel["j_vall_roll"] = shoulder_roll_2d(rs, re, "right")

            # 3. Váll pitch proxy karhossz alapján
            l_pitch, l_ratio = pitch_from_arm_length(left_len, neutral_left_len)
            r_pitch, r_ratio = pitch_from_arm_length(right_len, neutral_right_len)

            cel["b_vall"] = l_pitch
            cel["j_vall"] = r_pitch

            # 4. Felkarcsavarás / váll yaw proxy
            l_twist_raw = upper_arm_twist_proxy_2d(ls, le, lw, "left")
            r_twist_raw = upper_arm_twist_proxy_2d(rs, re, rw, "right")

            left_twist_conf_ok = (
                scores[LEFT_SHOULDER] >= TWIST_CONF_LIMIT and
                scores[LEFT_ELBOW] >= TWIST_CONF_LIMIT and
                scores[LEFT_WRIST] >= TWIST_CONF_LIMIT
            )
            right_twist_conf_ok = (
                scores[RIGHT_SHOULDER] >= TWIST_CONF_LIMIT and
                scores[RIGHT_ELBOW] >= TWIST_CONF_LIMIT and
                scores[RIGHT_WRIST] >= TWIST_CONF_LIMIT
            )

            l_twist_cmd = 0.0
            r_twist_cmd = 0.0

            if TWIST_ENABLED and left_twist_conf_ok:
                l_twist_cmd = clamp(
                    TWIST_ROBOT_SIGN_LEFT * l_twist_raw * TWIST_GAIN,
                    TWIST_MIN,
                    TWIST_MAX,
                )
                cel["b_vall_yaw"] = l_twist_cmd

            if TWIST_ENABLED and right_twist_conf_ok:
                r_twist_cmd = clamp(
                    TWIST_ROBOT_SIGN_RIGHT * r_twist_raw * TWIST_GAIN,
                    TWIST_MIN,
                    TWIST_MAX,
                )
                cel["j_vall_yaw"] = r_twist_cmd

            features = {
                "left_arm_len": float(left_len),
                "right_arm_len": float(right_len),
                "left_arm_ratio": float(l_ratio),
                "right_arm_ratio": float(r_ratio),
                "left_pitch_proxy": float(l_pitch),
                "right_pitch_proxy": float(r_pitch),
                "left_roll": float(cel["b_vall_roll"]),
                "right_roll": float(cel["j_vall_roll"]),
                "left_elbow_flex": float(l_elbow_flex),
                "right_elbow_flex": float(r_elbow_flex),
                "left_twist_raw": float(l_twist_raw),
                "right_twist_raw": float(r_twist_raw),
                "left_twist_cmd": float(l_twist_cmd),
                "right_twist_cmd": float(r_twist_cmd),
                "twist_enabled": bool(TWIST_ENABLED),
                "left_twist_conf_ok": bool(left_twist_conf_ok),
                "right_twist_conf_ok": bool(right_twist_conf_ok),
                "left_conf": {
                    "shoulder": float(scores[LEFT_SHOULDER]),
                    "elbow": float(scores[LEFT_ELBOW]),
                    "wrist": float(scores[LEFT_WRIST]),
                },
                "right_conf": {
                    "shoulder": float(scores[RIGHT_SHOULDER]),
                    "elbow": float(scores[RIGHT_ELBOW]),
                    "wrist": float(scores[RIGHT_WRIST]),
                },
            }

            status = "OK"

        LIMITEK = {
            "j_konyok": (ELBOW_MIN, ELBOW_MAX),
            "b_konyok": (ELBOW_MIN, ELBOW_MAX),
            "j_vall_roll": (ROLL_MIN, ROLL_MAX),
            "b_vall_roll": (ROLL_MIN, ROLL_MAX),
            "j_vall": (PITCH_MIN, PITCH_MAX),
            "b_vall": (PITCH_MIN, PITCH_MAX),
            "j_vall_yaw": (TWIST_MIN, TWIST_MAX),
            "b_vall_yaw": (TWIST_MIN, TWIST_MAX),
        }

        # Motor frissítés
        for k, target in cel.items():
            if k not in e:
                continue

            e[k] = SIMITAS * target + (1.0 - SIMITAS) * e[k]

            if k in motorok and motorok[k] != -1:
                val = e[k]

                if k in LIMITEK:
                    mn, mx = LIMITEK[k]
                    val = clamp(val, mn, mx)
                    e[k] = val

                adat.qpos[modell.jnt_qposadr[motorok[k]]] = val

        mujoco.mj_kinematics(modell, adat)
        viewer.sync()

        now = time.time()
        fps = 1.0 / (now - prev_time + 1e-9)
        prev_time = now

        joints_for_state = {
            "right_shoulder_pitch": float(e.get("j_vall", 0.0)),
            "left_shoulder_pitch": float(e.get("b_vall", 0.0)),
            "right_shoulder_roll": float(e.get("j_vall_roll", 0.0)),
            "left_shoulder_roll": float(e.get("b_vall_roll", 0.0)),
            "right_shoulder_yaw": float(e.get("j_vall_yaw", 0.0)),
            "left_shoulder_yaw": float(e.get("b_vall_yaw", 0.0)),
            "right_elbow": float(e.get("j_konyok", 0.0)),
            "left_elbow": float(e.get("b_konyok", 0.0)),
        }

        with state_lock:
            STATE["timestamp"] = time.time()
            STATE["fps"] = float(fps)
            STATE["status"] = status
            STATE["keypoints"] = keypoints_for_state(kpts, scores)
            STATE["features"] = features
            STATE["joints"] = joints_for_state
            STATE["neutral"]["left_arm_len"] = None if neutral_left_len is None else float(neutral_left_len)
            STATE["neutral"]["right_arm_len"] = None if neutral_right_len is None else float(neutral_right_len)
            STATE["config"]["twist_enabled"] = TWIST_ENABLED
            STATE["config"]["twist_gain"] = TWIST_GAIN
            STATE["config"]["twist_robot_sign_left"] = TWIST_ROBOT_SIGN_LEFT
            STATE["config"]["twist_robot_sign_right"] = TWIST_ROBOT_SIGN_RIGHT

        # Overlay
        rows = [
            f"Real2Sim | RTMPose + OpenClaw HTTP + MuJoCo G1 | {status}",
            f"FPS: {fps:.1f} | {DEVICE} | {MODEL}",
            f"OpenClaw: http://127.0.0.1:{OPENCLAW_HTTP_PORT}/state",
            f"Neutral L/R: {neutral_left_len if neutral_left_len else 0:.1f} / {neutral_right_len if neutral_right_len else 0:.1f}",
            f"Twist: {'ON' if TWIST_ENABLED else 'OFF'} | gain={TWIST_GAIN:.2f}",
        ]

        if features:
            rows.extend([
                f"L pitch proxy: {features['left_pitch_proxy']:+.3f} | ratio: {features['left_arm_ratio']:.3f}",
                f"R pitch proxy: {features['right_pitch_proxy']:+.3f} | ratio: {features['right_arm_ratio']:.3f}",
                f"L roll: {math.degrees(features['left_roll']):+.1f} deg | R roll: {math.degrees(features['right_roll']):+.1f} deg",
                f"L elbow: {math.degrees(features['left_elbow_flex']):+.1f} deg | R elbow: {math.degrees(features['right_elbow_flex']):+.1f} deg",
                f"L twist raw/cmd: {math.degrees(features['left_twist_raw']):+.1f}/{math.degrees(features['left_twist_cmd']):+.1f} deg",
                f"R twist raw/cmd: {math.degrees(features['right_twist_raw']):+.1f}/{math.degrees(features['right_twist_cmd']):+.1f} deg",
            ])

        y = 28
        color = (0, 255, 0) if status == "OK" else (0, 0, 255)
        for text in rows:
            cv2.putText(vis, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 2)
            y += 24

        cv2.imshow("Real2Sim - RTMPose OpenClaw MuJoCo G1", vis)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        if key == ord("c") and features:
            neutral_left_len = features["left_arm_len"]
            neutral_right_len = features["right_arm_len"]
            print(f"Neutral kalibralva: L={neutral_left_len:.1f}, R={neutral_right_len:.1f}")

        if key == ord("t"):
            TWIST_ENABLED = not TWIST_ENABLED
            print(f"Twist/yaw proxy: {'ON' if TWIST_ENABLED else 'OFF'}")

cap.release()
cv2.destroyAllWindows()

with state_lock:
    STATE["running"] = False

print("Leállítva.")