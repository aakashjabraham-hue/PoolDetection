import cv2
import numpy as np
import requests
import time
import json
import os
import threading
import base64
import re
import subprocess
import platform
import atexit
import ctypes
import shutil
from collections import deque

STREAM_URL         = "rtsp://192.168.68.75:8554/uppool"
CHECK_INTERVAL     = 30
AI_CHECK_INTERVAL  = 300
DEBRIS_THRESHOLD   = 5.0   # Alert when debris covers more than this % of the pool

BOT_TOKEN = "PUT_BOT_TOKEN_HERE"
CHAT_ID   = "PUT_CHAT_ID_HERE"

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL   = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")
OPENROUTER_URL     = "https://openrouter.ai/api/v1/chat/completions"

MASK_FILE          = "pool_mask.json"
COLOR_PROFILE_FILE = "pool_color_profile.json"

# -------------------------
# TUNING KNOBS
# -------------------------

MIN_CONTOUR_AREA  = 40

DEBRIS_COLOR_RANGES = [
    # --- Leaves ---
    ([10,  40,  30],  [25,  255, 200]),
    ([20,  40,  30],  [35,  255, 220]),
    ([35,  40,  30],  [85,  255, 200]),
    # --- Dirt / mud ---
    ([5,   20,  20],  [20,  120, 120]),
    ([0,   0,   20],  [180, 40,  100]),
    # --- Insects ---
    ([0,   0,   0],   [180, 80,  55]),
    # --- Toys / bright plastics (high saturation only) ---
    ([0,   200, 80],  [10,  255, 240]),
    ([160, 200, 80],  [180, 255, 240]),
    ([10,  200, 80],  [30,  255, 240]),
    ([85,  200, 80],  [130, 255, 240]),
    ([130, 200, 80],  [160, 255, 240]),
    ([145, 160, 80],  [165, 255, 240]),
    # --- White foam / styrofoam ---
    ([0,   0,   190], [180, 35,  230]),
]

REFLECTION_V_THRESHOLD        = 235
REFLECTION_VARIANCE_FRAMES    = 6
REFLECTION_VARIANCE_THRESHOLD = 18
CONFIRM_DILATE                = 12
MOTION_CONFIRM_RATIO          = 0.08
MAX_DECODE_ERRORS             = 10
RECONNECT_DELAY               = 3




# -------------------------
# SLEEP PREVENTION (best effort)
# -------------------------

class SleepInhibitor:
    def __init__(self):
        self.proc = None
        self.os_name = platform.system().lower()

    def start(self):
        try:
            if "windows" in self.os_name:
                # Prevent the machine from automatically sleeping while this process is active
                ES_CONTINUOUS = 0x80000000
                ES_SYSTEM_REQUIRED = 0x00000001
                ES_AWAYMODE_REQUIRED = 0x00000040
                ctypes.windll.kernel32.SetThreadExecutionState(
                    ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_AWAYMODE_REQUIRED
                )
                print("Sleep prevention enabled (Windows execution state)")
                return

            if "darwin" in self.os_name and shutil.which("caffeinate"):
                self.proc = subprocess.Popen(["caffeinate", "-dimsu"])
                print("Sleep prevention enabled via caffeinate")
                return

            if "linux" in self.os_name and shutil.which("systemd-inhibit"):
                self.proc = subprocess.Popen([
                    "systemd-inhibit",
                    "--what=sleep",
                    "--why=Pool monitor is running",
                    "bash",
                    "-lc",
                    "while true; do sleep 3600; done",
                ])
                print("Sleep prevention enabled via systemd-inhibit")
                return

            print("Sleep prevention unavailable on this OS/environment")
        except Exception as e:
            print(f"Sleep prevention setup failed: {e}")

    def stop(self):
        try:
            if "windows" in self.os_name:
                ES_CONTINUOUS = 0x80000000
                ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)
            if self.proc is not None:
                self.proc.terminate()
                self.proc.wait(timeout=2)
        except Exception:
            pass

# -------------------------
# TELEGRAM
# -------------------------

def send_alert(pct):
    message = f"⚠️ Pool debris detected\nPool coverage: {pct:.1f}%"
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    try:
        requests.post(url, data={"chat_id": CHAT_ID, "text": message}, timeout=5)
    except Exception as e:
        print(f"Alert failed: {e}")


# -------------------------
# OPENROUTER AI DETECTION
# -------------------------

def frame_to_jpeg_base64(frame):
    ok, buf = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    if not ok:
        return None
    return base64.b64encode(buf.tobytes()).decode("utf-8")


def extract_json(text):
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?", "", stripped).strip()
        stripped = re.sub(r"```$", "", stripped).strip()
    try:
        return json.loads(stripped)
    except Exception:
        m = re.search(r"\{[\s\S]*\}", text)
        if m:
            return json.loads(m.group(0))
    return None


def request_ai_debris_boxes(frame):
    if not OPENROUTER_API_KEY:
        return []

    image_b64 = frame_to_jpeg_base64(frame)
    if image_b64 is None:
        return []

    prompt = (
        "You are analyzing a pool image. The blue polygon outlines the valid pool surface area. "
        "Only identify debris inside that blue polygon. Ignore everything outside the polygon and ignore reflections. "
        "Return STRICT JSON only in this format: "
        "{\"debris\":[{\"label\":\"leaf\",\"bbox\":{\"x\":0.1,\"y\":0.2,\"w\":0.05,\"h\":0.04}}]} "
        "where x,y,w,h are normalized to [0,1] relative to full image width/height. "
        "If no debris exists, return {\"debris\":[]}"
    )

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": OPENROUTER_MODEL,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}}
            ]
        }],
        "temperature": 0,
        "response_format": {"type": "json_object"},
    }

    try:
        r = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=45)
        r.raise_for_status()
        data = r.json()
        text = data["choices"][0]["message"]["content"]
        parsed = extract_json(text)
        if not parsed or "debris" not in parsed:
            return []
        return parsed["debris"]
    except Exception as e:
        print(f"AI detection request failed: {e}")
        return []


def boxes_to_mask(boxes, frame_shape):
    h, w = frame_shape[:2]
    ai_mask = np.zeros((h, w), dtype=np.uint8)

    for item in boxes:
        bbox = item.get("bbox", {}) if isinstance(item, dict) else {}
        try:
            x = float(bbox.get("x", 0.0))
            y = float(bbox.get("y", 0.0))
            bw = float(bbox.get("w", 0.0))
            bh = float(bbox.get("h", 0.0))
        except Exception:
            continue

        x1 = int(max(min(x, 1.0), 0.0) * w)
        y1 = int(max(min(y, 1.0), 0.0) * h)
        x2 = int(max(min(x + bw, 1.0), 0.0) * w)
        y2 = int(max(min(y + bh, 1.0), 0.0) * h)

        if x2 <= x1 or y2 <= y1:
            continue
        cv2.rectangle(ai_mask, (x1, y1), (x2, y2), 255, -1)

    ai_mask = cv2.bitwise_and(ai_mask, get_pool_binary_mask(frame_shape))
    return ai_mask


# -------------------------
# MASK SAVE / LOAD
# -------------------------

def load_mask():
    if os.path.exists(MASK_FILE):
        with open(MASK_FILE, "r") as f:
            pts = json.load(f)
        print("Loaded saved pool mask")
        return np.array(pts)
    return None


def save_mask(pts):
    with open(MASK_FILE, "w") as f:
        json.dump(pts.tolist(), f)
    print("Pool mask saved")


# -------------------------
# WATER PROFILE SAVE / LOAD
# -------------------------

def load_water_profile():
    if os.path.exists(COLOR_PROFILE_FILE):
        with open(COLOR_PROFILE_FILE, "r") as f:
            p = json.load(f)
        mean = np.array(p["mean"])
        std  = np.array(p["std"])
        print(f"Loaded water color profile — mean HSV={mean.astype(int)}")
        return mean, std
    return None, None


def save_water_profile(mean, std):
    with open(COLOR_PROFILE_FILE, "w") as f:
        json.dump({"mean": mean.tolist(), "std": std.tolist()}, f)
    print(f"Water color profile saved — mean HSV={mean.astype(int)}")


# -------------------------
# SETUP HELPERS
# These run on the main thread BEFORE the processing thread starts,
# so there's no contention over cv2 windows.
# -------------------------

def run_pool_mask_setup(cap):
    """Interactive polygon selection. Returns np.array of points."""
    ret, frame = cap.read()
    if not ret:
        print("Could not read frame for mask setup")
        exit()

    points = []

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))

    cv2.namedWindow("Select Pool Area — click corners then press ENTER")
    cv2.setMouseCallback("Select Pool Area — click corners then press ENTER", mouse_callback)
    print("Click the corners of the pool, then press ENTER")

    while True:
        display = frame.copy()
        for p in points:
            cv2.circle(display, p, 5, (0, 0, 255), -1)
        if len(points) > 1:
            cv2.polylines(display, [np.array(points)], False, (255, 0, 0), 2)
        cv2.putText(display, f"{len(points)} points — press ENTER when done",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.imshow("Select Pool Area — click corners then press ENTER", display)
        if cv2.waitKey(16) == 13 and len(points) >= 3:
            break

    cv2.destroyAllWindows()
    polygon = np.array(points)
    save_mask(polygon)
    return polygon


def run_water_calibration(cap):
    """Interactive water color sampling. Returns (mean, std) HSV arrays."""
    ret, frame = cap.read()
    if not ret:
        print("Could not read frame for calibration")
        exit()

    hsv      = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    samples  = []
    cal_pts  = []

    def cal_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            cal_pts.append((x, y))
            samples.append(hsv[y, x].astype(np.float32))
            print(f"  Water sample {len(samples)}: HSV={hsv[y, x]}")

    cv2.namedWindow("Calibrate Water Color — click clean water then ENTER")
    cv2.setMouseCallback("Calibrate Water Color — click clean water then ENTER", cal_click)
    print("Click on clean water areas (at least 3), then press ENTER")

    while True:
        disp = frame.copy()
        for p in cal_pts:
            cv2.circle(disp, p, 6, (255, 255, 0), -1)
        cv2.putText(disp, f"{len(samples)} samples — press ENTER when done (need 3+)",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.imshow("Calibrate Water Color — click clean water then ENTER", disp)
        if cv2.waitKey(16) == 13 and len(samples) >= 3:
            break

    cv2.destroyAllWindows()

    arr  = np.array(samples)
    mean = arr.mean(axis=0)
    std  = arr.std(axis=0)
    save_water_profile(mean, std)
    return mean, std


# -------------------------
# CONNECT CAMERA
# -------------------------

cap = cv2.VideoCapture(STREAM_URL)
if not cap.isOpened():
    print("Camera connection failed")
    exit()


# -------------------------
# SETUP (all on main thread, before any processing thread starts)
# -------------------------

POOL_POLYGON = load_mask()
if POOL_POLYGON is None:
    POOL_POLYGON = run_pool_mask_setup(cap)

water_mean, water_std = load_water_profile()
if water_mean is None:
    water_mean, water_std = run_water_calibration(cap)

# Compute pool area in pixels once — used for percentage calculation
_pool_mask_ref = np.zeros((1080, 1920), dtype=np.uint8)   # placeholder shape
POOL_AREA_PX   = None   # computed from first real frame


def get_pool_area(frame_shape):
    global POOL_AREA_PX
    if POOL_AREA_PX is None:
        m = np.zeros(frame_shape[:2], dtype=np.uint8)
        cv2.fillPoly(m, [POOL_POLYGON], 255)
        POOL_AREA_PX = max(int(np.count_nonzero(m)), 1)
    return POOL_AREA_PX


# -------------------------
# POOL BINARY MASK
# -------------------------

POOL_BINARY_MASK = None


def get_pool_binary_mask(frame_shape):
    global POOL_BINARY_MASK
    if POOL_BINARY_MASK is None:
        m = np.zeros(frame_shape[:2], dtype=np.uint8)
        cv2.fillPoly(m, [POOL_POLYGON], 255)
        POOL_BINARY_MASK = m
        # Also set pool area while we're here
        get_pool_area(frame_shape)
    return POOL_BINARY_MASK


def mask_pool(frame):
    return cv2.bitwise_and(frame, frame, mask=get_pool_binary_mask(frame.shape))


# -------------------------
# REFLECTION SUPPRESSION
# -------------------------

v_buffer = deque(maxlen=REFLECTION_VARIANCE_FRAMES)


def build_reflection_mask(hsv_frame):
    v_channel = hsv_frame[:, :, 2]
    v_buffer.append(v_channel.astype(np.float32))
    if len(v_buffer) < 3:
        return np.zeros(v_channel.shape, dtype=np.uint8)
    variance     = np.std(np.stack(v_buffer, axis=0), axis=0)
    bright_mask  = (v_channel > REFLECTION_V_THRESHOLD).astype(np.uint8) * 255
    flicker_mask = (variance  > REFLECTION_VARIANCE_THRESHOLD).astype(np.uint8) * 255
    reflection   = cv2.bitwise_and(bright_mask, flicker_mask)
    return cv2.dilate(reflection, np.ones((9, 9), np.uint8))


# -------------------------
# HSV DEBRIS DETECTION
# -------------------------

def build_debris_mask(hsv_frame):
    debris = np.zeros(hsv_frame.shape[:2], dtype=np.uint8)
    for (lo, hi) in DEBRIS_COLOR_RANGES:
        debris |= cv2.inRange(hsv_frame,
                              np.array(lo, dtype=np.uint8),
                              np.array(hi, dtype=np.uint8))

    sigma     = 2.5
    water_lo  = np.clip(water_mean - sigma * water_std, 0, 255).astype(np.uint8)
    water_hi  = np.clip(water_mean + sigma * water_std, 0, 255).astype(np.uint8)
    water_mask = cv2.inRange(hsv_frame, water_lo, water_hi)

    debris = cv2.bitwise_and(debris, cv2.bitwise_not(water_mask))
    debris = cv2.bitwise_and(debris, cv2.bitwise_not(build_reflection_mask(hsv_frame)))
    debris = cv2.bitwise_and(debris, get_pool_binary_mask(hsv_frame.shape))

    k = np.ones((5, 5), np.uint8)
    debris = cv2.morphologyEx(debris, cv2.MORPH_OPEN,  k)
    debris = cv2.morphologyEx(debris, cv2.MORPH_CLOSE, k)
    return debris


def build_debris_mask_debug(hsv_frame):
    """Returns final mask plus intermediates for debug view."""
    raw = np.zeros(hsv_frame.shape[:2], dtype=np.uint8)
    for (lo, hi) in DEBRIS_COLOR_RANGES:
        raw |= cv2.inRange(hsv_frame,
                           np.array(lo, dtype=np.uint8),
                           np.array(hi, dtype=np.uint8))

    sigma      = 2.5
    water_lo   = np.clip(water_mean - sigma * water_std, 0, 255).astype(np.uint8)
    water_hi   = np.clip(water_mean + sigma * water_std, 0, 255).astype(np.uint8)
    water_mask = cv2.inRange(hsv_frame, water_lo, water_hi)
    refl_mask  = build_reflection_mask(hsv_frame)

    final = cv2.bitwise_and(raw,   cv2.bitwise_not(water_mask))
    final = cv2.bitwise_and(final, cv2.bitwise_not(refl_mask))
    final = cv2.bitwise_and(final, get_pool_binary_mask(hsv_frame.shape))
    k     = np.ones((5, 5), np.uint8)
    final = cv2.morphologyEx(final, cv2.MORPH_OPEN,  k)
    final = cv2.morphologyEx(final, cv2.MORPH_CLOSE, k)

    return final, raw, water_mask, refl_mask


# -------------------------
# MOTION MASK (secondary confirmer)
# -------------------------

bg_subtractor = cv2.createBackgroundSubtractorMOG2(
    history=800, varThreshold=30, detectShadows=True
)
fg_buffer = deque(maxlen=4)


def build_motion_mask(masked_frame):
    fg = bg_subtractor.apply(masked_frame)
    _, fg = cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY)
    fg_buffer.append(fg.astype(np.float32))
    accumulated = np.mean(fg_buffer, axis=0)
    _, motion = cv2.threshold(accumulated, 80, 255, cv2.THRESH_BINARY)
    return motion.astype(np.uint8)


# -------------------------
# DEBUG PANEL
# -------------------------

def make_debug_panel(frame, raw, water_mask, refl_mask, final_mask):
    h, w   = frame.shape[:2]
    th, tw = h // 2, w // 2

    def tile(mask, color, title):
        t = np.zeros((h, w, 3), dtype=np.uint8)
        t[mask > 0] = color
        t = cv2.resize(t, (tw, th))
        cv2.putText(t, title, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        return t

    top = np.hstack([tile(raw,        (0, 200, 255), "1. Raw color match"),
                     tile(water_mask, (255, 180, 0),  "2. Water exclusion")])
    bot = np.hstack([tile(refl_mask,  (0, 80,  255), "3. Reflection mask"),
                     tile(final_mask, (0, 255,  80), "4. Final debris mask")])
    return np.vstack([top, bot])


# -------------------------
# FRAME READER WITH RECONNECT
# -------------------------

def read_frame_safe(capture):
    ret, frame = capture.read()
    if not ret or frame is None:
        return False, None
    if frame.std() < 1.0:
        return False, None
    return True, frame


# -------------------------
# SHARED STATE
# -------------------------

lock             = threading.Lock()
latest_display   = None
latest_debug     = None
recalibrate_flag = False
debug_mode       = False
alert_sent       = False
last_check       = time.time()
last_ai_check    = 0
ai_debris_mask   = None
ai_debris_pct    = 0.0

print("\nPool monitor running")
print("Press 'c' to recalibrate water color")
print("Press 'd' to toggle debug view")
print("Press ESC to quit\n")

sleep_inhibitor = SleepInhibitor()
sleep_inhibitor.start()
atexit.register(sleep_inhibitor.stop)


# -------------------------
# PROCESSING THREAD
# -------------------------

def processing_loop():
    global alert_sent, last_check, latest_display, latest_debug
    global recalibrate_flag, water_mean, water_std, cap
    global last_ai_check, ai_debris_mask, ai_debris_pct

    consecutive_errors = 0

    while True:

        ok, frame = read_frame_safe(cap)

        if not ok:
            consecutive_errors += 1
            if consecutive_errors >= MAX_DECODE_ERRORS:
                print(f"Stream error — reconnecting...")
                cap.release()
                time.sleep(RECONNECT_DELAY)
                cap = cv2.VideoCapture(STREAM_URL)
                consecutive_errors = 0
                if cap.isOpened():
                    print("Reconnected")
                else:
                    print("Reconnect failed — retrying in 5s")
                    time.sleep(5)
            continue

        consecutive_errors = 0

        with lock:
            do_recal = recalibrate_flag
            is_debug = debug_mode
            recalibrate_flag = False

        # Recalibration is triggered by main thread but executed here so the
        # calibration window opens on the main thread via a flag — see below.
        # (Recal window is handled in the main loop instead.)

        display     = frame.copy()
        masked      = mask_pool(frame)
        hsv         = cv2.cvtColor(masked, cv2.COLOR_BGR2HSV)
        motion_mask = build_motion_mask(masked)

        if is_debug:
            debris_mask, raw, water_mask, refl_mask = build_debris_mask_debug(hsv)
        else:
            debris_mask = build_debris_mask(hsv)

        # Find contours and apply motion confirmation
        contours, _ = cv2.findContours(debris_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        confirmed_area = 0

        for c in contours:
            area = cv2.contourArea(c)
            if area < MIN_CONTOUR_AREA:
                continue

            if MOTION_CONFIRM_RATIO > 0:
                blob = np.zeros(frame.shape[:2], dtype=np.uint8)
                cv2.drawContours(blob, [c], -1, 255, -1)
                dilated        = cv2.dilate(blob, np.ones((CONFIRM_DILATE, CONFIRM_DILATE), np.uint8))
                overlap        = np.count_nonzero(cv2.bitwise_and(dilated, motion_mask))
                blob_px        = np.count_nonzero(blob)
                if blob_px > 0 and (overlap / blob_px) < MOTION_CONFIRM_RATIO:
                    continue

            confirmed_area += area
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(display, (x, y), (x + w, y + h), (0, 165, 255), 2)

        # Convert to percentage of pool area
        pool_area  = get_pool_area(frame.shape)
        debris_pct = min((confirmed_area / pool_area) * 100.0, 100.0)

        # Periodic alert
        now = time.time()
        if now - last_check > CHECK_INTERVAL:
            print(f"Debris coverage: {debris_pct:.1f}%")
            if debris_pct >= DEBRIS_THRESHOLD and not alert_sent:
                send_alert(debris_pct)
                print("Phone alert sent")
                alert_sent = True
            if debris_pct < DEBRIS_THRESHOLD:
                alert_sent = False
            last_check = now

        if OPENROUTER_API_KEY and (now - last_ai_check > AI_CHECK_INTERVAL):
            ai_boxes = request_ai_debris_boxes(display)
            new_ai_mask = boxes_to_mask(ai_boxes, frame.shape)
            ai_area = np.count_nonzero(new_ai_mask)
            ai_debris_mask = new_ai_mask
            ai_debris_pct = min((ai_area / pool_area) * 100.0, 100.0)
            print(f"AI debris coverage: {ai_debris_pct:.1f}%")
            last_ai_check = now

        # Overlay
        cv2.polylines(display, [POOL_POLYGON], True, (255, 0, 0), 2)

        bar_color = (0, 0, 255) if debris_pct >= DEBRIS_THRESHOLD else (0, 255, 0)
        cv2.putText(display, f"Debris: {debris_pct:.1f}%", (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, bar_color, 2)

        if OPENROUTER_API_KEY:
            cv2.putText(display, f"AI Debris: {ai_debris_pct:.1f}%", (10, 85),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)
            if ai_debris_mask is not None:
                ai_contours, _ = cv2.findContours(ai_debris_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for c in ai_contours:
                    area = cv2.contourArea(c)
                    if area < MIN_CONTOUR_AREA:
                        continue
                    x, y, w, h = cv2.boundingRect(c)
                    cv2.rectangle(display, (x, y), (x + w, y + h), (255, 0, 255), 2)

        # Small coverage bar under the text
        bar_w = 200
        filled = int(bar_w * debris_pct / 100.0)
        cv2.rectangle(display, (10, 45), (10 + bar_w, 58), (60, 60, 60), -1)
        if filled > 0:
            cv2.rectangle(display, (10, 45), (10 + filled, 58), bar_color, -1)

        with lock:
            latest_display = display
            if is_debug:
                latest_debug = make_debug_panel(frame, raw, water_mask, refl_mask, debris_mask)
            else:
                latest_debug = None


# -------------------------
# START PROCESSING THREAD
# -------------------------

proc_thread = threading.Thread(target=processing_loop, daemon=True)
proc_thread.start()


# -------------------------
# MAIN THREAD — display + input only
# waitKey(16) keeps the window responsive at ~60fps.
# Calibration is also handled here so cv2 windows stay on one thread.
# -------------------------

while True:
    with lock:
        frame_to_show = latest_display
        debug_to_show = latest_debug if debug_mode else None

    if frame_to_show is not None:
        cv2.imshow("Pool Monitor", frame_to_show)

    if debug_to_show is not None:
        cv2.imshow("Debug Masks", debug_to_show)

    key = cv2.waitKey(16)

    if key == 27:   # ESC
        break

    elif key == ord('c'):
        # Run calibration on main thread (cv2 windows must stay on one thread)
        ret, cal_frame = cap.read()
        if ret:
            new_mean, new_std = run_water_calibration(cap)
            with lock:
                water_mean = new_mean
                water_std  = new_std

    elif key == ord('d'):
        with lock:
            debug_mode = not debug_mode
        print("Debug mode:", "ON" if debug_mode else "OFF")


cap.release()
sleep_inhibitor.stop()
cv2.destroyAllWindows()
