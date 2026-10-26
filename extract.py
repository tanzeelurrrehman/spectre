import cv2
import json
from pathlib import Path
from copy import deepcopy

# ==========================================
# CONFIGURATION
# ==========================================

ROOT = Path(__file__).resolve().parent

video_path = ROOT / "rawCapture/p1ToD4.mp4"
config_path = ROOT / "hud/hud_regions.json"
output_path = ROOT / "dataset/match_0001.json"

sample_fps = 1
start_time = float(9*60 + 00)
end_time = float(9*60 + 25)

min_confidence = 0.85

# ==========================================
# OBSERVATION STRUCTURE
# ==========================================

observation_template = {
    "timestamp": None,
    "frame_index": None,

    "self": {
        "legend": None,
        "health": None,
        "shield": None,
        "weapons": [],
        "inventory": []
    },

    "teammate1": {
        "legend": None,
        "health": None,
        "shield": None
    },

    "teammate2": {
        "legend": None,
        "health": None,
        "shield": None
    },

    "confidence": None
}

match_memory = {
    "metadata": {
        "game": "Apex Legends",
        "map": None,
        "mode": None
    },
    "timeline": []
}

with open(config_path) as f:
    config = json.load(f)

# ==========================================
# SCREEN CLASSIFIER
# ==========================================

screen_identifiers = {}

for name, screen in config["screens"].items():

    identifier = screen.get("screen_identifier")
    if not identifier:
        continue

    x, y, w, h = identifier["roi"]

    # Support paths relative to either hud/ or project root
    path = Path(identifier["template"])
    if not path.is_absolute():
        hud_path = config_path.parent / path
        root_path = ROOT / path
        path = hud_path if hud_path.is_file() else root_path

    template = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

    if template is None:
        raise FileNotFoundError(path)

    if template.shape != (h, w):
        raise ValueError(f"Template size mismatch: {name}")

    screen_identifiers[name] = {
        "roi": (x, y, w, h),
        "template": template,
        "threshold": identifier.get("threshold", 0.90)
    }

print("Loaded screen identifiers:", list(screen_identifiers))


def detect_screen(frame):

    matches = []

    for name, data in screen_identifiers.items():

        x, y, w, h = data["roi"]
        crop = frame[y:y+h, x:x+w]

        if crop.shape[:2] != (h, w):
            continue

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

        # Reject nearly featureless patches
        if gray.std() < 5 or data["template"].std() < 5:
            continue

        score = float(cv2.matchTemplate(
            gray,
            data["template"],
            cv2.TM_CCOEFF_NORMED
        )[0, 0])

        if np.isfinite(score) and score >= data["threshold"]:
            matches.append((name, score))

    matches.sort(key=lambda item: item[1], reverse=True)

    if matches:
        if len(matches) > 1 and matches[0][1] - matches[1][1] < 0.05:
            return "unknown", matches[0][1]

        return matches[0]

    # Fallback: identify gameplay using self-legend portrait
    legend, score = recognize_legend(frame, "self")

    if legend is not None and score >= 0.90:
        return "gameplay", score

    return "unknown", score

# ==========================================
# LEGEND EXTRACTOR
# ==========================================

# Load and resize all templates ONCE
legend_data = {}

for player in ["self", "tm1", "tm2"]:

    region = config["screens"]["gameplay"][player]["legend"]
    x, y, w, h = region["roi"]

    templates = {}

    for name, path in region["templates"].items():

        template_path = config_path.parent / path
        template = cv2.imread(str(template_path), cv2.IMREAD_GRAYSCALE)

        if template is None:
            raise FileNotFoundError(template_path)

        templates[name] = cv2.resize(template, (w, h))

    legend_data[player] = {
        "roi": (x, y, w, h),
        "templates": templates
    }

    print(f"{player}: loaded {len(templates)} templates")


def recognize_legend(frame, player):

    data = legend_data[player]
    x, y, w, h = data["roi"]

    crop = frame[y:y+h, x:x+w]

    if crop.shape[:2] != (h, w):
        return None, 0.0

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    scores = {}

    for name, template in data["templates"].items():

        scores[name] = float(
            cv2.matchTemplate(
                gray,
                template,
                cv2.TM_CCOEFF_NORMED
            )[0, 0]
        )

    best = max(scores, key=scores.get)
    confidence = scores[best]

    return (
        best if confidence >= min_confidence else None,
        confidence
    )

# ==========================================
# SHIELD EXTRACTOR
# ==========================================

import numpy as np

shield_data = {}

# Load each player's six shield references once
for player in ["self", "tm1", "tm2"]:

    types = config["screens"]["gameplay"][player]["shield"]["types"]

    shield_data[player] = {
        name: data
        for name, data in types.items()
        if data.get("roi") and data.get("median_hsv")
    }

    print(f"{player}: loaded {len(shield_data[player])} shield references")

def recognize_shield(frame, player):

    candidates = []

    for name, data in shield_data[player].items():

        x, y, w, h = data["roi"]
        crop = frame[y:y+h, x:x+w]

        if crop.shape[:2] != (h, w):
            continue

        # Ignore upper/lower borders
        crop = crop[h//4 : 3*h//4]

        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV).astype(np.int16)

        ref_h, ref_s, ref_v = data["median_hsv"]

        dh = np.abs(hsv[:, :, 0] - ref_h)
        dh = np.minimum(dh, 180 - dh)

        ds = np.abs(hsv[:, :, 1] - ref_s)
        dv = np.abs(hsv[:, :, 2] - ref_v)

        if ref_s < 45:
            mask = (
                (ds < 45) &
                (dv < 65) &
                (hsv[:, :, 2] > 90)
            )
        else:
            mask = (
                (dh < 12) &
                (ds < 65) &
                (dv < 75)
            )

        # Is each horizontal column filled?
        columns = np.mean(mask, axis=0) > 0.45

        # Start near the left edge (allow small border padding)
        start = None

        for i in range(min(6, w)):
            if columns[i]:
                start = i
                break

        if start is None:
            continue

        # Track contiguous fill from left to right
        last_filled = start
        gap = 0
        max_gap = 4  # Tolerate small shield-segment gaps

        for i in range(start + 1, w):

            if columns[i]:
                last_filled = i
                gap = 0
            else:
                gap += 1

                if gap > max_gap:
                    break

        fill_fraction = (last_filled + 1) / w

        candidates.append({
            "type": name,
            "fill_fraction": float(fill_fraction),
            "max": data.get("max")
        })

    if not candidates:
        return {
            "type": None,
            "current": None,
            "max": None,
            "fill_percent": 0.0
        }, 0.0

    # Select the reference with the longest left-anchored match
    best = max(candidates, key=lambda c: c["fill_fraction"])

    maximum = best["max"]

    current = (
        round(best["fill_fraction"] * maximum)
        if maximum is not None else None
    )

    return {
        "type": best["type"],
        "current": current,
        "max": maximum,
        "fill_percent": round(best["fill_fraction"] * 100, 1)
    }, best["fill_fraction"]

# ==========================================
# WEAPON EXTRACTOR
# ==========================================

import numpy as np
import pytesseract
import re
from difflib import get_close_matches

weapon_config = config["screens"]["gameplay"]["self"]["weapons"]

WEAPON_NAMES = [
    "R-301", "FLATLINE", "HEMLOK", "HAVOC", "NEMESIS",
    "PROWLER", "R-99", "CAR", "ALTERNATOR", "VOLT",
    "PEACEKEEPER", "MASTIFF", "EVA-8", "MOZAMBIQUE",
    "WINGMAN", "P2020", "RE-45",
    "SPITFIRE", "DEVOTION", "L-STAR", "RAMPAGE",
    "G7 SCOUT", "30-30 REPEATER", "TRIPLE TAKE",
    "LONGBOW", "SENTINEL", "CHARGE RIFLE", "KRABER"
]

# Persistent state across sampled frames
weapon_state = {
    "weapon1": {"name": None, "magazine": None, "reserve": None},
    "weapon2": {"name": None, "magazine": None, "reserve": None},
    "active_weapon": None
}


def weapon_crop(frame, key):
    x, y, w, h = weapon_config[key]["roi"]
    return frame[y:y+h, x:x+w]


def read_text(image, psm=7):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    gray = cv2.resize(
        gray, None, fx=3, fy=3,
        interpolation=cv2.INTER_CUBIC
    )

    gray = cv2.threshold(
        gray, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )[1]

    return pytesseract.image_to_string(
        gray,
        config=f"--psm {psm}"
    ).strip().upper()


def read_weapon_name(image):

    text = read_text(image)

    # Remove leading slot number and punctuation
    text = re.sub(r"^[12]\s*", "", text)
    text = re.sub(r"[^A-Z0-9-]", "", text)

    if not text:
        return None

    choices = {
        re.sub(r"[^A-Z0-9]", "", name): name
        for name in WEAPON_NAMES
    }

    normalized = re.sub(r"[^A-Z0-9]", "", text)

    matches = get_close_matches(
        normalized,
        list(choices),
        n=1,
        cutoff=0.70
    )

    return choices[matches[0]] if matches else None


def read_ammo(image):

    h, w = image.shape[:2]

    # Exclude ammo-type icon on the right
    number = image[:, :int(w * 0.60)]

    gray = cv2.cvtColor(number, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(
        gray, None,
        fx=4, fy=4,
        interpolation=cv2.INTER_CUBIC
    )

    candidates = []

    for variant in [
        gray,
        cv2.threshold(
            gray, 0, 255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )[1],
        cv2.threshold(
            gray, 0, 255,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )[1]
    ]:

        for psm in [7, 8, 13]:

            data = pytesseract.image_to_data(
                variant,
                config=(
                    f"--psm {psm} "
                    "-c tessedit_char_whitelist=0123456789"
                ),
                output_type=pytesseract.Output.DICT
            )

            for text, confidence in zip(
                data["text"], data["conf"]
            ):
                text = text.strip()

                if text.isdigit() and float(confidence) >= 50:
                    candidates.append(
                        (int(text), float(confidence))
                    )

    if not candidates:
        return None

    # Prefer agreement between preprocessing methods
    from collections import defaultdict

    scores = defaultdict(float)

    for value, confidence in candidates:
        scores[value] += confidence

    best = max(scores, key=scores.get)

    return best


def get_color(image, start_fraction=0.0, end_fraction=1.0):

    h, w = image.shape[:2]

    crop = image[
        max(0, h // 4):max(1, 3 * h // 4),
        int(w * start_fraction):int(w * end_fraction)
    ]

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

    # Exclude white text, black background, and gray pixels
    mask = (
        (hsv[:, :, 1] > 40) &
        (hsv[:, :, 2] > 45) &
        (hsv[:, :, 2] < 245)
    )

    if not np.any(mask):
        return None

    # Median LAB color is useful for comparing different shades.
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)

    return np.median(
        lab[mask].astype(np.float32),
        axis=0
    )


def color_distance(a, b):

    if a is None or b is None:
        return float("inf")

    # Compare chromatic components rather than brightness
    return float(np.linalg.norm(a[1:] - b[1:]))

pending_reserve = {
    "weapon1": None,
    "weapon2": None
}

def update_reserve(slot, observed):

    if observed is None:
        return

    previous = weapon_state[slot]["reserve"]

    if previous is None or not isinstance(previous, int):
        weapon_state[slot]["reserve"] = observed
        pending_reserve[slot] = None
        return

    # Large sudden changes need confirmation
    suspicious = abs(observed - previous) > 30

    if suspicious:
        if pending_reserve[slot] == observed:
            weapon_state[slot]["reserve"] = observed
            pending_reserve[slot] = None
        else:
            pending_reserve[slot] = observed
    else:
        weapon_state[slot]["reserve"] = observed
        pending_reserve[slot] = None

# ==========================================
# WEAPON NAMES ONLY
# ==========================================

def recognize_weapons(frame):

    names = weapon_crop(frame, "names")

    if names.size == 0:
        return None, 0.0

    width = names.shape[1]
    half = width // 2

    name1 = read_weapon_name(names[:, :half])
    name2 = read_weapon_name(names[:, half:])

    return {
        "weapon1": name1,
        "weapon2": name2
    }, 1.0
    
EXTRACTORS = {
    "self.legend": lambda frame: recognize_legend(frame, "self"),
    "teammate1.legend": lambda frame: recognize_legend(frame, "tm1"),
    "teammate2.legend": lambda frame: recognize_legend(frame, "tm2"),

    "self.shield": lambda frame: recognize_shield(frame, "self"),
    "teammate1.shield": lambda frame: recognize_shield(frame, "tm1"),
    "teammate2.shield": lambda frame: recognize_shield(frame, "tm2"),

    "self.weapons": recognize_weapons,
}

# ==========================================
# VIDEO PROCESSING
# ==========================================

cap = cv2.VideoCapture(str(video_path))
assert cap.isOpened(), f"Cannot open {video_path}"

fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

assert fps > 0
assert 0 <= start_time < end_time <= total_frames / fps

start_frame = round(start_time * fps)
cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

current_frame = start_frame
sample_count = round((end_time - start_time) * sample_fps)

try:
    for i in range(sample_count):

        timestamp = start_time + i / sample_fps
        target_frame = round(timestamp * fps)

        # Advance sequentially
        while current_frame < target_frame:
            if not cap.grab():
                raise RuntimeError(f"Cannot advance to frame {target_frame}")
            current_frame += 1

        success, frame = cap.read()
        if not success:
            raise RuntimeError(f"Cannot read frame {target_frame}")

        current_frame += 1

        # Create observation
        observation = deepcopy(observation_template)
        observation["timestamp"] = round(timestamp, 3)
        observation["frame_index"] = target_frame

        screen, screen_score = detect_screen(frame)

        observation["screen"] = screen
        observation["screen_confidence"] = round(screen_score, 3)

        print(f"{timestamp:.1f}s | SCREEN: {screen} | Score: {screen_score:.3f}")

        if screen == "gameplay":
            # Run all registered gameplay extractors
            for field, extractor in EXTRACTORS.items():

                value, confidence = extractor(frame)

                player, attribute = field.split(".")

                observation[player][attribute] = value

                print(
                    f"{timestamp:.1f}s | {field}: {value} "
                    f"| Score: {confidence:.3f}"
                )

        match_memory["timeline"].append(observation)

finally:
    cap.release()


# ==========================================
# SAVE DATASET
# ==========================================

output_path.parent.mkdir(parents=True, exist_ok=True)

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(match_memory, f, indent=2)

print(f"\nSaved {len(match_memory['timeline'])} observations.")
print(f"Output: {output_path}")