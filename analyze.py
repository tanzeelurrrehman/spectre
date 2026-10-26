import cv2
import json
import base64
import requests
from textwrap import dedent
from pathlib import Path

# ==========================================
# CONFIGURATION
# ==========================================

ROOT = Path(__file__).resolve().parent

video_path = ROOT / "rawCapture/p1ToD4.mp4"
dataset_path = ROOT / "dataset/match_0001.json"
output_path = ROOT / "dataset/analysis_0001.json"

VLM_URL = "http://127.0.0.1:8080/v1/chat/completions"

BATCH_SIZE = 10
IMAGE_WIDTH = 1920
MAX_TOKENS = 8192

# Limit initial experiment to one batch.
# Set to None to analyze the entire dataset.
MAX_BATCHES = 1


# ==========================================
# LOAD DATA
# ==========================================

with open(dataset_path, "r") as f:
    match_data = json.load(f)

timeline = match_data["timeline"]

cap = cv2.VideoCapture(str(video_path))
assert cap.isOpened(), "Cannot open video"

analysis = {
    "metadata": match_data["metadata"],
    "segments": []
}


# ==========================================
# FRAME ENCODING
# ==========================================

def encode_frame(frame):

    h, w = frame.shape[:2]

    new_height = round(h * IMAGE_WIDTH / w)

    resized = cv2.resize(
        frame,
        (IMAGE_WIDTH, new_height),
        interpolation=cv2.INTER_AREA
    )

    success, buffer = cv2.imencode(
        ".jpg",
        resized,
        [cv2.IMWRITE_JPEG_QUALITY, 85]
    )

    if not success:
        raise RuntimeError("JPEG encoding failed")

    encoded = base64.b64encode(buffer).decode("utf-8")

    return f"data:image/jpeg;base64,{encoded}"


# ==========================================
# VLM ANALYSIS
# ==========================================

def analyze_batch(batch, previous_summaries):

    content = [{
        "type": "text",
        "text": (
            "You are SPECTRE, an Apex Legends tactical analyst.\n\n"
            "You will receive chronologically ordered gameplay frames, "
            "each paired with structured HUD observations.\n\n"

            "OBJECTIVES:\n"
            "1. Identify gameplay events visible in these frames.\n"
            "2. Describe player positioning, cover, engagements, "
            "and teammate coordination when visible.\n"
            "3. Identify tactical mistakes supported by visual evidence.\n"
            "4. Explain what the player could have done better.\n"
            "5. Distinguish observations from uncertain interpretations.\n\n"

            + dedent("""
            APEX LEGENDS HUD RULES:

            1. Two weapons may be equipped, but only one is active.
            2. The ammunition HUD belongs exclusively to the active weapon.
            3. Never assign visible ammunition to both weapons.
            4. When weapons switch, previously observed ammunition values
               for the inactive weapon become last-known values.
            5. A magazine count is not the total available ammunition.
            6. Flatline uses heavy ammunition.
            7. Mastiff uses shotgun ammunition.
            8. Never infer ammunition shortages solely from magazine capacity.
            9. If weapon identity or ammunition is uncertain, report unknown.
            10. Do not recommend actions based on uncertain ammunition readings.
            """)
            + "\n\nPrevious segment context:\n"
            + json.dumps(previous_summaries[-3:])
            + "\n\nAnalyze the following frames in order."
        )
    }]

    for observation in batch:

        timestamp = observation["timestamp"]
        frame_index = observation["frame_index"]

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)

        success, frame = cap.read()

        if not success:
            raise RuntimeError(
                f"Cannot read frame {frame_index}"
            )

        # Text observation immediately precedes its image
        content.append({
            "type": "text",
            "text": (
                f"\nTIMESTAMP: {timestamp}s\n"
                f"HUD: {json.dumps(observation)}\n"
                "Image for this timestamp:"
            )
        })

        content.append({
            "type": "image_url",
            "image_url": {
                "url": encode_frame(frame)
            }
        })

    content.append({
        "type": "text",
        "text": (
            "Return valid JSON with these fields:\n"
            "{\n"
            '  "summary": "brief description",\n'
            '  "events": [],\n'
            '  "mistakes": [],\n'
            '  "improvements": [],\n'
            '  "uncertainties": []\n'
            "}\n"
            "Include timestamps in event and mistake descriptions."
        )
    })

    payload = {
        "messages": [
            {
                "role": "user",
                "content": content
            }
        ],
        "temperature": 0.2,
        "max_tokens": MAX_TOKENS,
        "response_format": {
            "type": "json_object"
        },
        "stream": True,
        "stream_options": {
            "include_usage": True
        }
    }

    # ==========================================
    # STREAM VLM REASONING + ANSWER
    # ==========================================

    reasoning_parts = []
    answer_parts = []

    current_section = None
    usage = None

    with requests.post(
        VLM_URL,
        json=payload,
        stream=True,
        timeout=(30, 600)
    ) as response:

        response.raise_for_status()

        for line in response.iter_lines(decode_unicode=True):

            if not line:
                continue

            if not line.startswith("data:"):
                continue

            data = line[5:].strip()

            if data == "[DONE]":
                break

            try:
                chunk = json.loads(data)
            except json.JSONDecodeError:
                continue

            # Final token usage, if supplied by server
            if chunk.get("usage"):
                usage = chunk["usage"]

            choices = chunk.get("choices", [])

            if not choices:
                continue

            delta = choices[0].get("delta", {})

            # ----------------------------------
            # STREAM REASONING TOKENS
            # ----------------------------------

            reasoning = delta.get("reasoning_content")

            if reasoning:

                if current_section != "thinking":
                    print(
                        "\n\n========== VLM THINKING ==========\n",
                        flush=True
                    )
                    current_section = "thinking"

                print(reasoning, end="", flush=True)
                reasoning_parts.append(reasoning)

            # ----------------------------------
            # STREAM FINAL ANSWER TOKENS
            # ----------------------------------

            answer = delta.get("content")

            if answer:

                if current_section != "answer":
                    print(
                        "\n\n========== VLM ANSWER ==========\n",
                        flush=True
                    )
                    current_section = "answer"

                print(answer, end="", flush=True)
                answer_parts.append(answer)


    # ==========================================
    # FINALIZE RESPONSE
    # ==========================================

    reasoning_text = "".join(reasoning_parts)
    answer_text = "".join(answer_parts)

    print("\n")

    if usage:
        print("========== TOKEN USAGE ==========")
        print(json.dumps(usage, indent=2))

    if not answer_text.strip():
        raise RuntimeError(
            "VLM returned no final answer. "
            "Check reasoning format and max_tokens."
        )

    try:
        result = json.loads(answer_text)

    except json.JSONDecodeError as e:
        raise RuntimeError(
            f"VLM returned invalid JSON:\n{answer_text}"
        ) from e

    return result


# ==========================================
# SEQUENTIAL PROCESSING
# ==========================================

previous_summaries = []

try:

    batches = [
        timeline[i:i+BATCH_SIZE]
        for i in range(0, len(timeline), BATCH_SIZE)
    ]

    if MAX_BATCHES is not None:
        batches = batches[:MAX_BATCHES]

    for index, batch in enumerate(batches):

        start = batch[0]["timestamp"]
        end = batch[-1]["timestamp"]

        print(
            f"\nAnalyzing batch {index + 1}: "
            f"{start}s - {end}s",
            flush=True
        )

        result = analyze_batch(
            batch,
            previous_summaries
        )

        segment = {
            "start": start,
            "end": end,
            "analysis": result
        }

        analysis["segments"].append(segment)

        previous_summaries.append({
            "start": start,
            "end": end,
            "summary": result.get("summary", "")
        })

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        with open(output_path, "w") as f:
            json.dump(analysis, f, indent=2)

        print(json.dumps(result, indent=2))

finally:
    cap.release()

print(f"\nSaved: {output_path}")