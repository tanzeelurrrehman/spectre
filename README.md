
# SPECTRE
### A Vision-Language Tactical Coach for Apex Legends

SPECTRE is an offline AI system that analyzes recorded **Apex Legends** gameplay and generates timestamped tactical feedback using computer vision and a vision-language model (VLM).

The project explores a central question:

**Can structured gameplay observations help a general-purpose VLM produce more accurate, context-aware tactical analysis?**

Rather than requiring a VLM to interpret every part of a gameplay recording independently, SPECTRE extracts selected HUD information using conventional computer vision and supplies that information alongside gameplay frames.

**Project:** AI Systems Architecture — Utah State University  
**Status:** Working offline prototype

---

## 1. System Architecture

The prototype consists of two primary processing stages.

```text
Apex Legends Recording (.mp4)
              |
              v
     Input Validation
              |
              v
         extract.py
              |
      OpenCV Frame Sampling
              |
      Screen Classification
              |
       HUD Recognition
       |      |       |
    Legends Shields Weapons
       |      |       |
       +------+-------+
              |
              v
   Timestamped Match JSON
              |
              v
         analyze.py
              |
     Video Frame Retrieval
              |
    HUD + Frame Alignment
              |
     Multimodal Prompt
              |
              v
    Qwen VLM (llama.cpp)
              |
              v
    Response Postprocessing
              |
              v
    Tactical Analysis JSON
```

Execution logs expose screen classifications, matching scores, extracted observations, and model token usage.

### Why separate perception and reasoning?

Apex Legends has a visually complex interface containing small text, colored status indicators, weapon information, and several distinct screen layouts.

Conventional computer vision can efficiently extract information from known HUD locations. The VLM can then concentrate on the more difficult tasks: understanding positioning, engagements, teammate coordination, and tactical decisions.

This separation also makes debugging easier. HUD extraction can be tested independently of VLM inference.

---

## 2. Current Capabilities

| Component | Implementation | Status |
|---|---|---|
| Video ingestion | OpenCV | Implemented |
| Frame sampling | Configurable FPS and timestamps | Implemented |
| Screen classification | Template matching | Implemented |
| Squad legend recognition | Position-specific template matching | Implemented |
| Shield classification | Color-based comparison | Implemented |
| Shield strength estimation | Horizontal fill tracking | Experimental |
| Weapon identification | Tesseract OCR | Implemented |
| Timestamped observations | JSON | Implemented |
| Multimodal inference | Qwen via llama.cpp | Implemented |
| Sequential analysis | Ordered image batches | Implemented |
| Tactical feedback | Structured JSON | Experimental |
| Game-specific knowledge retrieval | Planned | Not implemented |
| Fight-aware sampling | Planned | Not implemented |
| Supervised fine-tuning | Planned | Not implemented |
| Real-time coaching | Planned | Not implemented |

The system is designed to return `unknown` when screen classification is insufficiently confident, rather than applying gameplay-specific detectors to unrelated interfaces.

---

## 3. Repository Structure

```text
spectre/
├── extract.py
├── analyze.py
├── hud/
│   ├── hud_regions.json
│   ├── preprocess.ipynb
│   └── [HUD reference templates]
├── dataset/
│   └── [generated JSON outputs]
├── rawCapture/
│   └── [local gameplay recordings]
├── reports/
│   ├── proposal.tex
│   └── proposal.pdf
├── requirements.txt
└── README.md
```

Large gameplay recordings, model weights, and generated match outputs are excluded from version control.

Reference templates and `hud_regions.json` are required for HUD extraction.

---

## 4. Requirements

- Ubuntu Linux
- Python 3.10+
- OpenCV
- NumPy
- Tesseract OCR
- pytesseract
- requests
- llama.cpp with multimodal support
- A compatible Qwen vision-language GGUF model and multimodal projector

### Install dependencies

```bash
sudo apt update
sudo apt install -y python3-venv tesseract-ocr libtesseract-dev

python3 -m venv .venv
source .venv/bin/activate

pip install opencv-python numpy pytesseract requests
```

Verify Tesseract:

```bash
tesseract --version
```

**Note:** Local inference performance depends on model size, quantization, available GPU memory, and the number and resolution of images submitted per request.

---

## 5. Running the Prototype

### Step 1 — Prepare the recording

Place an Apex Legends recording in `rawCapture/`.

The current configuration uses recordings at **3840 × 1600**, matching the coordinates stored in `hud_regions.json`.

Update the following values in `extract.py`:

```python
video_path = ROOT / "rawCapture/p1ToD4.mp4"
config_path = ROOT / "hud/hud_regions.json"
output_path = ROOT / "dataset/match_0001.json"

sample_fps = 1
start_time = 600
end_time = 610
```

The example processes ten seconds of gameplay at one frame per second.

Different recording resolutions may require recalibrating the HUD regions.

### Step 2 — Extract gameplay observations

```bash
python extract.py
```

The extractor:

1. Validates the video input and requested time interval.
2. Samples frames from the original recording.
3. Identifies the current screen.
4. Runs applicable HUD detectors.
5. Saves timestamped observations.

Example observation (illustrative):

```json
{
  "timestamp": 603.0,
  "frame_index": 36180,
  "screen": "gameplay",
  "self": {
    "legend": "wattson",
    "shield": {
      "type": "blue",
      "current": 73,
      "max": 75
    },
    "weapons": {
      "weapon1": "FLATLINE",
      "weapon2": "MASTIFF"
    }
  },
  "teammate1": {
    "legend": "catalyst"
  },
  "teammate2": {
    "legend": "conduit"
  }
}
```

Output:

```text
dataset/match_0001.json
```

### Step 3 — Start the VLM server

Build or install a multimodal-capable version of llama.cpp.

Start the server using your local model and projector paths:

```bash
llama-server \
  -m /path/to/model.gguf \
  --mmproj /path/to/mmproj.gguf \
  --host 127.0.0.1 \
  --port 8080 \
  -c 32768 \
  -ngl 99
```

Adjust GPU offloading, context size, and model-specific arguments for your hardware.

The analysis script expects the OpenAI-compatible endpoint:

```text
http://127.0.0.1:8080/v1/chat/completions
```

### Step 4 — Run tactical analysis

Verify the paths and batch settings in `analyze.py`.

```python
BATCH_SIZE = 10
MAX_BATCHES = 1
MAX_TOKENS = 4096
```

Run:

```bash
python -u analyze.py
```

The script retrieves the original video frames using frame indices stored in the match JSON.

**HUD extraction is not repeated during VLM analysis.**

The model receives:

- Chronologically ordered gameplay frames
- Corresponding timestamps
- Structured HUD observations
- Tactical analysis instructions
- Available context from preceding batches

The response is postprocessed into JSON.

Example output structure:

```json
{
  "summary": "Description of the gameplay segment.",
  "events": [
    "Timestamped observed event"
  ],
  "mistakes": [
    "Potential mistake supported by evidence"
  ],
  "improvements": [
    "Actionable tactical recommendation"
  ],
  "uncertainties": [
    "Information that cannot be established"
  ]
}
```

Output:

```text
dataset/analysis_0001.json
```

---

## 6. Early Experimental Observations

The prototype has been tested on recorded gameplay sequences involving movement, looting, inventory interaction, and combat.

### HUD recognition

In inspected footage, the template-based legend detector consistently recognized the player and teammates, with matching scores frequently exceeding 0.93.

These are template similarity scores, not measured classification accuracy across a complete dataset.

### Non-combat analysis

The VLM recognized movement between positions, looting interactions, inventory access, and subsequent repositioning.

It also identified that no combat was visible in the analyzed segment.

### Combat analysis

In a separate combat sequence, the VLM identified incoming damage, a downed teammate, and an enemy knockdown.

However, it failed to recognize that the player had eliminated the opposing squad.

This illustrates a major limitation of sparse temporal sampling and general-purpose visual reasoning.

---

## 7. Known Limitations

**Game-specific knowledge:** The VLM can misinterpret weapon mechanics. For example, it incorrectly characterized a full five-round Mastiff magazine as critically low ammunition.

**Temporal resolution:** Sampling at one frame per second can miss rapid combat events, including knockdowns, eliminations, and squad-wipe notifications.

**Missing observations:** An unmeasured HUD value must not be interpreted as zero. Earlier VLM experiments incorrectly treated `health: null` as evidence of player incapacitation.

**Shield estimation:** Color-based fill detection can be affected by HUD animations, segmented bars, and overlapping interface elements.

**Ammunition OCR:** Experiments revealed unreliable number recognition and incorrect association between ammunition counters and inactive weapons. Ammunition OCR is not part of the current primary extraction output.

**Tactical reliability:** Generated recommendations are experimental. A visually plausible interpretation does not guarantee an accurate understanding of the engagement.

---

## 8. Future Development

The next phase focuses on improving tactical reliability through:

1. **Game-specific knowledge:** Structured reference data for weapons, legends, abilities, and resource mechanics.
2. **Fight-aware sampling:** Higher frame rates around engagements, with context before and after decisive events.
3. **Temporal state tracking:** Separation of current observations, last-known values, and inferred events.
4. **Domain adaptation:** Annotated gameplay examples for evaluation and potential supervised fine-tuning.
5. **Deadline-aware inference:** Event-triggered analysis with the ability to suppress advice that arrives too late.

The long-term objective is a near-real-time tactical coach, but the present implementation is explicitly offline.

---

## 9. Research Scope

SPECTRE is an AI systems architecture prototype, not a production-ready coaching application.

Its current contribution is an end-to-end demonstration of how deterministic visual extraction and sequential vision-language inference can be integrated into a modular gameplay analysis pipeline.

The ongoing research question is whether combining **structured perception, game-specific knowledge, and temporal reasoning** can produce more reliable tactical feedback than direct VLM analysis of raw gameplay.

---

## Project Repository

https://github.com/tanzeelurrrehman/spectre
