# Sample Inspection Report - Human Readable Examples

**Date:** October 30, 2025  
**Purpose:** Understand exactly what the model sees and how it makes predictions

---

## 🔍 Key Discovery: Only First Frame Used!

**IMPORTANT:** The model does **NOT** use the full video. It only extracts and processes the **first frame** from each video file.

- Videos are 5-6 seconds long (125-152 frames at 25 FPS)
- Model extracts only frame #1
- This single frame is processed through BLIP-2 visual encoder
- Pre-extracted features (FaceMAE, VideoMAE, Audio) are computed from the **full video**

---

## 📊 Sample #1: samplenew_00006611

### Video Information
```
File: samplenew_00006611.mp4
Location: /export/home/scratch/qze/datasets/MER2023/test3/
Resolution: 1920x1080 (Full HD)
Duration: 5.00 seconds (125 frames @ 25 FPS)
Format: MP4

USED BY MODEL: First frame only (1080x1920x3 RGB image)
```

### Ground Truth
```
Emotion: sad
Emotion ID: 125
```

### What the Person Says (Transcription)
```
"Sorry, big sister, for causing trouble to everyone."
```

**Analysis:** Apologetic language, expressing regret → Aligns with "sad" emotion

### Visual Frame (What the Model Actually Sees)

**First Frame Extracted:**

![First Frame - samplenew_00006611](samplenew_00006611_first_frame.jpg)

*The actual first frame (frame #0) from the video that the BLIP-2 visual encoder processes. This single static image is resized to 448x448 and used for visual context, while the pre-extracted features below capture information from the entire 5-second video.*

### Features Used (Pre-extracted from full video)
```
1. FaceMAE (Facial Expression Features):
   - Dimensions: 1024
   - Range: [-6.61, 3.37]
   - Mean: -0.02, Std: 0.55
   - Purpose: Captures facial muscle movements, expression patterns

2. VideoMAE (Video Content Features):
   - Dimensions: 1024
   - Range: [-9.37, 16.17]
   - Mean: 0.00, Std: 1.00
   - Purpose: Temporal dynamics, scene context, body language

3. HuBERT-Large (Audio Features):
   - Dimensions: 1024
   - Range: [-1.69, 0.47]
   - Mean: -0.00, Std: 0.10
   - Purpose: Voice tone, prosody, acoustic patterns

Total: 3072 dimensions concatenated
```

### Actual Model Prompt
```
[INST] <video><VideoHere></video> <feature><FeatureHere></feature> 
The person in video says: Sorry, big sister, for causing trouble to everyone.. 
[emotion] Please determine which emotion label in the video represents: 
happy, sad, neutral, angry, worried, surprise, fear, contempt, doubt. 
[/INST]
```

**Special Tokens:**
- `<video><VideoHere></video>`: Replaced with visual features from first frame
- `<feature><FeatureHere></feature>`: Replaced with 3072-dim concatenated features
- `[INST] ... [/INST]`: Llama-2 instruction format

### Expected Output
```
sad
```

Single word response, no explanation.

---

## 📊 Sample #2: sample_00002328

### Video Information
```
File: sample_00002328.avi
Location: /export/home/scratch/qze/datasets/MER2023/test3/
Resolution: 1104x622
Duration: 6.08 seconds (152 frames @ 25 FPS)
Format: AVI

USED BY MODEL: First frame only (622x1104x3 RGB image)
```

### Ground Truth
```
Emotion: sad
Emotion ID: 150
```

### What the Person Says (Transcription)
```
"Dad, don't be angry. Ah, it's all over now. I won't have any more dealings with him."
```

**Analysis:** 
- Trying to calm someone ("don't be angry")
- Resignation ("it's all over now")
- Cutting ties ("won't have any more dealings")
- → Suggests sadness/regret about a relationship

### Features Used
```
1. FaceMAE: 1024 dimensions
   Range: [-7.13, 4.28], Mean: -0.01, Std: 0.49

2. VideoMAE: 1024 dimensions
   Range: [-10.85, 16.03], Mean: 0.00, Std: 1.00

3. Audio: 1024 dimensions
   Range: [-0.98, 0.27], Mean: -0.00, Std: 0.06
   
Note: Lower audio std (0.06 vs 0.10) suggests quieter/more subdued speech
```

### Actual Model Prompt
```
[INST] <video><VideoHere></video> <feature><FeatureHere></feature> 
The person in video says: Dad, don't be angry. Ah, it's all over now. 
I won't have any more dealings with him.. 
[emotion] Please determine which emotion label in the video represents: 
happy, sad, neutral, angry, worried, surprise, fear, contempt, doubt. 
[/INST]
```

### Expected Output
```
sad
```

---

## 🎯 How the Model Processes Each Sample

### Step-by-Step Pipeline

```
1. VIDEO LOADING
   ├─ Load video file (.mp4 or .avi)
   ├─ Extract ONLY first frame
   └─ Convert to RGB (1080x1920x3 or similar)

2. VISUAL PROCESSING
   ├─ Resize to 448x448 (BLIP-2 standard)
   ├─ Normalize pixel values
   └─ Pass through Q-Former → Visual embeddings

3. FEATURE LOADING
   ├─ Load FaceMAE features (1024-dim .npy file)
   ├─ Load VideoMAE features (1024-dim .npy file)
   ├─ Load Audio features (1024-dim .npy file)
   └─ Concatenate → 3072-dim vector

4. TEXT PREPARATION
   ├─ Lookup transcription in CSV
   ├─ Construct character line: "The person in video says: {text}."
   ├─ Add task marker: [emotion]
   ├─ Add instruction: "Please determine which emotion..."
   └─ Wrap in Llama-2 format: [INST] ... [/INST]

5. MULTIMODAL FUSION
   ├─ Visual embeddings from frame
   ├─ Feature embeddings from 3072-dim vector
   ├─ Text embeddings from prompt
   └─ Fused in MiniGPT-v2 architecture

6. GENERATION
   ├─ Llama-2-7b-chat generates response
   ├─ LoRA fine-tuning applied (64 rank, 16 alpha)
   ├─ Greedy decoding (do_sample=False)
   └─ Extract last word as emotion label

7. POST-PROCESSING
   ├─ Split response by spaces
   ├─ Take last word
   ├─ Validate against 6-class set
   └─ Default to "neutral" if invalid
```

---

## 🔑 Key Insights

### 1. Frame vs. Video Paradox ⚠️

**Problem:** Model only uses **first frame** but features are from **full video**

- **Visual input:** Single static image (first frame)
- **FaceMAE features:** Computed from entire video
- **VideoMAE features:** Temporal features across all frames
- **Audio features:** Full audio track analyzed

**Implication:** 
- The first frame might not be representative of the emotion
- Pre-extracted features capture the full emotional arc
- Visual and feature modalities may be misaligned

**Example:**
```
Video: Person crying at the end (frames 100-125)
First frame: Person looking neutral (frame 1)
Features: Capture the crying (averaged over full video)
Result: Model sees neutral face but "sad" features → Confusion
```

### 2. Transcription Importance

The transcription provides **crucial context**:

Sample #1: "Sorry, big sister, for causing trouble to everyone."
→ Apologetic, remorseful language strongly suggests sadness

Sample #2: "Dad, don't be angry. Ah, it's all over now."
→ Resignation, conflict resolution → Sadness/regret

**Impact:** 
- Text alone might be sufficient for many cases
- Visual features may be complementary, not primary
- Audio prosody adds emotional tone

### 3. Feature Dimensions Explained

```
FaceMAE (1024-dim):
├─ Facial Action Units (AU)
├─ Muscle movements
├─ Expression patterns
└─ Face geometry changes

VideoMAE (1024-dim):
├─ Temporal motion
├─ Scene context
├─ Body posture/gestures
└─ Background/environment

HuBERT-Large (1024-dim):
├─ Voice pitch/tone
├─ Speech rate
├─ Prosody (emotional intonation)
└─ Acoustic patterns
```

Each modality contributes 1/3 of the total feature representation.

### 4. Prompt Engineering

The prompt is carefully structured:

```
[Multimodal Tokens] + [Context] + [Task] + [Instruction]
```

Components:
1. **Multimodal tokens:** `<video>` and `<feature>` placeholders
2. **Context:** Transcription as "The person says: ..."
3. **Task marker:** `[emotion]` (vs `[reason]` for other tasks)
4. **Instruction:** Lists all possible emotion labels

**Why 9 classes in prompt but 6 in eval?**
- Training included 9-class data (fear, contempt, doubt)
- Evaluation uses 6-class subset (MER2023 standard)
- Post-processing filters invalid outputs

---

## 📈 Model Behavior Patterns

### What the Model Sees (Per Sample)

```
INPUT:
├─ Image: 448x448 RGB frame (after processing)
├─ Features: 3072-dimensional vector
│   ├─ Face: How facial expression looks
│   ├─ Video: How person moves/acts
│   └─ Audio: How voice sounds
└─ Text: What the person says + instruction

PROCESSING:
├─ Visual encoder extracts image features
├─ Feature encoder processes pre-computed features
├─ Text encoder processes transcription + instruction
├─ All fused in transformer layers
└─ Llama-2 generates emotion label

OUTPUT:
└─ Single word: neutral/angry/happy/sad/worried/surprise
```

### Generation Strategy

```
Greedy Decoding:
- No sampling (do_sample=False)
- Always picks highest probability token
- Deterministic output (same input → same output)

Max Tokens: 500
- Model can generate up to 500 tokens
- Usually generates 1-2 words
- Last word extracted as prediction

LoRA Fine-tuning:
- Only 33.5M parameters trained (0.495% of 6.8B)
- Target modules: q_proj, v_proj (attention layers)
- Rank 64, Alpha 16 → Controls adaptation strength
```

---

## 🎬 Visual Frame Analysis

You can view the actual frames extracted:

```bash
# Sample 1
display /tmp/samplenew_00006611_first_frame.jpg

# Sample 2  
display /tmp/sample_00002328_first_frame.jpg
```

These are the **exact images** the model's visual encoder processes.

**Note:** These frames are saved temporarily by `inspect_sample.py`

---

## 🧪 How to Inspect Any Sample

### By Index
```bash
python inspect_sample.py --sample_id 0     # First sample
python inspect_sample.py --sample_id 100   # Sample 100
python inspect_sample.py --sample_id 833   # Last sample
```

### By Video Name
```bash
python inspect_sample.py --sample_id samplenew_00006611
python inspect_sample.py --sample_id sample_00002328
```

### Output Includes
- ✅ Video properties (resolution, FPS, duration)
- ✅ Ground truth emotion
- ✅ Complete transcription
- ✅ Feature statistics (shape, range, mean, std)
- ✅ Actual prompt sent to model
- ✅ Expected output
- ✅ First frame saved to /tmp/

---

## 💡 Recommendations for Analysis

### 1. Check Temporal Alignment
Verify if first frame is representative:
```bash
# Extract multiple frames to see emotion progression
ffmpeg -i video.mp4 -vf "select='not(mod(n\,25))'" -vsync 0 frames/frame_%03d.jpg
```

### 2. Analyze Misclassifications
For errors in confusion matrix (e.g., Happy→Neutral):
```bash
# Find happy samples misclassified as neutral
# Then inspect with inspect_sample.py
# Check if first frame looks neutral despite "happy" label
```

### 3. Feature Importance
Compare feature statistics between correct and incorrect predictions:
- Do misclassified samples have unusual feature ranges?
- Is audio std significantly different for errors?

### 4. Transcription Analysis
Extract all transcriptions for each emotion class:
```python
# Check if text alone is predictive
# E.g., "sorry" often correlates with sadness
```

---

## 📝 Summary

### What We Learned

1. **Input:** Model uses **first frame only** + full-video features + transcription
2. **Features:** 3072 dimensions (Face 1024 + Video 1024 + Audio 1024)
3. **Prompt:** Structured with special tokens, context, task, and instruction
4. **Output:** Single emotion word extracted from generated text
5. **Visual Evidence:** See the actual first frame image above (samplenew_00006611_first_frame.jpg)
5. **Processing:** Multimodal fusion in MiniGPT-v2 + Llama-2 generation

### Critical Points

⚠️ **First Frame Limitation:** May not capture peak emotional expression  
✅ **Rich Features:** Pre-extracted features from full video compensate  
✅ **Strong Text Signal:** Transcription provides valuable context  
✅ **Multimodal Fusion:** Combines visual, audio, and linguistic cues  

---

**Generated:** October 30, 2025  
**Samples Inspected:** 2 (samplenew_00006611, sample_00002328)  
**Both Ground Truth:** sad  
**Tool:** `inspect_sample.py` - Run on any sample for detailed analysis
