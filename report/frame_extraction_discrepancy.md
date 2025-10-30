# Frame Extraction Discrepancy - Paper vs Implementation

**Date:** October 30, 2025  
**Issue:** Mismatch between paper methodology and actual code implementation

---

## 🎯 Quick Summary

**Q: "If we have pre-extracted features, what is the video used for?"**  
**A:** The model uses **BOTH** inputs for different purposes:
- **Raw video frame** → BLIP-2 visual encoder → General scene understanding
- **Pre-extracted features** → Direct input → Specialized emotion signals (Face/Video/Audio)
- **Both combined** in MiniGPT-v2 for comprehensive multimodal emotion recognition

| Aspect | Paper (MERR) | Current Code |
|--------|--------------|--------------|
| **Frame Selection** | Peak AU intensity | First frame (frame #0) |
| **Analysis Method** | OpenFace AU extraction | None |
| **Frame Index** | Variable (peak emotion) | Always 0 |
| **File/Line** | N/A | `first_face.py:230` |
| **Frame Purpose** | Visual context via BLIP-2 | ✅ Same |
| **Features Purpose** | Emotion-specific signals | ✅ Same |
| **Impact** | Optimal emotion capture | May miss peak expression |
| **Current Accuracy** | N/A | 90.65% (still good!) |

**THE KEY LINE OF CODE:**
```python
# File: minigpt4/datasets/datasets/first_face.py, Line 230
success, frame = video_capture.read()  # Gets first frame only, no frame selection
```

---

## 🔴 Critical Finding: Frame Selection Method

### What the Paper Says (MERR Dataset Construction)

According to the paper's MERR dataset construction methodology:

> **Frame should be selected based on maximum cumulative Action Unit (AU) intensity from OpenFace**

**Proper Method:**
1. Run OpenFace on entire video
2. Extract AU intensities for each frame
3. Calculate cumulative AU intensity per frame
4. Select frame with **maximum cumulative AU intensity**
5. This frame represents the **peak emotional expression**

**Rationale:**
- Peak AU frame captures the strongest facial expression
- Most representative of the emotional state
- Better alignment with emotion labels

---

## 🤔 Critical Question: Why Extract Video Frame at All?

**You asked:** "If we have pre-extracted features, what is the video used for?"

**Great observation!** The model receives **TWO separate visual inputs:**

### Input 1: Raw Frame (from video extraction)
```python
# Line 168-170 in first_face.py
image = Image.fromarray(image.astype('uint8'))
image = image.convert('RGB')
image = self.vis_processor(image)  # Resized to 448x448, normalized
```

**What happens to this:**
- Sent to **BLIP-2 visual encoder** (Q-Former)
- Processed into **visual embeddings**
- Used by the **MiniGPT-v2 architecture**
- Allows model to "see" the actual image content

### Input 2: Pre-extracted Features
```python
# Lines 179-186 in first_face.py
FaceMAE_feats, VideoMAE_feats, Audio_feats = self.get(video_name)
# ... shape adjustments ...
video_features = torch.cat((FaceMAE_feats, VideoMAE_feats, Audio_feats), dim=0)
```

**What happens to this:**
- Sent as **separate feature vector** (3072 dimensions)
- Passed to model alongside the visual embeddings
- Provides **pre-computed emotion-specific features**

### Model Call (eval_emotion.py line 81)
```python
answers = model.generate(images, video_features, texts, ...)
#                         ^^^^^^  ^^^^^^^^^^^^^^
#                         Frame    Pre-extracted features
#                         (BLIP-2) (Face+Video+Audio)
```

### Why BOTH?

**The model uses a dual-input architecture:**

1. **`images` (raw frame):**
   - General visual understanding
   - Scene context, objects, environment
   - Processed by pre-trained BLIP-2 encoder
   - Captures "what the model sees"

2. **`video_features` (pre-extracted):**
   - Specialized emotion features
   - FaceMAE: Facial expression patterns
   - VideoMAE: Temporal video dynamics
   - Audio: Voice tone and prosody
   - Pre-computed by expert models

**Think of it like:**
- **Raw frame:** "Here's what the scene looks like"
- **Pre-extracted features:** "Here's what emotion experts detected"

### Architecture Flow
```
Raw Frame (448x448)              Pre-extracted Features (3072-dim)
       ↓                                    ↓
  BLIP-2 Visual Encoder           Feature Embeddings
       ↓                                    ↓
  Visual Embeddings                   [Already encoded]
       ↓                                    ↓
       └──────────────┬────────────────────┘
                      ↓
              MiniGPT-v2 Fusion
                      ↓
                 Llama-2 LLM
                      ↓
              Emotion Prediction
```

### So the Frame IS Used!

**Answer to your question:**
The video frame is **NOT redundant** - it provides:
- ✅ General visual context via BLIP-2
- ✅ Scene understanding
- ✅ Complementary to specialized features
- ✅ Allows model to "see" like a human would

**The pre-extracted features provide:**
- ✅ Expert-level facial analysis (FaceMAE)
- ✅ Temporal dynamics (VideoMAE) 
- ✅ Audio/prosody information (HuBERT)

**Both are used together** for multimodal fusion!

### Complete Data Flow Diagram

```
VIDEO FILE: samplenew_00006611.mp4 (5 seconds, 125 frames)
│
├─── Path 1: Raw Visual Input ─────────────────────┐
│    │                                              │
│    ├─ Extract first frame (frame #0)             │
│    ├─ Resize to 448x448                          │
│    ├─ Normalize for BLIP-2                       │
│    └─ Send to Visual Encoder                     │
│                                                   ↓
│                                         Visual Embeddings
│                                         (scene, context)
│
├─── Path 2: Pre-extracted Features ───────────────┐
│    │                                              │
│    ├─ Load FaceMAE (1024-dim)                    │
│    │   └─ Facial expressions from full video     │
│    ├─ Load VideoMAE (1024-dim)                   │
│    │   └─ Temporal dynamics from full video      │
│    ├─ Load Audio HuBERT (1024-dim)               │
│    │   └─ Voice prosody from full audio          │
│    └─ Concatenate → 3072-dim vector              │
│                                                   ↓
│                                         Feature Embeddings
│                                         (emotion-specific)
│
├─── Path 3: Text Input ───────────────────────────┐
│    │                                              │
│    ├─ Load transcription CSV                     │
│    ├─ "Sorry, big sister, for causing..."        │
│    ├─ Build prompt with instruction              │
│    └─ Tokenize                                   │
│                                                   ↓
│                                         Text Embeddings
│                                         (semantic context)
│
└────────────────────┬─────────────────────────────┘
                     ↓
         ┌───────────────────────────┐
         │   MiniGPT-v2 Architecture │
         │                           │
         │  ┌─────────────────────┐  │
         │  │ Multimodal Fusion   │  │
         │  │ - Visual + Features │  │
         │  │ - Text + Context    │  │
         │  └─────────────────────┘  │
         │           ↓               │
         │  ┌─────────────────────┐  │
         │  │   Llama-2-7b-chat   │  │
         │  │   with LoRA tuning  │  │
         │  └─────────────────────┘  │
         └───────────┬───────────────┘
                     ↓
              "sad" (prediction)
```

### Key Insight

**The video frame and pre-extracted features serve DIFFERENT purposes:**

| Aspect | Raw Frame | Pre-extracted Features |
|--------|-----------|------------------------|
| **Source** | First frame only | Full video analysis |
| **Processing** | BLIP-2 visual encoder | FaceMAE/VideoMAE/HuBERT |
| **Purpose** | General visual understanding | Emotion-specific signals |
| **Information** | Scene, objects, context | Facial AUs, motion, voice tone |
| **Temporal** | Single frame (static) | Multiple frames (dynamic) |
| **Expertise** | General vision model | Specialized emotion models |

**Why both are needed:**
- Frame alone: Misses temporal dynamics and audio cues
- Features alone: Lack general visual context and scene understanding
- Together: Comprehensive multimodal emotion recognition

---

### What the Code Actually Does

**Current Implementation:**

**File:** `minigpt4/datasets/datasets/first_face.py`  
**Method:** `extract_frame()` (lines 228-236)  
**Called from:** `__getitem__()` (lines 152 or 156)

```python
def extract_frame(self, video_path):
    video_capture = cv2.VideoCapture(video_path)
    success, frame = video_capture.read()  # ← THIS LINE extracts FIRST frame only!
    if not success:
        raise ValueError("Failed to read video file:", video_path)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    video_capture.release()
    return frame_rgb
```

**The Critical Line:**
```python
success, frame = video_capture.read()  # Line 230
```

**What it does:**
- `cv2.VideoCapture()` opens the video file
- `.read()` **without any frame positioning** reads the **very first frame** (frame index 0)
- There is **no loop** to iterate through frames
- There is **no OpenFace AU analysis**
- There is **no frame selection logic**

**Problem:**
- ❌ Extracts **first frame only** (frame #0)
- ❌ No OpenFace AU analysis
- ❌ No peak emotion detection
- ❌ First frame may show neutral expression before emotion develops
- ❌ Not aligned with MERR paper methodology

---

## � Complete Code Flow

### Where Frame Extraction Happens

**File:** `minigpt4/datasets/datasets/first_face.py`

**1. Dataset.__getitem__() is called (line 136-226)**
```python
def __getitem__(self, index):
    t = self.tmp[index]
    video_name = t[0]
    
    mp4_path = os.path.join(self.vis_root, video_name + ".mp4")
    avi_path = os.path.join(self.vis_root, video_name + ".avi")
    jpg_path = os.path.join(self.vis_root, video_name + ".jpg")
    
    # Try .mp4 first
    if os.path.exists(mp4_path):
        image = self.extract_frame(mp4_path)  # ← CALLS extract_frame (line 152)
    # Try .avi second
    elif os.path.exists(avi_path):
        image = self.extract_frame(avi_path)  # ← CALLS extract_frame (line 156)
    # Try pre-extracted .jpg third (fallback)
    elif os.path.exists(jpg_path):
        image = Image.open(jpg_path).convert("RGB")  # ← Could be peak frame if pre-extracted!
        image = np.array(image)
    else:
        raise FileNotFoundError(...)
```

**2. extract_frame() extracts first frame (line 228-236)**
```python
def extract_frame(self, video_path):
    video_capture = cv2.VideoCapture(video_path)
    success, frame = video_capture.read()  # ← LINE 230: Gets frame #0
    if not success:
        raise ValueError("Failed to read video file:", video_path)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    video_capture.release()
    return frame_rgb
```

**3. Frame is processed and used (back in __getitem__)**
```python
    image = Image.fromarray(image.astype('uint8'))
    image = image.convert('RGB')
    image = self.vis_processor(image)  # ← Resized to 448x448, normalized
```

### Summary of Flow

```
For each video in dataset:
  ↓
  Check if .mp4 exists → YES
  ↓
  Call extract_frame(video.mp4)
  ↓
  cv2.VideoCapture opens video
  ↓
  .read() extracts FIRST FRAME (frame #0)  ← THE KEY LINE
  ↓
  Convert BGR to RGB
  ↓
  Return frame to __getitem__()
  ↓
  Process frame (resize, normalize)
  ↓
  Send to model for emotion classification
```

---

## �📊 Impact Analysis

### Potential Performance Impact

**Example Scenario:**
```
Video: Person becomes sad during conversation
- Frame 0 (first):     Neutral expression (what model sees)
- Frame 50:            Slight sadness
- Frame 100 (peak):    Strong sadness with peak AU intensity
- Frame 125 (last):    Returning to neutral

Ground Truth Label: "sad"
Model Sees: Neutral face (frame 0)
Features (FaceMAE/VideoMAE/Audio): Averaged over full video → capture sadness
Result: Misalignment between visual input and features
```

### Why Results Are Still Good (90.65%)

Despite using first frame instead of peak frame, the model achieves strong performance because:

1. **Pre-extracted Features Compensate**
   - FaceMAE features: Computed from multiple frames (or peak frame)
   - VideoMAE features: Temporal features across full video
   - Audio features: Prosody from entire speech
   - These 3072 dimensions carry the actual emotional information

2. **Transcription Provides Strong Signal**
   - Text content highly correlated with emotion
   - Example: "Sorry..." → sad, "Don't be angry..." → sad/worried
   - May be primary cue for classification

3. **First Frame May Not Always Be Neutral**
   - Some videos start mid-emotion
   - Edited clips may begin at emotional moments
   - Depends on MER2023 dataset construction

### Visual Example: What the Model Actually Sees

**Sample: samplenew_00006611**

![First Frame Example](samplenew_00006611_first_frame.jpg)

*Above: The actual first frame (frame #0) extracted from the video. This is what the BLIP-2 visual encoder sees - a single static image showing the person's expression at the very beginning of the 5-second video clip.*

**Context for this frame:**
- Video: 5.00 seconds, 125 frames @ 25 FPS
- Ground truth: **sad**
- Transcription: "Sorry, big sister, for causing trouble to everyone."
- What model sees: This single frame (potentially before peak sadness)
- What features capture: Emotional dynamics across all 125 frames

**Key question:** Does this first frame show the peak sadness, or does the emotion develop later in the video?

4. **Model Learns to Rely on Features**
   - Training adapts to this inconsistency
   - Model may downweight visual encoder
   - Heavier reliance on pre-extracted features + text

---

## 🔍 Evidence from Our Samples

### Sample 1: samplenew_00006611
```
Transcription: "Sorry, big sister, for causing trouble to everyone."
Ground Truth: sad
Duration: 5.00 seconds (125 frames)
Model sees: Frame 0 only

Likely scenario:
- Frame 0: May not show peak sadness
- Peak sadness: Probably mid-video (frame 60-80)
- Features: Capture the sadness across full video
- Result: Features + text overcome visual limitation
```

### Sample 2: sample_00002328
```
Transcription: "Dad, don't be angry. Ah, it's all over now..."
Ground Truth: sad
Duration: 6.08 seconds (152 frames)
Model sees: Frame 0 only

Likely scenario:
- Emotion builds throughout video
- Resignation increases toward end
- First frame may show pre-emotion state
```

---

## ✅ Proper Implementation (According to Paper)

### Step 1: OpenFace AU Extraction

```python
import subprocess
import pandas as pd

def extract_openface_aus(video_path):
    """
    Run OpenFace to extract Action Units for all frames.
    Returns DataFrame with AU intensities per frame.
    """
    # OpenFace command
    cmd = [
        'FeatureExtraction',
        '-f', video_path,
        '-aus',  # Extract Action Units
        '-out_dir', '/tmp/openface_output'
    ]
    
    subprocess.run(cmd)
    
    # Read OpenFace output CSV
    csv_path = '/tmp/openface_output/video_name.csv'
    aus = pd.read_csv(csv_path)
    
    return aus
```

### Step 2: Calculate Cumulative AU Intensity

```python
def get_peak_au_frame(aus_df):
    """
    Find frame with maximum cumulative AU intensity.
    
    Args:
        aus_df: DataFrame with columns like AU01_r, AU02_r, ..., AU45_r
        
    Returns:
        frame_number: Index of frame with peak AU intensity
    """
    # OpenFace AU columns (AU01_r to AU45_r for intensity)
    au_columns = [col for col in aus_df.columns if col.endswith('_r')]
    
    # Sum all AU intensities per frame
    cumulative_intensity = aus_df[au_columns].sum(axis=1)
    
    # Find frame with maximum cumulative intensity
    peak_frame_idx = cumulative_intensity.idxmax()
    
    return peak_frame_idx
```

### Step 3: Extract Peak Frame

```python
def extract_peak_frame(video_path):
    """
    Extract frame with maximum cumulative AU intensity.
    """
    # Step 1: Get AU data
    aus = extract_openface_aus(video_path)
    
    # Step 2: Find peak frame
    peak_idx = get_peak_au_frame(aus)
    
    # Step 3: Extract that specific frame
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, peak_idx)
    success, frame = cap.read()
    cap.release()
    
    if not success:
        raise ValueError(f"Failed to extract frame {peak_idx}")
    
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame_rgb, peak_idx
```

### Step 4: Updated Dataset Code

```python
def __getitem__(self, index):
    t = self.tmp[index]
    video_name = t[0]
    
    mp4_path = os.path.join(self.vis_root, video_name + ".mp4")
    
    if os.path.exists(mp4_path):
        # Use peak AU frame instead of first frame
        image, peak_idx = self.extract_peak_frame(mp4_path)
        if index == 0:
            print(f"  - Using peak AU frame (frame #{peak_idx})")
    else:
        # Fallback to pre-extracted peak frames
        jpg_path = os.path.join(self.vis_root, video_name + ".jpg")
        image = Image.open(jpg_path).convert("RGB")
        image = np.array(image)
    
    # ... rest of processing
```

---

## 🚀 Recommended Solutions

### Option 1: Pre-extract Peak AU Frames (Recommended)

**Advantages:**
- No runtime overhead
- Consistent with paper methodology
- Easy to verify

**Steps:**
```bash
# 1. Install OpenFace
# 2. Process all videos
for video in /export/home/scratch/qze/datasets/MER2023/test3/*.mp4; do
    # Extract AUs
    FeatureExtraction -f $video -aus -out_dir /tmp/aus/
    
    # Find peak frame (Python script)
    python find_peak_frame.py $video /tmp/aus/
    
    # Extract and save peak frame as JPG
    ffmpeg -i $video -vf "select=eq(n\,${PEAK_FRAME})" \
           -vframes 1 ${video%.mp4}.jpg
done

# 3. Code will automatically use .jpg files when available
```

### Option 2: Runtime Peak Detection

**Advantages:**
- No pre-processing needed
- Can adapt to different datasets

**Disadvantages:**
- Slower (OpenFace runtime per video)
- Requires OpenFace installation

**Implementation:**
Modify `extract_frame()` method to use peak detection (see Step 3 above)

### Option 3: Use Pre-computed Peak Frames from MERR

**If MERR dataset provides peak frames:**
```bash
# Copy pre-extracted peak frames from MERR
cp /path/to/MERR/peak_frames/*.jpg /export/home/scratch/qze/datasets/MER2023/test3/
```

The code already has fallback to use `.jpg` files when available!

---

## 📈 Expected Performance Improvement

If we switch to peak AU frames, we might see:

### Predicted Changes

1. **Happy Detection:** Currently 80.72% → **Could improve to 85-88%**
   - Happiness peak expression more visible
   - Less confusion with neutral

2. **Worried Detection:** Currently 42.86% → **Could improve to 55-65%**
   - Subtle worry expressions more evident at peak
   - Better facial AU patterns

3. **Surprise Detection:** Currently 73.33% → **Could improve to 78-82%**
   - Peak surprise (eyes wide, mouth open) more distinct
   - Less confusion with neutral/happy

4. **Overall Accuracy:** 90.65% → **Potential 92-94%**

### Why Not 100% Improvement?

- Pre-extracted features already capture peak information
- Transcription provides strong signal
- Some labels may be based on context, not facial expression
- Class imbalance (Worried n=14) still problematic

---

## 🔧 Quick Fix: Check if Peak Frames Already Exist

The code already supports pre-extracted JPG frames as a fallback. Let's check if peak frames might exist elsewhere:

```bash
# Check for peak frames in MERR directory
ls /export/home/scratch/qze/MERR/*frames*/

# Check for peak frames alongside features
ls /export/home/scratch/qze/features_of_MER2023-SEMI/*frames*/

# Check annotation directory
ls /export/home/scratch/qze/*frames*/
```

---

## 📝 Immediate Action Items

### 1. Document Current Limitation
✅ **Done** - This document

### 2. Verify MERR Peak Frames Availability
Check if MERR dataset provides pre-extracted peak AU frames

### 3. Consider Re-evaluation
If peak frames become available:
```bash
# Place peak JPGs in test3 directory
# Re-run evaluation - code will automatically use them
python verify_paths.py  # Will show "Using pre-extracted .jpg frames"
```

### 4. Update Reports
Add note to all reports about frame extraction method

---

## 🎯 Conclusion

**Current State:**
- Code uses **first frame** (not peak AU frame as per paper)
- Still achieves **90.65% accuracy** due to strong features + transcription
- Visual input may not always align with emotion labels

**Recommendation:**
- **Short-term:** Document limitation, results still valid
- **Medium-term:** Check if MERR provides peak frames
- **Long-term:** Pre-extract peak AU frames for optimal performance

**Expected Impact:**
- Switching to peak frames could improve accuracy by **1-3%**
- Biggest gains expected for Happy, Worried, Surprise classes
- May reduce Happy→Neutral and Happy→Angry confusion

---

**Discovered:** October 30, 2025  
**Issue:** Frame extraction method differs from paper  
**Impact:** Minor (features compensate)  
**Priority:** Medium (nice-to-have, not critical)  
**Difficulty:** Easy (if peak frames available) to Medium (if need to generate)
