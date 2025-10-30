# Emotion-LLaMA Evaluation Run Report

**Date:** October 30, 2025  
**Evaluation Type:** Emotion Recognition on MER2023 Test3 Dataset  
**Command Executed:**
```tcsh
source .venv/bin/activate.csh
.venv/bin/torchrun --nproc_per_node 1 eval_emotion.py --cfg-path eval_configs/eval_emotion.yaml --dataset feature_face_caption
```

---

## Executive Summary

This evaluation run tests the **Emotion-LLaMA** multimodal emotion recognition model on the MER2023 test3 dataset. The model combines visual (video frames, facial expressions), audio, and textual (transcription) features to classify emotions into 6 categories: neutral, angry, happy, sad, worried, and surprise.

### **Results Summary**
- **Dataset Size:** 834 videos
- **Overall Accuracy:** 90.65%
- **Weighted Precision:** 90.44%
- **Weighted Recall:** 90.65%
- **Weighted F1 Score:** 90.45%
- **Status:** ✅ Successfully Completed

### **Data Sources Confirmed** (See `path_verification_results.md` for details)
- **Videos:** ✅ `/export/home/scratch/qze/datasets/MER2023/test3` (corrected path - .mp4 files)
- **Features:** ⚠️ `/export/home/scratch/qze/features_of_MER2023-SEMI` (fallback, but all 834 samples present)
- **Transcriptions:** ✅ `/export/home/scratch/qze/transcription_en_all.csv` (sentence_en column)
- **Annotations:** ✅ `/export/home/scratch/qze/relative_test3_NCEV.txt`

---

## 1. Model Architecture

### Base Model Configuration
- **Architecture:** MiniGPT-v2
- **Language Model:** Llama-2-7b-chat-hf
- **Model Type:** Pretrained with fine-tuning
- **Location:** `/local/user/qze/Code/Emotion-LLaMA/checkpoints/Llama-2-7b-chat-hf`

### Fine-tuned Checkpoint
- **Path:** `/export/home/scratch/qze/checkpoint/stage2/checkpoint_best.pth`
- **Stage:** Stage 2 (emotion-specific fine-tuning)
- **LoRA Configuration:**
  - `lora_r`: 64 (LoRA rank)
  - `lora_alpha`: 16 (LoRA scaling parameter)
  - Low-resource mode: False

### Model Parameters
- **Max text length:** 500 tokens
- **End symbol:** `</s>`
- **Prompt template:** `[INST] {} [/INST]` (Llama-2 instruction format)
- **Max new tokens (generation):** 500
- **Sampling:** Greedy decoding (do_sample=False)

---

## 2. Dataset Configuration

### Dataset Used: `feature_face_caption`

#### Video Data
- **Dataset Path:** `/export/home/scratch/qze/datasets/MER2023/test3`
- **Video Format:** `.mp4` files (with `.avi` and `.jpg` fallbacks)
- **Purpose:** Contains raw video files for MER2023 test3 split
- **Frame Extraction:** First frame extracted from each video using OpenCV

#### Annotation File
- **Path:** `/export/home/scratch/qze/relative_test3_NCEV.txt`
- **Format:** Space-separated text file
  ```
  <video_name> <label_id> <emotion_label>
  ```
- **Example:**
  ```
  samplenew_00006611 0 neutral
  samplenew_00006612 1 angry
  ```
- **Emotion Categories (6 classes):**
  1. neutral
  2. angry
  3. happy
  4. sad
  5. worried
  6. surprise

**Note:** The code also supports 9 classes (adds: fear, contempt, doubt) but this evaluation uses 6 classes.

---

## 3. Feature Files

The model uses **three types of pre-extracted features** for each video:

### 3.1 FaceMAE Features
- **Dimension:** 340-dimensional
- **Purpose:** Facial expression features
- **Extractor:** FaceMAE model
- **Location Priority:**
  1. `<annotation_dir>/mae_340_23_UTT/<video_name>.npy` (MER2024 style)
  2. `/export/home/scratch/qze/features_of_MER2023-SEMI/mae_340_UTT_MER2023-SEMI/<video_name>.npy` (MER2023-SEMI fallback)

### 3.2 VideoMAE Features
- **Dimension:** 399-dimensional
- **Purpose:** Video content and motion features
- **Extractor:** VideoMAE model
- **Location Priority:**
  1. `<annotation_dir>/maeVideo_399_23_UTT/<video_name>.npy` (MER2024 style)
  2. `/export/home/scratch/qze/features_of_MER2023-SEMI/maeV_399_UTT_MER2023-SEMI/<video_name>.npy` (MER2023-SEMI fallback)

### 3.3 Audio Features
- **Dimension:** Variable (HuBERT-Large features)
- **Purpose:** Acoustic and paralinguistic features
- **Extractor:** HuBERT-Large model
- **Location Priority:**
  1. `<annotation_dir>/HL_23_UTT/<video_name>.npy` (MER2024 style)
  2. `/export/home/scratch/qze/features_of_MER2023-SEMI/HL-UTT_MER2023-SEMI/<video_name>.npy` (MER2023-SEMI fallback)

### Feature Concatenation
All three feature types are concatenated along the temporal dimension:
```python
video_features = torch.cat((FaceMAE_feats, VideoMAE_feats, Audio_feats), dim=0)
```

---

## 4. Transcription Data

### Transcription CSV
- **Purpose:** Provides textual transcription of spoken content in videos
- **Search Priority:**
  1. Environment variable: `$EMOTION_LLAMA_TRANSCRIPT_CSV`
  2. `<annotation_dir>/transcription_en_all.csv`
  3. `<annotation_dir>/transcription_all_new.csv`
  4. `/export/home/scratch/qze/transcription_en_all.csv`
  5. `/export/home/scratch/qze/transcription_all_new.csv`

- **Required Columns:**
  - `name`: Video identifier
  - `sentence` or `sentence_en`: English transcription text

### Usage in Prompt
The transcription is incorporated into the model input:
```python
character_line = "The person in video says: {}. ".format(sentence)
```

---

## 5. Additional Reference Data

### MERR (Multimodal Emotion Reason Recognition) Datasets

#### 5.1 Coarse-Grained Annotations
- **Path:** `/export/home/scratch/qze/MERR/MERR_coarse_grained.json`
- **Purpose:** Emotion reasoning captions (coarse level)
- **Format:** JSON dictionary
  ```json
  {
    "video_name": {
      "caption": "The person appears happy because..."
    }
  }
  ```
- **Used for:** "reason" task type in training/evaluation

#### 5.2 Fine-Grained Annotations
- **Path:** `/export/home/scratch/qze/MERR/MERR_fine_grained.json`
- **Purpose:** Detailed emotion reasoning captions
- **Format:** JSON dictionary with `smp_reason_caption` field
- **Used for:** "reason_v2" task type in training/evaluation

**Note:** In this evaluation run, only the **"emotion"** task was used (emotion classification), not reasoning tasks.

---

## 6. Data Processing Pipeline

### 6.1 Visual Processing
1. **Frame Extraction:** First frame extracted from video file
2. **Image Conversion:** Convert to RGB PIL Image
3. **BLIP-2 Preprocessing:**
   - Processor: `blip2_image_train`
   - Image size: 448×448
   - Normalization and augmentation

### 6.2 Text Processing
- **Processor:** `blip_caption`
- **Purpose:** Tokenization and formatting of emotion labels

### 6.3 Instruction Format
The model receives multimodal instructions in this format:
```
<video><VideoHere></video> <feature><FeatureHere></feature> The person in video says: {transcription}. [emotion] {instruction}
```

**Example instruction templates (random selection):**
- "Please determine which emotion label in the video represents: happy, sad, neutral, angry, worried, surprise, fear, contempt, doubt."

---

## 7. Evaluation Process

### 7.1 Data Loading
- **Batch size:** 1 (single video at a time)
- **Shuffle:** False (sequential evaluation)
- **DataLoader:** PyTorch DataLoader with FeatureFaceDataset

### 7.2 Inference
- **Device:** Distributed training setup (torchrun with 1 GPU)
- **Generation:**
  - Max new tokens: 500
  - Sampling: Greedy (do_sample=False)
  - Temperature: Default
  
### 7.3 Post-processing
```python
# Extract last word as emotion label
answer = answer.split(" ")[-1]
target = target.split(" ")[-1]

# Validate against allowed emotions
if answer not in ['neutral', 'angry', 'happy', 'sad', 'worried', 'surprise']:
    print("Error: ", answer, " Target:", target)
    answer = 'neutral'  # Default to neutral for invalid predictions
```

### 7.4 Metrics Computed
- **Accuracy:** Overall classification accuracy
- **Precision:** Weighted precision across all classes
- **Recall:** Weighted recall across all classes
- **F1 Score:** Weighted F1 score across all classes
- **Confusion Matrix:** 6×6 matrix showing prediction patterns

---

## 8. Output and Results

### Results Directory
- **Path:** `/export/home/scratch/qze/results`
- **Configured in:** `eval_configs/eval_emotion.yaml` under `run.save_path`

### Console Output
The evaluation prints:
1. Configuration summary
2. Dataset paths and parameters
3. Per-sample predictions (if errors occur)
4. Final metrics:
   - Accuracy
   - Precision
   - Recall
   - F1 Score
   - Confusion Matrix

---

## 9. Task Types Available (Not Used in This Run)

The FeatureFaceDataset supports multiple task types (configured in `task_pool`):

### 9.1 Emotion Task (USED)
- **Pool:** `["emotion"]`
- **Output:** Emotion label only
- **Instructions:** From `emotion_instruction_pool`

### 9.2 Reason Task (AVAILABLE)
- **Caption source:** MERR_coarse_grained.json
- **Output:** Reasoning explanation with emotion
- **Instructions:** From `reason_instruction_pool`

### 9.3 Reason_v2 Task (AVAILABLE)
- **Caption source:** MERR_fine_grained.json
- **Output:** Detailed reasoning with emotion
- **Instructions:** From `reason_instruction_pool`

**Current Configuration:** Only "emotion" task is active in this evaluation.

---

## 10. Key Configuration Decisions

### 10.1 Path Correction
**Issue:** Original config pointed to `/export/home/scratch/qze/emotion-llama/MER2023+MER2024`  
**Problem:** Video file `samplenew_00006611.mp4` not found  
**Solution:** Updated to `/export/home/scratch/qze/datasets/MER2023/test3` (actual video location)

### 10.2 Feature Resolution Strategy
The code implements a smart fallback mechanism:
1. First checks for features next to annotation file (MER2024 structure)
2. Falls back to MER2023-SEMI hardcoded paths if not found

This allows the same code to work with different dataset organizations.

### 10.3 Transcription CSV Resolution
Similar fallback strategy for finding transcription files, with environment variable override capability for flexibility.

---

## 11. System Requirements

### Environment
- **Shell:** tcsh
- **Python Environment:** Virtual environment at `.venv/`
- **Activation:** `source .venv/bin/activate.csh`

### Key Dependencies
- PyTorch with distributed support (torchrun)
- Transformers (Llama-2 model)
- OpenCV (video processing)
- Scikit-learn (metrics)
- PIL/Pillow (image processing)
- NumPy, Pandas

### Hardware
- **GPUs:** 1 GPU (configured via `--nproc_per_node 1`)
- **Memory:** Sufficient for Llama-2-7b model + features

---

## 12. Troubleshooting Notes

### Common Issues Encountered

1. **Virtual Environment Not Activated**
   - **Symptom:** `torchrun` command not found or wrong Python version
   - **Solution:** Always run `source .venv/bin/activate.csh` first in tcsh

2. **Command Line Backslashes**
   - **Symptom:** "unrecognized arguments" error
   - **Issue:** Multi-line backslash continuation in tcsh interprets spaces as arguments
   - **Solution:** Use single-line command or proper tcsh continuation

3. **Video File Not Found**
   - **Symptom:** `FileNotFoundError` for video files
   - **Solution:** Verify `img_path` points to actual video directory
   - **Fallback:** Code tries `.mp4`, `.avi`, and `.jpg` extensions

4. **Missing Transcription**
   - **Symptom:** `FileNotFoundError` for transcription CSV
   - **Solution:** Set `EMOTION_LLAMA_TRANSCRIPT_CSV` environment variable or place file in annotation directory

5. **Missing Features**
   - **Symptom:** NumPy file not found errors
   - **Solution:** Ensure feature files exist in expected locations (see Section 3)

---

## 13. File Structure Summary

```
Emotion-LLaMA/
├── eval_emotion.py                    # Main evaluation script
├── eval_configs/
│   └── eval_emotion.yaml             # Configuration file (MODIFIED)
├── minigpt4/
│   ├── datasets/datasets/
│   │   └── first_face.py             # FeatureFaceDataset implementation
│   └── models/
│       └── minigpt_v2.py             # MiniGPT-v2 model
├── checkpoints/
│   └── Llama-2-7b-chat-hf/          # Base LLM
└── .venv/                            # Python virtual environment

External Directories:
├── /export/home/scratch/qze/
│   ├── checkpoint/stage2/            # Fine-tuned model weights
│   ├── datasets/MER2023/test3/       # Video files (CORRECTED PATH)
│   ├── relative_test3_NCEV.txt       # Annotation file
│   ├── features_of_MER2023-SEMI/     # Pre-extracted features
│   ├── MERR/                         # Reasoning annotations
│   ├── transcription_en_all.csv      # Video transcriptions
│   └── results/                      # Output directory
```

---

## 14. Evaluation Results

### Model Loading
- **Checkpoint Loading:** Successfully loaded in 2 shards (18 seconds total, ~9s per shard)
- **LoRA Configuration Applied:**
  - Target modules: `['q_proj', 'v_proj']`
  - Trainable parameters: 33,554,432 (0.495% of total 6,771,970,048 parameters)
  - Dropout: 0.05
  - Bias: None
- **Position Interpolation:** 16×16 → 32×32
- **Checkpoint Status:** ✅ Loaded from `/export/home/scratch/qze/checkpoint/stage2/checkpoint_best.pth`

### Dataset Statistics
- **Total Videos:** 834 samples
- **Annotation File:** `/export/home/scratch/qze/relative_test3_NCEV.txt`
- **Video Directory:** `/export/home/scratch/qze/datasets/MER2023/test3`
- **Batch Size:** 1 (sequential processing)
- **Max New Tokens:** 500

### Performance Metrics

#### Overall Performance
```
Accuracy:  90.65%
Precision: 90.44% (weighted)
Recall:    90.65% (weighted)
F1 Score:  90.45% (weighted)
```

#### Confusion Matrix
```
                Predicted
              Neu  Ang  Hap  Sad  Wor  Sur
Actual    ┌─────────────────────────────────
  Neutral │ 179    1    2    1    0    0     (183 samples)
  Angry   │   2  154    7    3    2    1     (169 samples)
  Happy   │  10   10  134    6    2    4     (166 samples)
  Sad     │   2    2    1  250    0    2     (257 samples)
  Worried │   1    0    3    1    6    3     ( 14 samples)
  Surprise│   5    1    3    2    1   33     ( 45 samples)
```

**Legend:**
- Neu = Neutral
- Ang = Angry
- Hap = Happy
- Sad = Sad
- Wor = Worried
- Sur = Surprise

### Per-Class Performance Analysis

#### Class Distribution and Accuracy

1. **Sad (257 samples - 30.8%)**
   - Correctly classified: 250/257
   - Class accuracy: **97.28%** ⭐ Best performing class
   - Main confusions: 2→Neutral, 2→Angry, 2→Surprise

2. **Neutral (183 samples - 21.9%)**
   - Correctly classified: 179/183
   - Class accuracy: **97.81%** ⭐ Second best
   - Main confusions: 2→Happy, 1→Angry, 1→Sad

3. **Angry (169 samples - 20.3%)**
   - Correctly classified: 154/169
   - Class accuracy: **91.12%**
   - Main confusions: 7→Happy, 3→Sad, 2→Neutral, 2→Worried

4. **Happy (166 samples - 19.9%)**
   - Correctly classified: 134/166
   - Class accuracy: **80.72%**
   - Main confusions: 10→Neutral, 10→Angry, 6→Sad, 4→Surprise
   - Notable: Most confusion with Neutral and Angry

5. **Surprise (45 samples - 5.4%)**
   - Correctly classified: 33/45
   - Class accuracy: **73.33%**
   - Main confusions: 5→Neutral, 3→Happy, 2→Sad
   - Challenge: Small sample size

6. **Worried (14 samples - 1.7%)** ⚠️
   - Correctly classified: 6/14
   - Class accuracy: **42.86%** (poorest performance)
   - Main confusions: 3→Happy, 3→Surprise, 1→Neutral, 1→Sad
   - Challenge: Very small sample size and high confusion

### Key Observations

#### Strengths
✅ **Excellent on "Sad" emotion:** 97.28% accuracy (dominant class)  
✅ **Strong on "Neutral":** 97.81% accuracy  
✅ **Robust overall performance:** 90.65% accuracy across all classes  
✅ **Good on "Angry":** 91.12% accuracy

#### Challenges
⚠️ **"Worried" emotion:** Only 42.86% accuracy (14 samples)  
⚠️ **"Surprise" emotion:** 73.33% accuracy (45 samples)  
⚠️ **Class imbalance:** Worried (1.7%) vs Sad (30.8%)  
⚠️ **"Happy" confusions:** Significant overlap with Neutral (10) and Angry (10)

#### Error Analysis

**Most Common Misclassifications:**
1. Happy → Neutral (10 cases): Subtle expressions may appear neutral
2. Happy → Angry (10 cases): Possible intensity confusion
3. Angry → Happy (7 cases): Could be sarcastic or mixed emotions
4. Sad → Neutral (2 cases): Minimal confusion on dominant class

**Prediction Error:**
- **1 instance** predicted "doubt" (not in 6-class taxonomy) for a "surprise" sample
- System correctly defaulted to "neutral" for invalid prediction
- Suggests model occasionally outputs 9-class labels despite 6-class training

### Runtime Information

**Warnings Encountered:**
1. ⚠️ Transformers deprecation warning: `_register_pytree_node` (cosmetic, no impact)
2. ⚠️ Generation config warning: `top_p=0.9` set but `do_sample=False` (no impact on greedy decoding)

**Processing:**
- All 834 videos processed successfully
- No failures or missing files reported
- Smooth end-to-end evaluation

---

## 15. Conclusion and Recommendations

### Summary

This evaluation run assesses the Emotion-LLaMA model's ability to classify emotions in multimodal video data by:
- Processing video frames (visual features)
- Analyzing facial expressions (FaceMAE)
- Understanding video content (VideoMAE)
- Processing audio signals (HuBERT)
- Incorporating spoken content (transcriptions)
- Generating emotion predictions using Llama-2-7b

The model uses a sophisticated multimodal fusion approach within the MiniGPT-v2 architecture to classify 6 emotion categories on the MER2023 test3 dataset.

**Final Performance:** 90.65% accuracy demonstrates strong emotion recognition capabilities, particularly excelling on Sad and Neutral emotions while facing challenges with underrepresented classes (Worried, Surprise).

### Recommendations

1. **Class Imbalance:** Consider data augmentation or class weighting for "Worried" (14 samples) and "Surprise" (45 samples)

2. **Happy/Neutral Confusion:** Investigate multimodal cues that distinguish subtle happiness from neutral states

3. **Model Output:** Review why model occasionally outputs 9-class labels ("doubt") despite 6-class fine-tuning

4. **Generation Config:** Clean up warning by either:
   - Setting `do_sample=True` with `top_p=0.9` for diverse outputs
   - Removing `top_p` parameter for pure greedy decoding

5. **Feature Analysis:** Given strong Sad/Neutral performance, analyze which features (Face/Video/Audio) contribute most to successful predictions

---

**Report Generated:** October 30, 2025  
**Evaluation Status:** ✅ Completed Successfully  
**Total Videos Processed:** 834  
**Overall Accuracy:** 90.65%  
**Best Performing Classes:** Neutral (97.81%), Sad (97.28%)  
**Challenging Classes:** Worried (42.86%), Surprise (73.33%)  
**Next Steps:** Analyze per-class errors and investigate class imbalance solutions
