# File Paths Reference Guide

## Quick Reference for All Files Used in Evaluation

---

## Model Files

### Base Language Model
```
/local/user/qze/Code/Emotion-LLaMA/checkpoints/Llama-2-7b-chat-hf
```
- **Type:** Pre-trained Llama-2 7B Chat model
- **Purpose:** Base language model for instruction following
- **Size:** ~13GB

### Fine-tuned Checkpoint
```
/export/home/scratch/qze/checkpoint/stage2/checkpoint_best.pth
```
- **Type:** Emotion-specific fine-tuned weights
- **Purpose:** Specialized emotion recognition capabilities
- **Training Stage:** Stage 2 (instruction tuning)

---

## Dataset Files

### Video Files
```
/export/home/scratch/qze/datasets/MER2023/test3/
```
- **Contents:** Raw video files (.mp4, .avi)
- **Example:** `samplenew_00006611.mp4`
- **Purpose:** Source videos for emotion recognition
- **Dataset:** MER2023 test3 split

### Annotation File
```
/export/home/scratch/qze/relative_test3_NCEV.txt
```
- **Format:** Space-separated values
- **Structure:** `<video_name> <label_id> <emotion_label>`
- **Purpose:** Ground truth labels for evaluation
- **Emotion Classes:** neutral, angry, happy, sad, worried, surprise

### Transcription Files (CSV)
**Primary Locations (searched in order):**
1. `$EMOTION_LLAMA_TRANSCRIPT_CSV` (environment variable)
2. `<annotation_dir>/transcription_en_all.csv`
3. `<annotation_dir>/transcription_all_new.csv`
4. `/export/home/scratch/qze/transcription_en_all.csv`
5. `/export/home/scratch/qze/transcription_all_new.csv`

**Required Columns:**
- `name`: Video identifier matching annotation file
- `sentence` or `sentence_en`: English transcription text

---

## Feature Files

### Directory Structure (MER2024 Style - Preferred)
Located relative to annotation file directory:
```
<annotation_dir>/
├── mae_340_23_UTT/
│   └── <video_name>.npy          # FaceMAE features (340-dim)
├── maeVideo_399_23_UTT/
│   └── <video_name>.npy          # VideoMAE features (399-dim)
└── HL_23_UTT/
    └── <video_name>.npy          # HuBERT-Large audio features
```

### Directory Structure (MER2023-SEMI Style - Fallback)
```
/export/home/scratch/qze/features_of_MER2023-SEMI/
├── mae_340_UTT_MER2023-SEMI/
│   └── <video_name>.npy          # FaceMAE features (340-dim)
├── maeV_399_UTT_MER2023-SEMI/
│   └── <video_name>.npy          # VideoMAE features (399-dim)
└── HL-UTT_MER2023-SEMI/
    └── <video_name>.npy          # HuBERT-Large audio features
```

**Feature Details:**
- **FaceMAE:** Facial expression embeddings (340 dimensions)
- **VideoMAE:** Video content embeddings (399 dimensions)
- **Audio:** HuBERT-Large acoustic features (variable dimensions)
- **Format:** NumPy arrays (.npy files)
- **Loading:** `torch.tensor(np.load(path))`

---

## Reasoning Annotation Files (Not Used in Current Run)

### Coarse-Grained Reasoning
```
/export/home/scratch/qze/MERR/MERR_coarse_grained.json
```
- **Format:** JSON dictionary
- **Structure:**
  ```json
  {
    "video_name": {
      "caption": "Reasoning description..."
    }
  }
  ```
- **Purpose:** Emotion reasoning explanations (coarse level)

### Fine-Grained Reasoning
```
/export/home/scratch/qze/MERR/MERR_fine_grained.json
```
- **Format:** JSON dictionary
- **Structure:**
  ```json
  {
    "video_name": {
      "smp_reason_caption": "Detailed reasoning..."
    }
  }
  ```
- **Purpose:** Detailed emotion reasoning with multimodal cues

---

## Configuration Files

### Evaluation Config
```
/local/user/qze/Code/Emotion-LLaMA/eval_configs/eval_emotion.yaml
```
- **Purpose:** Evaluation parameters and paths
- **Key Sections:**
  - `model`: Model architecture and checkpoint paths
  - `datasets`: Dataset processing configuration
  - `evaluation_datasets`: Eval-specific paths and parameters
  - `run`: Output and task configuration

### Dataset Config
```
/local/user/qze/Code/Emotion-LLaMA/minigpt4/configs/datasets/firstface/featureface.yaml
```
- **Purpose:** Training dataset configuration
- **Note:** Separate from evaluation config

### Model Config
```
/local/user/qze/Code/Emotion-LLaMA/minigpt4/configs/models/minigpt_v2.yaml
```
- **Purpose:** MiniGPT-v2 architecture configuration

---

## Output Files

### Results Directory
```
/export/home/scratch/qze/results/
```
- **Purpose:** Evaluation results and logs
- **Contents:** Model predictions, metrics, analysis outputs

---

## Code Files

### Main Evaluation Script
```
/local/user/qze/Code/Emotion-LLaMA/eval_emotion.py
```
- **Purpose:** Main evaluation logic
- **Functions:**
  - Dataset loading
  - Model inference
  - Metric computation
  - Result saving

### Dataset Implementation
```
/local/user/qze/Code/Emotion-LLaMA/minigpt4/datasets/datasets/first_face.py
```
- **Class:** `FeatureFaceDataset`
- **Purpose:** Data loading and preprocessing
- **Key Methods:**
  - `__getitem__`: Load video, features, transcription
  - `extract_frame`: Extract first frame from video
  - `get`: Load pre-extracted features

---

## Environment Files

### Virtual Environment
```
/local/user/qze/Code/Emotion-LLaMA/.venv/
```
- **Activation (tcsh):** `source .venv/bin/activate.csh`
- **Python:** 3.11
- **Key Tools:** torchrun, transformers, opencv

### Requirements
```
/local/user/qze/Code/Emotion-LLaMA/requirements.txt
/local/user/qze/Code/Emotion-LLaMA/environment.yml
```
- **Purpose:** Python package dependencies
- **Installation:** `pip install -r requirements.txt`

---

## Path Resolution Logic

### For Each Video Sample:

1. **Video File** (first match wins):
   - `{img_path}/{video_name}.mp4`
   - `{img_path}/{video_name}.avi`
   - `{img_path}/{video_name}.jpg` (pre-extracted frame)

2. **FaceMAE Features** (first match wins):
   - `{ann_dir}/mae_340_23_UTT/{video_name}.npy`
   - `/export/home/scratch/qze/features_of_MER2023-SEMI/mae_340_UTT_MER2023-SEMI/{video_name}.npy`

3. **VideoMAE Features** (first match wins):
   - `{ann_dir}/maeVideo_399_23_UTT/{video_name}.npy`
   - `/export/home/scratch/qze/features_of_MER2023-SEMI/maeV_399_UTT_MER2023-SEMI/{video_name}.npy`

4. **Audio Features** (first match wins):
   - `{ann_dir}/HL_23_UTT/{video_name}.npy`
   - `/export/home/scratch/qze/features_of_MER2023-SEMI/HL-UTT_MER2023-SEMI/{video_name}.npy`

5. **Transcription**:
   - Lookup `video_name` in CSV file (found via search priority)
   - Extract from `sentence` or `sentence_en` column

Where:
- `{img_path}` = `/export/home/scratch/qze/datasets/MER2023/test3`
- `{ann_dir}` = `/export/home/scratch/qze` (directory of annotation file)
- `{video_name}` = e.g., `samplenew_00006611`

---

## File Format Specifications

### Annotation TXT Format
```
samplenew_00006611 0 neutral
samplenew_00006612 1 angry
samplenew_00006613 2 happy
```
- Column 1: Video identifier (no extension)
- Column 2: Emotion ID (0-5)
- Column 3: Emotion label (string)

### Transcription CSV Format
```csv
name,sentence_en,sentence
samplenew_00006611,"Hello, how are you?","你好，你好吗？"
```
- `name`: Video identifier
- `sentence_en`: English transcription (preferred)
- `sentence`: Original language transcription (fallback)

### Feature NPY Format
- **Type:** NumPy array saved with `np.save()`
- **Loading:** `np.load(path)`
- **Shape:** 
  - FaceMAE: (340,) or (T, 340)
  - VideoMAE: (399,) or (T, 399)
  - Audio: (D,) or (T, D) where D varies

---

## Verification Checklist

Before running evaluation, verify these files exist:

- [ ] Base model: `/local/user/qze/Code/Emotion-LLaMA/checkpoints/Llama-2-7b-chat-hf`
- [ ] Checkpoint: `/export/home/scratch/qze/checkpoint/stage2/checkpoint_best.pth`
- [ ] Video directory: `/export/home/scratch/qze/datasets/MER2023/test3/`
- [ ] Annotation: `/export/home/scratch/qze/relative_test3_NCEV.txt`
- [ ] Transcription CSV: One of the search locations
- [ ] Features: Either MER2024-style or MER2023-SEMI directories
- [ ] Config: `/local/user/qze/Code/Emotion-LLaMA/eval_configs/eval_emotion.yaml`
- [ ] Output dir: `/export/home/scratch/qze/results/` (created if not exists)

---

**Last Updated:** October 30, 2025  
**Dataset:** MER2023 test3  
**Evaluation Type:** Emotion Classification (6 classes)
