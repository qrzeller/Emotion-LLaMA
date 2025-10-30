# Analysis Tools - Quick Reference

**Location:** `/local/user/qze/Code/Emotion-LLaMA/scripts/`

---

## 🛠️ Available Tools

### 1. **verify_paths.py** - Path Verification
**Purpose:** Check which files will be used (before running evaluation)

**Usage:**
```bash
python scripts/verify_paths.py
```

**Output:**
- ✅ Model checkpoint locations
- ✅ Dataset paths (videos, annotations)
- ✅ Feature directories (MER2024-style vs MER2023-SEMI)
- ✅ Transcription CSV paths
- ✅ Verification of first sample completeness

**When to use:** Before running evaluation to ensure correct paths

---

### 2. **inspect_sample.py** - Sample Inspector
**Purpose:** Deep dive into individual samples (human-readable format)

**Usage:**
```bash
# Inspect first sample
python scripts/inspect_sample.py

# Inspect specific sample by index
python scripts/inspect_sample.py --sample_id 0
python scripts/inspect_sample.py --sample_id 100

# Inspect specific sample by video name
python scripts/inspect_sample.py --sample_id samplenew_00006611
python scripts/inspect_sample.py --sample_id sample_00002328
```

**Output:**
- 📹 Video properties (resolution, FPS, duration)
- 🎯 Ground truth emotion
- 💬 Complete transcription text
- 📊 Feature statistics (3 modalities: Face, Video, Audio)
- 📝 Actual prompt sent to model
- 💾 First frame saved to /tmp/ for viewing

**When to use:** 
- Understand what model sees for specific examples
- Debug misclassifications
- Analyze feature patterns

---

## 📊 Example Workflow

### Before Evaluation
```bash
# 1. Verify all paths are correct
python verify_paths.py

# Output shows:
# - Which video directory will be used
# - Which feature source (MER2024 vs MER2023-SEMI)
# - Which transcription CSV
# - Sample verification for first video
```

### During Analysis
```bash
# 2. Inspect interesting samples
python inspect_sample.py --sample_id 0      # First sample
python inspect_sample.py --sample_id 45     # A "surprise" sample
python inspect_sample.py --sample_id 500    # Mid-dataset sample

# 3. View extracted frames
display /tmp/samplenew_00006611_first_frame.jpg
```

### Error Investigation
```bash
# If model made errors, find them in confusion matrix
# Example: Happy predicted as Neutral (10 cases)

# Then inspect those specific samples:
python inspect_sample.py --sample_id <happy_sample_id>

# Check:
# - Does first frame look neutral?
# - What does transcription say?
# - Are features unusual?
```

---

## 📁 Report Documents

All reports are in `report/` directory:

### 1. **evaluation_run_report.md**
Complete evaluation documentation with:
- Model architecture
- Dataset configuration  
- Feature file details
- Full results (90.65% accuracy)
- Confusion matrix analysis
- Per-class performance
- Recommendations

### 2. **results_summary.md**
Concise results overview:
- Overall metrics table
- Per-class performance
- Confusion matrix
- Error analysis
- Key findings

### 3. **path_verification_results.md**
Proof of which files were used:
- Video paths (✅ corrected)
- Feature paths (⚠️ MER2023-SEMI fallback)
- Transcription sources
- Complete data flow diagram
- Validation that fallback is expected

### 4. **sample_inspection_report.md** (NEW!)
Human-readable sample analysis:
- **KEY FINDING:** Only first frame used from videos
- Detailed walkthrough of 2 samples
- Feature dimension explanations
- Prompt construction details
- Model processing pipeline
- Critical insights about frame/video mismatch

### 5. **file_paths_reference.md**
Quick path reference:
- All file locations organized by category
- Path resolution logic
- File format specifications
- Verification checklist

---

## 🔑 Key Discoveries

### From Path Verification
✅ **Videos:** Using correct MER2023/test3 directory (after fix)  
⚠️ **Features:** Using MER2023-SEMI fallback (expected, all 834 samples present)  
✅ **Transcriptions:** Using transcription_en_all.csv  
✅ **All 834 samples have matching files**

### From Sample Inspection
⚠️ **CRITICAL:** Model only uses **first frame** from videos (not full video)  
✅ **Features:** Pre-extracted from full videos (Face + Video + Audio = 3072-dim)  
✅ **Transcriptions:** Provide crucial contextual information  
✅ **Prompt:** Structured instruction format with special tokens  

---

## 💡 How to Use These Tools

### For Validation
```bash
# Before any evaluation run:
python verify_paths.py > path_verification_log.txt

# Check the log to ensure:
# - Correct video directory
# - Feature files exist
# - Transcription CSV found
```

### For Understanding
```bash
# Pick random samples to inspect:
python inspect_sample.py --sample_id $(shuf -i 0-833 -n 1)

# Inspect each emotion class:
# - Find sample indices for each class from annotation file
# - Run inspect_sample.py on representatives
```

### For Debugging
```bash
# When model makes error:
# 1. Note the video name from error log
# 2. Inspect it:
python inspect_sample.py --sample_id <video_name>

# 3. Check if error makes sense:
#    - Is first frame misleading?
#    - Does transcription conflict with label?
#    - Are features in normal range?
```

---

## 🎯 Next Steps

### Recommended Analyses

1. **Frame Representativeness Study**
   ```bash
   # For misclassified samples, extract all frames
   # Check if first frame matches emotion label
   ffmpeg -i video.mp4 frames/frame_%03d.jpg
   ```

2. **Feature Correlation Analysis**
   ```python
   # Load features for correct vs incorrect predictions
   # Compare statistical properties
   # Identify patterns in misclassifications
   ```

3. **Transcription Text Analysis**
   ```python
   # Extract all transcriptions per emotion class
   # Build vocabulary distributions
   # Check if certain words predict emotions
   ```

4. **Temporal Analysis**
   ```bash
   # For each video, check when emotion peaks
   # Compare to first frame timing
   # Assess if first-frame limitation impacts accuracy
   ```

---

## 📞 Quick Command Reference

```bash
# Verify paths
python verify_paths.py

# Inspect first sample
python inspect_sample.py

# Inspect sample by index
python inspect_sample.py --sample_id 123

# Inspect sample by name
python inspect_sample.py --sample_id samplenew_00006611

# View extracted frame
display /tmp/<video_name>_first_frame.jpg

# Run evaluation
source .venv/bin/activate.csh
.venv/bin/torchrun --nproc_per_node 1 eval_emotion.py \
  --cfg-path eval_configs/eval_emotion.yaml \
  --dataset feature_face_caption
```

---

**Created:** October 30, 2025  
**Purpose:** Tools for validation, inspection, and analysis of Emotion-LLaMA evaluation  
**Status:** All tools tested and working  
**Documentation:** Complete with usage examples
