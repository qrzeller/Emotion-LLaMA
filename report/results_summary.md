# Evaluation Results Summary

**Date:** October 30, 2025  
**Model:** Emotion-LLaMA (MiniGPT-v2 + Llama-2-7b-chat)  
**Dataset:** MER2023 Test3 (834 videos)  
**Task:** 6-class Emotion Recognition

---

## 🎯 Overall Performance

| Metric | Score |
|--------|-------|
| **Accuracy** | **90.65%** |
| **Precision** (weighted) | 90.44% |
| **Recall** (weighted) | 90.65% |
| **F1 Score** (weighted) | 90.45% |

---

## 📊 Per-Class Performance

| Emotion | Samples | % of Total | Correct | Accuracy | Performance |
|---------|---------|------------|---------|----------|-------------|
| **Neutral** | 183 | 21.9% | 179 | **97.81%** | ⭐⭐⭐ Excellent |
| **Sad** | 257 | 30.8% | 250 | **97.28%** | ⭐⭐⭐ Excellent |
| **Angry** | 169 | 20.3% | 154 | **91.12%** | ⭐⭐ Very Good |
| **Happy** | 166 | 19.9% | 134 | **80.72%** | ⭐ Good |
| **Surprise** | 45 | 5.4% | 33 | **73.33%** | ⚠️ Moderate |
| **Worried** | 14 | 1.7% | 6 | **42.86%** | ❌ Challenging |

---

## 🔢 Confusion Matrix

```
                    PREDICTED
                Neu  Ang  Hap  Sad  Wor  Sur  │ Total
ACTUAL      ┌────────────────────────────────────┤
  Neutral   │ 179    1    2    1    0    0  │  183
  Angry     │   2  154    7    3    2    1  │  169
  Happy     │  10   10  134    6    2    4  │  166
  Sad       │   2    2    1  250    0    2  │  257
  Worried   │   1    0    3    1    6    3  │   14
  Surprise  │   5    1    3    2    1   33  │   45
            └────────────────────────────────────┤
  Total     │ 199  168  150  263   11   43  │  834
```

**Diagonal = Correct Predictions (Green)**  
**Off-diagonal = Misclassifications (Red)**

---

## 📈 Key Findings

### ✅ Strengths

1. **Excellent Overall Accuracy:** 90.65% across 834 test samples
2. **Strong on Dominant Classes:**
   - Sad: 97.28% (largest class with 257 samples)
   - Neutral: 97.81% (second largest with 183 samples)
3. **Robust Angry Detection:** 91.12% accuracy
4. **Minimal Cross-Class Confusion:** Most emotions well-separated

### ⚠️ Challenges

1. **Class Imbalance Issues:**
   - Worried: Only 14 samples (1.7% of dataset) → 42.86% accuracy
   - Surprise: Only 45 samples (5.4% of dataset) → 73.33% accuracy

2. **Happy Emotion Confusion:**
   - 10 cases misclassified as Neutral
   - 10 cases misclassified as Angry
   - May indicate difficulty with subtle positive expressions

3. **Worried Class Performance:**
   - Most challenging class (42.86% accuracy)
   - High confusion with Happy (3), Surprise (3), and others
   - Severely underrepresented in test set

---

## 🔍 Error Analysis

### Top Misclassification Patterns

| True Label | Predicted | Count | Possible Reasons |
|------------|-----------|-------|------------------|
| Happy | Neutral | 10 | Subtle expressions, low arousal |
| Happy | Angry | 10 | High arousal similarity, intensity confusion |
| Angry | Happy | 7 | Sarcasm, mixed emotions, cultural differences |
| Sad | Neutral | 2 | Low intensity sadness, composed expression |
| Surprise | Neutral | 5 | Brief or mild surprise, quick return to neutral |

### Notable Error

- **1 prediction:** Model output "doubt" (9-class label) instead of valid 6-class emotion
- **System handled:** Auto-corrected to "neutral" as fallback
- **Implication:** Model occasionally reverts to 9-class taxonomy despite 6-class fine-tuning

---

## 💡 Insights

### What Works Well

1. **Negative Emotions:** Sad (97.28%) and Angry (91.12%) recognized robustly
2. **Neutral State:** Highest accuracy (97.81%), rarely confused
3. **Multimodal Fusion:** Strong performance suggests effective integration of:
   - Visual features (FaceMAE, VideoMAE)
   - Audio features (HuBERT-Large)
   - Textual context (transcriptions)

### Areas for Improvement

1. **Minority Classes:**
   - Collect more samples for Worried and Surprise
   - Apply class-balancing techniques (SMOTE, weighted loss)

2. **Happy vs. Neutral/Angry:**
   - Enhance features capturing subtle positive affect
   - Investigate arousal vs. valence representations

3. **Model Consistency:**
   - Ensure strict 6-class output constraint
   - Prevent 9-class label leakage from training data

---

## 🛠️ Technical Details

### Model Configuration
- **Base Model:** Llama-2-7b-chat-hf
- **Architecture:** MiniGPT-v2 with LoRA fine-tuning
- **LoRA Parameters:**
  - Rank (r): 64
  - Alpha: 16
  - Trainable: 33.5M params (0.495% of total)
- **Checkpoint:** Stage 2 best checkpoint

### Inference Settings
- **Batch Size:** 1
- **Max New Tokens:** 500
- **Sampling:** Greedy (do_sample=False)
- **Generation Strategy:** Deterministic

### Dataset Paths
- **Videos:** `/export/home/scratch/qze/datasets/MER2023/test3`
- **Annotations:** `/export/home/scratch/qze/relative_test3_NCEV.txt`
- **Features:** MER2023-SEMI feature directories
- **Transcriptions:** `/export/home/scratch/qze/transcription_en_all.csv`

---

## 📋 Recommendations

### Immediate Actions

1. **Address Class Imbalance:**
   ```python
   # Option 1: Class weighting
   class_weights = {
       'worried': 18.35,  # 257/14
       'surprise': 5.71,  # 257/45
       'happy': 1.55,     # 257/166
       # ... others
   }
   ```

2. **Clean Generation Config:**
   - Remove `top_p` parameter or enable sampling
   - Eliminate deprecation warnings

3. **Enforce 6-Class Output:**
   - Add post-processing validation
   - Retrain with strict 6-class labels only

### Long-Term Improvements

1. **Data Augmentation:**
   - Synthesize Worried samples via generative models
   - Apply temporal/audio augmentation for Surprise

2. **Feature Analysis:**
   - Ablation study: Face vs. Video vs. Audio contributions
   - Identify which modality drives Sad/Neutral success

3. **Model Architecture:**
   - Experiment with attention visualization
   - Investigate why Happy confuses with Neutral/Angry

4. **Multi-Task Learning:**
   - Incorporate reasoning tasks (MERR annotations)
   - Joint training on emotion + valence/arousal

---

## 📝 Conclusion

The Emotion-LLaMA model achieves **strong performance** (90.65% accuracy) on the MER2023 Test3 dataset, demonstrating:

✅ Excellent recognition of Sad and Neutral emotions  
✅ Robust multimodal feature fusion  
✅ Good generalization across most emotion classes  

However, **class imbalance** significantly impacts Worried (42.86%) and Surprise (73.33%) performance. Addressing data distribution and enhancing minority class representations would likely push overall accuracy beyond 92%.

**Status:** Model ready for deployment with caveats about Worried/Surprise reliability.

---

**Generated:** October 30, 2025  
**Command:** `torchrun --nproc_per_node 1 eval_emotion.py --cfg-path eval_configs/eval_emotion.yaml --dataset feature_face_caption`  
**Runtime:** ~18 seconds model loading + inference time  
**Full Report:** See `evaluation_run_report.md`
