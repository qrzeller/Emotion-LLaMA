#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Verify which paths and files will be used during evaluation.
This script checks file existence without loading the model.
"""

import os
import sys
import yaml

def check_path(path, description):
    """Check if a path exists and report status."""
    exists = os.path.exists(path)
    symbol = "[OK]" if exists else "[XX]"
    status = "EXISTS" if exists else "MISSING"
    print(f"{symbol} [{status}] {description}")
    print(f"   Path: {path}")
    return exists

def check_directory_contents(path, description, pattern=None):
    """Check directory and optionally count files matching pattern."""
    if not os.path.isdir(path):
        print(f"[XX] [MISSING] {description}")
        print(f"   Path: {path}")
        return False, 0
    
    if pattern:
        files = [f for f in os.listdir(path) if f.endswith(pattern)]
        count = len(files)
        print(f"[OK] [EXISTS] {description}")
        print(f"   Path: {path}")
        print(f"   Files matching '{pattern}': {count}")
        if count > 0:
            print(f"   Example: {files[0]}")
        return True, count
    else:
        print(f"[OK] [EXISTS] {description}")
        print(f"   Path: {path}")
        return True, 0

print("="*80)
print("EMOTION-LLAMA PATH VERIFICATION")
print("="*80)

# Load config
config_path = "/local/user/qze/Code/Emotion-LLaMA/eval_configs/eval_emotion.yaml"
print(f"\nLoading config: {config_path}")
with open(config_path, 'r') as f:
    cfg = yaml.safe_load(f)

print("\n" + "="*80)
print("1. MODEL FILES")
print("="*80)
check_path(cfg['model']['llama_model'], "Base LLM (Llama-2-7b-chat)")
check_path(cfg['model']['ckpt'], "Fine-tuned checkpoint")

print("\n" + "="*80)
print("2. DATASET FILES - feature_face_caption")
print("="*80)
eval_cfg = cfg['evaluation_datasets']['feature_face_caption']
ann_path = eval_cfg['eval_file_path']
img_path = eval_cfg['img_path']

check_path(ann_path, "Annotation file")
if os.path.exists(ann_path):
    with open(ann_path, 'r') as f:
        lines = f.readlines()
    print(f"   Total samples: {len(lines)}")
    if len(lines) > 0:
        print(f"   First sample: {lines[0].strip()}")

print()
check_directory_contents(img_path, "Video directory", ".mp4")
check_directory_contents(img_path, "Alternative video format (.avi)", ".avi")
check_directory_contents(img_path, "Pre-extracted frames (.jpg)", ".jpg")

print("\n" + "="*80)
print("3. FEATURE FILES")
print("="*80)
ann_dir = os.path.dirname(ann_path)
print(f"Annotation directory: {ann_dir}")

# Check MER2024-style features (priority)
print("\n--- Option 1: MER2024-style (relative to annotation) ---")
mer2024_face = os.path.join(ann_dir, 'mae_340_23_UTT')
mer2024_video = os.path.join(ann_dir, 'maeVideo_399_23_UTT')
mer2024_audio = os.path.join(ann_dir, 'HL_23_UTT')

face_exists, face_count = check_directory_contents(mer2024_face, "FaceMAE features (mae_340_23_UTT)", ".npy")
video_exists, video_count = check_directory_contents(mer2024_video, "VideoMAE features (maeVideo_399_23_UTT)", ".npy")
audio_exists, audio_count = check_directory_contents(mer2024_audio, "Audio features (HL_23_UTT)", ".npy")

mer2024_available = face_exists and video_exists and audio_exists

# Check MER2023-SEMI features (fallback)
print("\n--- Option 2: MER2023-SEMI (fallback) ---")
mer2023_base = "/export/home/scratch/qze/features_of_MER2023-SEMI"
mer2023_face = os.path.join(mer2023_base, 'mae_340_UTT_MER2023-SEMI')
mer2023_video = os.path.join(mer2023_base, 'maeV_399_UTT_MER2023-SEMI')
mer2023_audio = os.path.join(mer2023_base, 'HL-UTT_MER2023-SEMI')

face_fb_exists, face_fb_count = check_directory_contents(mer2023_face, "FaceMAE features (MER2023-SEMI)", ".npy")
video_fb_exists, video_fb_count = check_directory_contents(mer2023_video, "VideoMAE features (MER2023-SEMI)", ".npy")
audio_fb_exists, audio_fb_count = check_directory_contents(mer2023_audio, "Audio features (MER2023-SEMI)", ".npy")

mer2023_available = face_fb_exists and video_fb_exists and audio_fb_exists

print("\n--- FEATURE PATH DECISION ---")
if mer2024_available:
    print("[TARGET] WILL USE: MER2024-style features (Option 1)")
    print(f"   Location: {ann_dir}")
elif mer2023_available:
    print("[WARN]  WILL USE: MER2023-SEMI fallback (Option 2)")
    print(f"   Location: {mer2023_base}")
else:
    print("[ERROR] ERROR: No valid feature directories found!")

print("\n" + "="*80)
print("4. TRANSCRIPTION FILES")
print("="*80)

# Check transcription CSV candidates
candidates = [
    (os.environ.get('EMOTION_LLAMA_TRANSCRIPT_CSV'), "Environment variable"),
    (os.path.join(ann_dir, 'transcription_en_all.csv'), "Relative to annotation (en)"),
    (os.path.join(ann_dir, 'transcription_all_new.csv'), "Relative to annotation (new)"),
    ('/export/home/scratch/qze/transcription_en_all.csv', "Common path (en)"),
    ('/export/home/scratch/qze/transcription_all_new.csv', "Common path (new)"),
]

transcript_found = None
for path, desc in candidates:
    if path and os.path.isfile(path):
        if transcript_found is None:
            print(f"[TARGET] WILL USE: {desc}")
            print(f"   Path: {path}")
            transcript_found = path
        else:
            print(f"[+] [AVAILABLE] {desc} (not used, lower priority)")
            print(f"   Path: {path}")
    else:
        print(f"[-] [MISSING] {desc}")
        if path:
            print(f"   Path: {path}")
        else:
            print(f"   Path: (not set)")

if transcript_found is None:
    print("\n[ERROR] ERROR: No transcription CSV found!")

print("\n" + "="*80)
print("5. MERR ANNOTATION FILES (for reasoning tasks)")
print("="*80)
check_path("/export/home/scratch/qze/MERR/MERR_coarse_grained.json", "MERR coarse-grained")
check_path("/export/home/scratch/qze/MERR/MERR_fine_grained.json", "MERR fine-grained")

print("\n" + "="*80)
print("6. OUTPUT DIRECTORY")
print("="*80)
output_path = cfg['run']['save_path']
output_exists = os.path.exists(output_path)
if output_exists:
    print(f"[+] [EXISTS] Output directory")
else:
    print(f"[WARN]  [WILL CREATE] Output directory")
print(f"   Path: {output_path}")

print("\n" + "="*80)
print("SUMMARY")
print("="*80)

# Sample verification - check if first video has all required files
if os.path.exists(ann_path):
    with open(ann_path, 'r') as f:
        first_line = f.readline().strip().split()
        if first_line:
            sample_name = first_line[0]
            print(f"\n[CHECK] Checking first sample: {sample_name}")
            
            # Check video
            sample_mp4 = os.path.join(img_path, sample_name + ".mp4")
            sample_avi = os.path.join(img_path, sample_name + ".avi")
            sample_jpg = os.path.join(img_path, sample_name + ".jpg")
            
            video_found = False
            if os.path.exists(sample_mp4):
                print(f"   [+] Video: {sample_name}.mp4")
                video_found = True
            elif os.path.exists(sample_avi):
                print(f"   [+] Video: {sample_name}.avi")
                video_found = True
            elif os.path.exists(sample_jpg):
                print(f"   [+] Frame: {sample_name}.jpg")
                video_found = True
            else:
                print(f"   [-] No video/frame found for {sample_name}")
            
            # Check features
            if mer2024_available:
                feat_face = os.path.join(mer2024_face, sample_name + '.npy')
                feat_video = os.path.join(mer2024_video, sample_name + '.npy')
                feat_audio = os.path.join(mer2024_audio, sample_name + '.npy')
                feat_loc = "MER2024-style"
            else:
                feat_face = os.path.join(mer2023_face, sample_name + '.npy')
                feat_video = os.path.join(mer2023_video, sample_name + '.npy')
                feat_audio = os.path.join(mer2023_audio, sample_name + '.npy')
                feat_loc = "MER2023-SEMI"
            
            print(f"   Features from: {feat_loc}")
            print(f"   {'[+]' if os.path.exists(feat_face) else '[-]'} FaceMAE")
            print(f"   {'[+]' if os.path.exists(feat_video) else '[-]'} VideoMAE")
            print(f"   {'[+]' if os.path.exists(feat_audio) else '[-]'} Audio")

print("\n" + "="*80)
print("[DONE] Verification complete!")
print("="*80)
