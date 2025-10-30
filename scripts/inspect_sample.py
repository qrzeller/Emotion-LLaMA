#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Inspect a sample from the evaluation dataset.
Shows all human-readable elements: video, features, prompt, ground truth, etc.
"""

import os
import sys
import yaml
import numpy as np
import pandas as pd
from PIL import Image
import cv2

def print_section(title):
    """Print a section header."""
    print("\n" + "="*80)
    print(title)
    print("="*80)

def inspect_video(video_path):
    """Extract and display information about a video file."""
    if not os.path.exists(video_path):
        print(f"[ERROR] Video not found: {video_path}")
        return None
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = frame_count / fps if fps > 0 else 0
    
    print(f"Video Properties:")
    print(f"  Path: {video_path}")
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {fps:.2f}")
    print(f"  Total Frames: {frame_count}")
    print(f"  Duration: {duration:.2f} seconds")
    
    # Extract first frame (what the model actually uses)
    success, frame = cap.read()
    if success:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        print(f"\n  First Frame Extracted: {frame_rgb.shape}")
        print(f"  Note: Only the FIRST FRAME is used by the model, not the full video!")
        cap.release()
        return frame_rgb
    else:
        print("  [ERROR] Failed to extract first frame")
        cap.release()
        return None

def inspect_features(video_name, ann_dir, mer2023_base):
    """Load and inspect pre-extracted features."""
    print("\nFeature Files:")
    
    # Try MER2024-style first
    rel_face = os.path.join(ann_dir, 'mae_340_23_UTT', video_name + '.npy')
    rel_video = os.path.join(ann_dir, 'maeVideo_399_23_UTT', video_name + '.npy')
    rel_audio = os.path.join(ann_dir, 'HL_23_UTT', video_name + '.npy')
    
    if os.path.isfile(rel_face) and os.path.isfile(rel_video) and os.path.isfile(rel_audio):
        print("  Source: MER2024-style (relative to annotation)")
        face_path, video_path, audio_path = rel_face, rel_video, rel_audio
    else:
        print("  Source: MER2023-SEMI (fallback)")
        face_path = os.path.join(mer2023_base, 'mae_340_UTT_MER2023-SEMI', video_name + '.npy')
        video_path = os.path.join(mer2023_base, 'maeV_399_UTT_MER2023-SEMI', video_name + '.npy')
        audio_path = os.path.join(mer2023_base, 'HL-UTT_MER2023-SEMI', video_name + '.npy')
    
    # Load features
    try:
        face_feat = np.load(face_path)
        video_feat = np.load(video_path)
        audio_feat = np.load(audio_path)
        
        print(f"\n  1. FaceMAE Features:")
        print(f"     Path: {face_path}")
        print(f"     Shape: {face_feat.shape}")
        print(f"     Dtype: {face_feat.dtype}")
        print(f"     Range: [{face_feat.min():.4f}, {face_feat.max():.4f}]")
        print(f"     Mean: {face_feat.mean():.4f}, Std: {face_feat.std():.4f}")
        
        print(f"\n  2. VideoMAE Features:")
        print(f"     Path: {video_path}")
        print(f"     Shape: {video_feat.shape}")
        print(f"     Dtype: {video_feat.dtype}")
        print(f"     Range: [{video_feat.min():.4f}, {video_feat.max():.4f}]")
        print(f"     Mean: {video_feat.mean():.4f}, Std: {video_feat.std():.4f}")
        
        print(f"\n  3. Audio Features (HuBERT-Large):")
        print(f"     Path: {audio_path}")
        print(f"     Shape: {audio_feat.shape}")
        print(f"     Dtype: {audio_feat.dtype}")
        print(f"     Range: [{audio_feat.min():.4f}, {audio_feat.max():.4f}]")
        print(f"     Mean: {audio_feat.mean():.4f}, Std: {audio_feat.std():.4f}")
        
        print(f"\n  Combined Features (concatenated):")
        print(f"     Total dimensions: {face_feat.size + video_feat.size + audio_feat.size}")
        
        return True
    except Exception as e:
        print(f"  [ERROR] Failed to load features: {e}")
        return False

def get_transcription(video_name, ann_dir):
    """Get transcription for the video."""
    # Search for transcription CSV
    candidates = [
        os.path.join(ann_dir, 'transcription_en_all.csv'),
        os.path.join(ann_dir, 'transcription_all_new.csv'),
        '/export/home/scratch/qze/transcription_en_all.csv',
        '/export/home/scratch/qze/transcription_all_new.csv',
    ]
    
    for csv_path in candidates:
        if os.path.isfile(csv_path):
            df = pd.read_csv(csv_path)
            
            # Determine column
            if 'sentence_en' in df.columns:
                col = 'sentence_en'
            elif 'sentence' in df.columns:
                col = 'sentence'
            else:
                continue
            
            # Find transcription
            row = df.loc[df['name'] == video_name, col]
            if len(row) > 0:
                print(f"\nTranscription:")
                print(f"  Source: {csv_path}")
                print(f"  Column: {col}")
                print(f'  Text: "{row.values[0]}"')
                return row.values[0]
    
    print("\n[ERROR] Transcription not found")
    return None

def construct_prompt(transcription, task="emotion"):
    """Construct the actual prompt sent to the model."""
    
    # Instruction pools from first_face.py
    emotion_instructions = [
        "Please determine which emotion label in the video represents: happy, sad, neutral, angry, worried, surprise, fear, contempt, doubt.",
    ]
    
    # Character line
    character_line = f"The person in video says: {transcription}. "
    
    # Full instruction (note: <video> and <feature> are special tokens)
    instruction = f"<video><VideoHere></video> <feature><FeatureHere></feature> {character_line}[{task}] {emotion_instructions[0]}"
    
    # Llama-2 prompt template
    prompt_template = "[INST] {} [/INST]"
    final_prompt = prompt_template.format(instruction)
    
    return instruction, final_prompt

# Main inspection
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Inspect a sample from the evaluation dataset')
    parser.add_argument('--sample_id', type=str, default=None, 
                        help='Video name (e.g., samplenew_00006611) or sample index (e.g., 0)')
    parser.add_argument('--config', type=str, 
                        default='/local/user/qze/Code/Emotion-LLaMA/eval_configs/eval_emotion.yaml',
                        help='Path to eval config')
    args = parser.parse_args()
    
    # Load config
    print_section("SAMPLE INSPECTOR - Emotion-LLaMA Evaluation")
    print(f"Config: {args.config}")
    
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)
    
    eval_cfg = cfg['evaluation_datasets']['feature_face_caption']
    ann_path = eval_cfg['eval_file_path']
    img_path = eval_cfg['img_path']
    ann_dir = os.path.dirname(ann_path)
    mer2023_base = "/export/home/scratch/qze/features_of_MER2023-SEMI"
    
    # Read annotation file
    with open(ann_path, 'r') as f:
        annotations = [line.strip().split() for line in f.readlines()]
    
    # Determine which sample to inspect
    if args.sample_id is None:
        # Use first sample
        sample_idx = 0
        video_name = annotations[0][0]
        print(f"\nNo sample specified, using first sample (index 0)")
    elif args.sample_id.isdigit():
        # Sample index
        sample_idx = int(args.sample_id)
        video_name = annotations[sample_idx][0]
        print(f"\nUsing sample at index {sample_idx}")
    else:
        # Video name
        video_name = args.sample_id
        sample_idx = None
        for i, ann in enumerate(annotations):
            if ann[0] == video_name:
                sample_idx = i
                break
        if sample_idx is None:
            print(f"[ERROR] Video '{video_name}' not found in annotations")
            sys.exit(1)
        print(f"\nUsing sample '{video_name}' (index {sample_idx})")
    
    # Parse annotation
    ann_data = annotations[sample_idx]
    emotion_id = ann_data[1]
    emotion_label = ann_data[2]
    
    print_section(f"SAMPLE: {video_name} (Index {sample_idx})")
    
    # 1. Annotation Info
    print(f"\nAnnotation Entry:")
    print(f"  Raw line: {' '.join(ann_data)}")
    print(f"  Video name: {video_name}")
    print(f"  Emotion ID: {emotion_id}")
    print(f"  GROUND TRUTH EMOTION: {emotion_label}")
    
    # 2. Video Inspection
    print_section("VIDEO ANALYSIS")
    video_file = None
    for ext in ['.mp4', '.avi', '.jpg']:
        video_file = os.path.join(img_path, video_name + ext)
        if os.path.exists(video_file):
            break
    
    if video_file and video_file.endswith(('.mp4', '.avi')):
        first_frame = inspect_video(video_file)
    elif video_file and video_file.endswith('.jpg'):
        print(f"Using pre-extracted frame: {video_file}")
        first_frame = np.array(Image.open(video_file))
        print(f"  Frame shape: {first_frame.shape}")
    else:
        print("[ERROR] No video/frame found")
        first_frame = None
    
    # 3. Feature Inspection
    print_section("FEATURE ANALYSIS")
    inspect_features(video_name, ann_dir, mer2023_base)
    
    # 4. Transcription
    print_section("TRANSCRIPTION")
    transcription = get_transcription(video_name, ann_dir)
    
    # 5. Prompt Construction
    print_section("MODEL INPUT PROMPT")
    if transcription:
        instruction, final_prompt = construct_prompt(transcription)
        
        print("\nInstruction (before Llama-2 template):")
        print("-" * 80)
        print(instruction)
        print("-" * 80)
        
        print("\nFinal Prompt (with Llama-2 template):")
        print("-" * 80)
        print(final_prompt)
        print("-" * 80)
        
        print("\nSpecial Tokens:")
        print("  <video><VideoHere></video>   - Placeholder for visual features from first frame")
        print("  <feature><FeatureHere></feature> - Placeholder for FaceMAE+VideoMAE+Audio features")
        print("  [INST] ... [/INST]           - Llama-2 instruction format")
    
    # 6. Expected Output
    print_section("EXPECTED MODEL OUTPUT")
    print(f"\nGround Truth: {emotion_label}")
    print(f"\nExpected model response format: '{emotion_label}' (single word)")
    print(f"\nValid emotions: neutral, angry, happy, sad, worried, surprise")
    print(f"Note: Model sometimes outputs: fear, contempt, doubt (9-class set)")
    
    # 7. Summary
    print_section("SUMMARY")
    print(f"\nVideo: {video_name}")
    print(f"  Location: {video_file}")
    print(f"  Input Method: First frame only (not full video)")
    print(f"\nGround Truth: {emotion_label}")
    print(f"\nTranscription: \"{transcription}\"")
    print(f"\nFeatures: 3 modalities (Face + Video + Audio) from MER2023-SEMI")
    print(f"\nTask: Emotion classification (6 classes)")
    print(f"\nModel: Emotion-LLaMA (MiniGPT-v2 + Llama-2-7b-chat)")
    print(f"  Checkpoint: {cfg['model']['ckpt']}")
    
    print("\n" + "="*80)
    print("[DONE] Sample inspection complete")
    print("="*80)
    
    # Additional: Suggest viewing the frame
    if first_frame is not None:
        output_frame = f"/tmp/{video_name}_first_frame.jpg"
        Image.fromarray(first_frame).save(output_frame)
        print(f"\nFirst frame saved to: {output_frame}")
        print("You can view it with: display {output_frame}  (if ImageMagick installed)")
