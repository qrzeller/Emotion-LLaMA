import glob
import os
import json
import pickle
import random
import time
import itertools
import pandas as pd
import json

import torch.nn.functional as F

import numpy as np
from PIL import Image
import skimage.io as io
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon, Rectangle
import torch
from torch.utils.data import Dataset
import webdataset as wds
import cv2

from minigpt4.datasets.datasets.base_dataset import BaseDataset

class FeatureFaceDataset(Dataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_path):

        self.vis_root = vis_root

        self.vis_processor = vis_processor
        self.text_processor = text_processor

        self.caption_instruction_pool = [
            "Please describe the details of the expression and tone the video.",
            "Can you provide a description of the facial expression and tone shown by the person in the video?",
            "Could you outline the facial expressions and vocal tones displayed in the video?",
            "Detail the expressions and tone used in the video.",
            "Explain the visual and auditory expressions captured in the video.",
            "Provide an analysis of the expressions and tone featured in the video.",
        ]

        self.emotion_instruction_pool = [
            "Please determine which emotion label in the video represents: happy, sad, neutral, angry, worried, surprise, fear, contempt, doubt.",

            # "Please determine which emotion label in the video represents: happy, sad, neutral, angry, worried, surprise.",
            # "Identify the displayed emotion in the video: is it happy, sad, neutral, angry, worried, or surprise?",
            # "Determine the emotional state shown in the video, choosing from happy, sad, neutral, angry, worried, or surprise.",
            # "Please ascertain the specific emotion portrayed in the video, whether it be happy, sad, neutral, angry, worried, or surprise.",
            # "Assess and label the emotion evident in the video: could it be happy, sad, neutral, angry, worried, surprise?",
        ]

        self.reason_instruction_pool = [
            "Please analyze all the clues in the video and reason out the emotional label of the person in the video.",
            "What is the emotional state of the person in the video? Please tell me the reason.",
            "What are the facial expressions and vocal tone used in the video? What is the intended meaning behind his words? Which emotion does this reflect?",
            "Please integrate information from various modalities to infer the emotional category of the person in the video.",
            "Could you describe the emotion-related features of the individual in the video? What emotional category do they fall into?",
        ]

        # self.task_pool = [
        #    "emotion",
        #    "reason",
        #    "infer",
        # ]

        self.task_pool = [
           "emotion",
        ]

        print("ann_path: ", ann_path)
        self.ann_path = ann_path
        self.file_path = os.path.dirname(ann_path)
        self.tmp = [x.strip().split(' ') for x in open(ann_path)]
        print(('video number:%d' % (len(self.tmp))))

        # emos = ['neutral', 'angry', 'happy', 'sad', 'worried', 'surprise']
        emos = ['neutral', 'angry', 'happy', 'sad', 'worried', 'surprise', 'fear', 'contempt', 'doubt']

        self.emo2idx, self.idx2emo = {}, {}
        for ii, emo in enumerate(emos): self.emo2idx[emo] = ii
        for ii, emo in enumerate(emos): self.idx2emo[ii] = emo

        json_file_path = "/export/home/scratch/qze/MERR/MERR_coarse_grained.json" 
        with open(json_file_path, 'r') as json_file:
            self.MERR_coarse_grained_dict = json.load(json_file)

        reason_json_file_path = "/export/home/scratch/qze/MERR/MERR_fine_grained.json"
        with open(reason_json_file_path, 'r') as json_file:
            self.MERR_fine_grained_dict = json.load(json_file)

        # Resolve transcription CSV (MER2023/MER2024): allow env override and local fallback
        candidates = []
        env_csv = os.environ.get('EMOTION_LLAMA_TRANSCRIPT_CSV')
        if env_csv:
            candidates.append(env_csv)
        # Try alongside the annotation file
        candidates.append(os.path.join(self.file_path, 'transcription_en_all.csv'))
        candidates.append(os.path.join(self.file_path, 'transcription_all_new.csv'))
        # Common cluster paths fallback
        candidates.append('/export/home/scratch/qze/transcription_en_all.csv')
        candidates.append('/export/home/scratch/qze/transcription_all_new.csv')

        self.character_lines = None
        transcription_csv_path = None
        for cand in candidates:
            if cand and os.path.isfile(cand):
                self.character_lines = pd.read_csv(cand)
                transcription_csv_path = cand
                break
        if self.character_lines is None:
            raise FileNotFoundError(
                'Could not find transcription CSV. Set EMOTION_LLAMA_TRANSCRIPT_CSV or place "transcription_en_all.csv" '
                f'in the same directory as the annotation file: {self.file_path}'
            )

        # Determine sentence column
        if 'sentence' in self.character_lines.columns:
            self._sentence_col = 'sentence'
        elif 'sentence_en' in self.character_lines.columns:
            self._sentence_col = 'sentence_en'
        else:
            raise KeyError(
                'Transcription CSV missing expected columns. Need one of: sentence, sentence_en. '
                f'Columns found: {list(self.character_lines.columns)}'
            )
        
        # Log transcription source
        print(f"✓ Loaded transcription CSV: {transcription_csv_path}")
        print(f"  - Using column: '{self._sentence_col}'")
        print(f"  - Total transcriptions: {len(self.character_lines)}")


    def __len__(self):
        return len(self.tmp)

    def __getitem__(self, index):
        t = self.tmp[index]
        video_name = t[0]

        mp4_path = os.path.join(self.vis_root, video_name + ".mp4")
        avi_path = os.path.join(self.vis_root, video_name + ".avi")
        jpg_path = os.path.join(self.vis_root, video_name + ".jpg")

        # Log video format used for first sample only
        if index == 0:
            print(f"✓ Video source directory: {self.vis_root}")

        if os.path.exists(mp4_path):
            if index == 0:
                print(f"  - Using .mp4 video files (extracting first frame)")
            image = self.extract_frame(mp4_path)
        elif os.path.exists(avi_path):
            if index == 0:
                print(f"  - Using .avi video files (extracting first frame)")
            image = self.extract_frame(avi_path)
        elif os.path.exists(jpg_path):
            if index == 0:
                print(f"  - Using pre-extracted .jpg frames")
            # Fallback: use pre-extracted frame if available
            image = Image.open(jpg_path).convert("RGB")
            image = np.array(image)
        else:
            raise FileNotFoundError(
                f"Could not find video/image for '{video_name}'. Searched: {mp4_path}, {avi_path}, {jpg_path}"
            )

        image = Image.fromarray(image.astype('uint8'))
        image = image.convert('RGB')
        image = self.vis_processor(image)


        # image_file = '{}.jpg'.format(video_name)
        # image_path = os.path.join(self.vis_root, image_file)
        # image = Image.open(image_path).convert("RGB")
        # image = self.vis_processor(image)


        FaceMAE_feats, VideoMAE_feats, Audio_feats = self.get(video_name)
        if len(VideoMAE_feats.shape) == 1:
            VideoMAE_feats = VideoMAE_feats.unsqueeze(0)
        if len(Audio_feats.shape) == 1:
            Audio_feats = Audio_feats.unsqueeze(0)
        if len(FaceMAE_feats.shape) == 1:
            FaceMAE_feats = FaceMAE_feats.unsqueeze(0)
        video_features = torch.cat((FaceMAE_feats, VideoMAE_feats, Audio_feats), dim=0)


        # random task
        task = random.choice(self.task_pool)
        if task == "emotion":
            caption = t[2] # llama2 putput only emotion class
            caption = self.text_processor(caption)
            instruction_pool = self.emotion_instruction_pool
        elif task == "reason":
            caption = self.MERR_coarse_grained_dict[video_name]['caption']

            caption = self.text_processor(caption)
            instruction_pool = self.reason_instruction_pool

        elif task == "reason_v2":
            caption = self.MERR_fine_grained_dict[video_name]['smp_reason_caption']

            # caption = "" # for test reasoning

            caption = self.text_processor(caption)
            instruction_pool = self.reason_instruction_pool


        emotion = self.emo2idx[t[2]]
        sentence_row = self.character_lines.loc[self.character_lines['name'] == video_name, self._sentence_col]
        if len(sentence_row.values) == 0:
            raise KeyError(f"Name '{video_name}' not found in transcription CSV for column {self._sentence_col}.")
        sentence = sentence_row.values[0]
        character_line = "The person in video says: {}. ".format(sentence)
        
        instruction = "<video><VideoHere></video> <feature><FeatureHere></feature> {} [{}] {} ".format(character_line, task, random.choice(instruction_pool))

        return {
            "image": image,
            "video_features": video_features,
            "instruction_input": instruction,
            "answer": caption,
            "emotion": emotion,
            "image_id": video_name
        }
    
    def extract_frame(self, video_path):
        video_capture = cv2.VideoCapture(video_path)
        success, frame = video_capture.read()
        if not success:
            raise ValueError("Failed to read video file:", video_path)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        video_capture.release()

        return frame_rgb


    def get(self, video_name):
        """
        Load precomputed features for the given video name.

        Priority:
        1) If MER2024-style directories exist next to the annotation file (self.file_path),
           use them: mae_340_23_UTT, maeVideo_399_23_UTT, HL_23_UTT.
        2) Otherwise, fall back to the original MER2023-SEMI hardcoded base path and names.
        """

        # Option 1: MER2024-style relative directories next to ann file
        rel_face = os.path.join(self.file_path, 'mae_340_23_UTT', video_name + '.npy')
        rel_video = os.path.join(self.file_path, 'maeVideo_399_23_UTT', video_name + '.npy')
        rel_audio = os.path.join(self.file_path, 'HL_23_UTT', video_name + '.npy')

        if os.path.isfile(rel_face) and os.path.isfile(rel_video) and os.path.isfile(rel_audio):
            # Log first time to confirm MER2024-style features are used
            if not hasattr(self, '_feature_path_logged'):
                print(f"✓ Using MER2024-style features from: {self.file_path}")
                print(f"  - FaceMAE: mae_340_23_UTT/")
                print(f"  - VideoMAE: maeVideo_399_23_UTT/")
                print(f"  - Audio: HL_23_UTT/")
                self._feature_path_logged = True
            FaceMAE_feats = torch.tensor(np.load(rel_face))
            VideoMAE_feats = torch.tensor(np.load(rel_video))
            Audio_feats = torch.tensor(np.load(rel_audio))
            return FaceMAE_feats, VideoMAE_feats, Audio_feats

        # Option 2: MER2023-SEMI hardcoded base path
        features_base_path = "/export/home/scratch/qze/features_of_MER2023-SEMI"

        # Log first time to confirm fallback is used
        if not hasattr(self, '_feature_path_logged'):
            print(f"⚠ Using MER2023-SEMI fallback features from: {features_base_path}")
            print(f"  - FaceMAE: mae_340_UTT_MER2023-SEMI/")
            print(f"  - VideoMAE: maeV_399_UTT_MER2023-SEMI/")
            print(f"  - Audio: HL-UTT_MER2023-SEMI/")
            self._feature_path_logged = True

        FaceMAE_feats_path = os.path.join(features_base_path, 'mae_340_UTT_MER2023-SEMI', video_name + '.npy')
        VideoMAE_feats_path = os.path.join(features_base_path, 'maeV_399_UTT_MER2023-SEMI', video_name + '.npy')
        Audio_feats_path = os.path.join(features_base_path, 'HL-UTT_MER2023-SEMI', video_name + '.npy')

        FaceMAE_feats = torch.tensor(np.load(FaceMAE_feats_path))
        VideoMAE_feats = torch.tensor(np.load(VideoMAE_feats_path))
        Audio_feats = torch.tensor(np.load(Audio_feats_path))

        return FaceMAE_feats, VideoMAE_feats, Audio_feats
