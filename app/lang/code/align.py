from transformers import AutoProcessor, AutoModelForZeroShotImageClassification
import torch
import hashlib
import os
import cv2
import numpy as np
from PIL import Image
import json
import pickle
import time

class AltCLIPSimilarityCalculator:
    def __init__(self, model_path="/mnt/gzz/turbo/models/align"):
        """Initialize the AltCLIP similarity calculator"""
        # Set the device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model = AutoModelForZeroShotImageClassification.from_pretrained(model_path)
        self.processor = AutoProcessor.from_pretrained(model_path)
        
        # Move the model to GPU and set to evaluation mode
        self.model = self.model.to(self.device)
        self.model.eval()  # Set to evaluation mode to improve inference speed
        
        # Initialize memory cache dictionaries to store feature vectors of video frames and text
        self.video_cache = {}
        self.text_cache = {}

    def calculate_similarity_optimized(self, video_paths, texts):
        """Calculate the similarity between video and text
        
        Args:
            video_paths: List of video paths
            texts: List of texts, corresponding to videos one-to-one
            
        Returns:
            float: Average similarity of all video-text pairs
        """
        if len(video_paths) != len(texts):
            raise ValueError("The number of videos and texts must be equal")
        
        similarities = []
        total_pairs = len(video_paths)
        
        for i in range(total_pairs):
            video_path = video_paths[i]
            text = texts[i]
            
            # Calculate the similarity between video and text
            similarity = self._compute_video_text_similarity(video_path, text)
            similarities.append(similarity)
        
        # Calculate average similarity
        avg_similarity = sum(similarities) / total_pairs if similarities else 0
        
        return avg_similarity
    
    def _compute_video_text_similarity(self, video_path, text):
        """Calculate the similarity between a single video and text"""
        # Get video features
        video_features = self._get_video_features(video_path)
        
        # Get text features
        text_features = self._get_text_features(text)
        
        # Calculate similarity (using cosine similarity)
        similarity = torch.nn.functional.cosine_similarity(video_features, text_features, dim=1).mean().item()
        
        return similarity
    
    def _get_video_features(self, video_path):
        """Extract features from video, using cache to avoid redundant computation"""
        # Calculate the hash value of the video path as the cache key
        video_hash = hashlib.md5(video_path.encode()).hexdigest()
        
        # If the feature vector of this video exists in the cache, return it directly
        if video_hash in self.video_cache:
            return self.video_cache[video_hash]
        
        # Sample 8 frames from the video
        frames = self._sample_frames_from_video(video_path, num_frames=8)
        
        # Initialize frame features list
        frame_features_list = []
        
        # Extract features for each frame
        for frame in frames:
            # Convert OpenCV BGR image format to PIL RGB format
            frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            # Process image
            inputs = self.processor(images=frame_pil, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Extract features
            with torch.no_grad():
                outputs = self.model.get_image_features(**inputs)
                
            # Normalize feature vectors
            frame_features = outputs / outputs.norm(dim=1, keepdim=True)
            frame_features_list.append(frame_features)
        
        # Compute the average of all frame features as video features
        video_features = torch.mean(torch.cat(frame_features_list, dim=0), dim=0, keepdim=True)
        
        # Cache video features
        self.video_cache[video_hash] = video_features
        
        return video_features
    
    def _get_text_features(self, text):
        """Extract features from text, using cache to avoid redundant computation"""
        # Calculate the hash value of the text as the cache key
        text_hash = hashlib.md5(text.encode()).hexdigest()
        
        # If the feature vector of this text exists in the cache, return it directly
        if text_hash in self.text_cache:
            return self.text_cache[text_hash]
        
        # Process text
        inputs = self.processor(text=text, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Extract features
        with torch.no_grad():
            outputs = self.model.get_text_features(**inputs)
            
        # Normalize feature vectors
        text_features = outputs / outputs.norm(dim=1, keepdim=True)
        
        # Cache text features
        self.text_cache[text_hash] = text_features
        
        return text_features
    
    def _sample_frames_from_video(self, video_path, num_frames=8):
        """Uniformly sample a specified number of frames from the video"""
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file does not exist: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        
        # Get the total number of frames in the video
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames == 0:
            raise ValueError(f"Unable to read video frames: {video_path}")
        
        # Calculate sampling interval
        if total_frames <= num_frames:
            # If the total number of frames is less than or equal to the required number, sample all frames
            frame_indices = list(range(total_frames))
        else:
            # Uniform sampling
            frame_indices = [int(i * total_frames / num_frames) for i in range(num_frames)]
        
        frames = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
        
        cap.release()
        
        if not frames:
            raise ValueError(f"Unable to extract frames from video: {video_path}")
        
        return frames
        
if __name__ == "__main__":
    # Create an instance of the similarity calculator
    calculator = AltCLIPSimilarityCalculator()
    
    # Test video and text paths
    video_paths = ["/mnt/gzz/turbo/app/lang/dataset/candidate_video/segment_001.mp4", "/mnt/gzz/turbo/app/lang/dataset/candidate_video/segment_002.mp4"]
    texts = ["Description of the first video", "Description of the second video"]
    
    try:
        avg_similarity = calculator.calculate_similarity_optimized(video_paths, texts)
        print(f"Average similarity: {avg_similarity}")
    except ValueError as e:
        print(f"Error: {e}")
