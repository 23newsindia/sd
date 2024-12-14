import os
import numpy as np
from PIL import Image

def load_pose_images(pose_directory, width, height):
    pose_images = []
    for filename in sorted(os.listdir(pose_directory)):
        if filename.endswith(('.png', '.jpg')):
            img = Image.open(os.path.join(pose_directory, filename)).convert('RGB')
            img = img.resize((width, height))
            pose_images.append(img)
    return pose_images

def convert_output_frames(output_frames):
    frames = []
    for frame in output_frames:
        frame_np = np.array(frame).astype(np.float32) / 255.0
        frames.append(frame_np)
    return np.stack(frames)