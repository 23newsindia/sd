import os
import torch
import numpy as np
from PIL import Image

from .utils.model_loader import load_models
from .utils.image_utils import load_pose_images, convert_output_frames
from .pipeline import InferenceAnimationPipeline

class StableAnimatorLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_path": ("STRING", {"default": "checkpoints/stable-video-diffusion-img2vid-xt"}),
                "posenet_path": ("STRING", {"default": "checkpoints/Animation/pose_net.pth"}),
                "face_encoder_path": ("STRING", {"default": "checkpoints/Animation/face_encoder.pth"}),
                "unet_path": ("STRING", {"default": "checkpoints/Animation/unet.pth"}),
            }
        }
    
    RETURN_TYPES = ("STABLEANIMATOR_PIPELINE",)
    FUNCTION = "load_pipeline"
    CATEGORY = "StableAnimator"

    def load_pipeline(self, model_path, posenet_path, face_encoder_path, unet_path):
        feature_extractor, noise_scheduler, image_encoder, vae, unet, pose_net, face_encoder, face_model = load_models(
            model_path, posenet_path, face_encoder_path, unet_path
        )

        pipeline = InferenceAnimationPipeline(
            vae=vae,
            image_encoder=image_encoder,
            unet=unet,
            scheduler=noise_scheduler,
            feature_extractor=feature_extractor,
            pose_net=pose_net,
            face_encoder=face_encoder,
        )

        return (pipeline,)

class StableAnimatorGenerator:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pipeline": ("STABLEANIMATOR_PIPELINE",),
                "reference_image": ("IMAGE",),
                "pose_directory": ("STRING", {"default": "poses"}),
                "width": ("INT", {"default": 512, "min": 512, "max": 1024}),
                "height": ("INT", {"default": 512, "min": 512, "max": 1024}),
                "guidance_scale": ("FLOAT", {"default": 3.0, "min": 1.0, "max": 20.0}),
                "num_inference_steps": ("INT", {"default": 25, "min": 1, "max": 100}),
                "fps": ("INT", {"default": 8, "min": 1, "max": 60}),
                "frames_overlap": ("INT", {"default": 4, "min": 0, "max": 16}),
                "tile_size": ("INT", {"default": 16, "min": 8, "max": 64}),
                "noise_aug_strength": ("FLOAT", {"default": 0.02, "min": 0.0, "max": 1.0}),
                "seed": ("INT", {"default": -1}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate"
    CATEGORY = "StableAnimator"

    def generate(self, pipeline, reference_image, pose_directory, width, height, guidance_scale, 
                num_inference_steps, fps, frames_overlap, tile_size, noise_aug_strength, seed):
        
        if seed == -1:
            seed = torch.randint(0, 2**32 - 1, (1,)).item()
        
        generator = torch.Generator("cuda").manual_seed(seed)
        
        # Convert reference image
        reference_pil = Image.fromarray(np.uint8(reference_image * 255))
        
        # Load pose images
        pose_images = load_pose_images(pose_directory, width, height)

        # Generate animation
        output = pipeline(
            image=reference_pil,
            image_pose=pose_images,
            height=height,
            width=width,
            num_frames=len(pose_images),
            tile_size=tile_size,
            tile_overlap=frames_overlap,
            motion_bucket_id=127,
            fps=fps,
            min_guidance_scale=guidance_scale,
            max_guidance_scale=guidance_scale,
            noise_aug_strength=noise_aug_strength,
            num_inference_steps=num_inference_steps,
            generator=generator,
            output_type="pil"
        ).frames[0]

        return (convert_output_frames(output),)

NODE_CLASS_MAPPINGS = {
    "StableAnimatorLoader": StableAnimatorLoader,
    "StableAnimatorGenerator": StableAnimatorGenerator
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "StableAnimatorLoader": "Load StableAnimator Pipeline",
    "StableAnimatorGenerator": "Generate Animation"
}