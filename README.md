# StableAnimator ComfyUI Node

This is a ComfyUI custom node implementation of StableAnimator, allowing you to generate high-quality identity-preserving human image animations directly in ComfyUI.

## Installation

1. Clone this repository into your ComfyUI custom_nodes folder:
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/your-username/StableAnimator .
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3. Download the model weights and place them in the checkpoints folder:
- stable-video-diffusion-img2vid-xt
- Animation/pose_net.pth
- Animation/face_encoder.pth  
- Animation/unet.pth

## Usage

1. Load the StableAnimator pipeline using the "Load StableAnimator Pipeline" node
2. Connect your reference image and pose directory to the "Generate Animation" node
3. Configure the generation parameters as needed
4. Run the workflow to generate your animation

## Parameters

- width/height: Supported resolutions are 512x512 and 576x1024
- guidance_scale: Controls how closely to follow the pose guidance (recommended: 3.0)
- num_inference_steps: Number of denoising steps (recommended: 25)
- fps: Output video frame rate
- frames_overlap: Overlap between processed frame chunks (recommended: 4)
- tile_size: Size of processed tiles (recommended: 16)
- noise_aug_strength: Strength of noise augmentation (recommended: 0.02)
- seed: Random seed for reproducibility (-1 for random)

## License

This project is licensed under the MIT License. See the LICENSE file for details.