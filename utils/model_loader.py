import torch
from diffusers.models.attention_processor import XFormersAttnProcessor
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
from diffusers import AutoencoderKLTemporalDecoder, EulerDiscreteScheduler

from ..models.attention_processor import AnimationAttnProcessor, AnimationIDAttnNormalizedProcessor
from ..models.face_model import FaceModel
from ..models.id_encoder import FusionFaceId
from ..models.pose_net import PoseNet
from ..models.unet import UNetSpatioTemporalConditionModel

def load_models(model_path, posenet_path=None, face_encoder_path=None, unet_path=None):
    feature_extractor = CLIPImageProcessor.from_pretrained(model_path, subfolder="feature_extractor")
    noise_scheduler = EulerDiscreteScheduler.from_pretrained(model_path, subfolder="scheduler")
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(model_path, subfolder="image_encoder")
    vae = AutoencoderKLTemporalDecoder.from_pretrained(model_path, subfolder="vae")
    unet = UNetSpatioTemporalConditionModel.from_pretrained(model_path, subfolder="unet", low_cpu_mem_usage=True)
    
    pose_net = PoseNet(noise_latent_channels=unet.config.block_out_channels[0])
    face_encoder = FusionFaceId(cross_attention_dim=1024, id_embeddings_dim=512, clip_embeddings_dim=1024, num_tokens=4)
    face_model = FaceModel()

    # Load model weights
    if posenet_path and os.path.exists(posenet_path):
        pose_net.load_state_dict(torch.load(posenet_path))
    if face_encoder_path and os.path.exists(face_encoder_path):
        face_encoder.load_state_dict(torch.load(face_encoder_path))
    if unet_path and os.path.exists(unet_path):
        unet.load_state_dict(torch.load(unet_path))

    return feature_extractor, noise_scheduler, image_encoder, vae, unet, pose_net, face_encoder, face_model