import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + ".")
import torch
from PIL import Image
import numpy as np
import io
import random
from diffusers import StableDiffusionPipeline, AutoencoderKL
import cv2
from omegaconf import OmegaConf
#from infinityvae.model import InfinityVAE
from diffusers import DDPMScheduler, UNet2DModel

import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import glob
import matplotlib.pyplot as plt
from io import BytesIO


# 1: gaussian_noise
# 2. gaussian_blur
# 2: color_jitter_attack,
# 3: geometric_attack,
# 4: jpeg_compression_attack,
# 5: vae_sd_attack,
# 6: infinity_vae_attack,
# 7: ctrlReg attack
# 8: diffprue attack

transform_size_to_512 = transforms.Compose([
        transforms.Resize(512, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(512),
        ])
from color_matcher import ColorMatcher
from color_matcher.normalizer import Normalizer

def color_match(ref_img, src_img):
    cm = ColorMatcher() 
    img_ref_np = Normalizer(np.asarray(ref_img)).type_norm()
    img_src_np = Normalizer(np.asarray(src_img)).type_norm()

    img_res = cm.transfer(src=img_src_np, ref=img_ref_np, method='hm-mkl-hm')   # hm-mvgd-hm / hm-mkl-hm
    img_res = Normalizer(img_res).uint8_norm()
    img_res = Image.fromarray(img_res)
    return img_res


ATTACK_TYPE = {
    1:"Gaussian",
    2:"Color",
    3:"Geometric",
    4:"JPEG",
    5:"VAE_SD",
    6:"VAE_INFINITY",
    7:"CTRL_REGEN",
}
class WatermarkAttacker:
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu",attack_type=1,attack_strength=0.5):
        self.device = device
            
        if attack_type == 5:
            self.init_vae_sd_attack()
        if attack_type == 6:
            self.init_vae_infinity_attack()
        if attack_type == 7:
            self.init_ctrlregen_attack()

        self.attack_type = attack_type
        self.attack_name = ATTACK_TYPE[attack_type]
        
    def init_vae_sd_attack(self):
        # Initialize VAE from Stable Diffusion 1.5
        try:
            self.sd_vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", torch_dtype=torch.float16).to(self.device)
            print("Stable Diffusion VAE loaded successfully")
        except Exception as e:
            print(f"Failed to load SD VAE: {e}")
            self.sd_vae = None
            
        # We'll initialize other models when they're first used to save memory

    def init_vae_infinity_attack(self):
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        vae_path = '/home/c01xuwa/CISPA-work/c01xuwa/FreqWatermark/infinity_vae_d32reg.pth'
        # load vae
        from infinity.models.bsq_vae.vae import vae_model
        schedule_mode = "dynamic"
        codebook_dim = 32
        codebook_size = 2**codebook_dim
        patch_size = 16
        encoder_ch_mult=[1, 2, 4, 4, 4]
        decoder_ch_mult=[1, 2, 4, 4, 4]
        self.vae = vae_model(vae_path, schedule_mode, codebook_dim, codebook_size, patch_size=patch_size, 
                    encoder_ch_mult=encoder_ch_mult, decoder_ch_mult=decoder_ch_mult, test_mode=True).to(device)
            
    def init_ctrlregen_attack(self):

        from controlnet_aux import CannyDetector
        from diffusers import ControlNetModel, UniPCMultistepScheduler, AutoencoderKL
        from tools.custom_i2i_pipeline import CustomStableDiffusionControlNetImg2ImgPipeline
        from transformers import AutoModel, AutoImageProcessor
        device = self.device
        DIFFUSION_MODEL = 'SG161222/Realistic_Vision_V4.0_noVAE'
        SPATIAL_CONTROL_PATH = '/home/c01xuwa/CISPA-work/c01xuwa/FreqWatermark/ctrlregen/spatialnet_ckp/spatial_control_ckp_14000'
        #SPATIAL_CONTROL_PATH = 'spatialnet_ckp/spatial_control_ckp_14000'
        SEMANTIC_CONTROL_PATH = '/home/c01xuwa/CISPA-work/c01xuwa/FreqWatermark/ctrlregen/semanticnet_ckp'
        SEMANTIC_CONTROL_NAME = 'semantic_control_ckp_435000.bin'
        IMAGE_ENCODER = 'facebook/dinov2-giant'
        VAE = 'stabilityai/sd-vae-ft-mse'

        spatialnet = [ControlNetModel.from_pretrained(SPATIAL_CONTROL_PATH, torch_dtype=torch.float16,)]
        pipe = CustomStableDiffusionControlNetImg2ImgPipeline.from_pretrained(DIFFUSION_MODEL, \
                                                                controlnet=spatialnet, \
                                                                torch_dtype=torch.float16,
                                                                safety_checker = None,
                                                                requires_safety_checker = False
                                                                )
        pipe.costum_load_ip_adapter(SEMANTIC_CONTROL_PATH, subfolder='models', weight_name=SEMANTIC_CONTROL_NAME)
        pipe.image_encoder = AutoModel.from_pretrained(IMAGE_ENCODER).to(device, dtype=torch.float16)
        pipe.feature_extractor = AutoImageProcessor.from_pretrained(IMAGE_ENCODER)
        pipe.vae = AutoencoderKL.from_pretrained(VAE).to(dtype=torch.float16)
        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
        pipe.set_ip_adapter_scale(1.0)
        pipe.set_progress_bar_config(disable=True)
        pipe.to(device)
        self.pipe = pipe
        self.processor = CannyDetector()

    def gaussian_attack(self, img):
        """Apply random Gaussian noise and blur."""
        # Convert to tensor if not already

        noise = torch.randn_like(img) * 0.1
        img = img + noise
        kernel_size = 7
        sigma = 1.0
        img = TF.gaussian_blur(img, kernel_size=kernel_size, sigma=sigma)
        img = torch.clamp(img, 0, 1)    
        return img.numpy()

    
    def color_jitter_attack(self, img):
        """Apply color jitter with hue, saturation and contrast adjustments."""
            
        # Apply color jitter
        img = TF.adjust_hue(img, random.uniform(-0.3, 0.3))
        img = TF.adjust_saturation(img, random.uniform(1.0, 3.0))
        img = TF.adjust_contrast(img, random.uniform(1.0, 3.0))
        
        return img.numpy()
    
    def geometric_attack(self, img):
        """Apply crop & resize and random rotation."""

            
        # Get dimensions
        _, h, w = img.shape
        
        # Crop and resize
        crop_factor = 0.7
        crop_size_h, crop_size_w = int(h * crop_factor), int(w * crop_factor)
        top = random.randint(0, h - crop_size_h)
        left = random.randint(0, w - crop_size_w)
        img = TF.crop(img, top, left, crop_size_h, crop_size_w)
        return img.numpy()
    
    def jpeg_compression_attack(self, img):
        """Apply JPEG compression with random quality factor."""
        img = img.cpu().numpy() # of shape (3,H,W)
        # Convert to PIL Image
        img_pil = Image.fromarray((img * 255).astype(np.uint8).transpose(1, 2, 0))

        
        # Create an in-memory buffer
        
        buffer = BytesIO()
        
        quality = 25
        img_pil.save(buffer, format="JPEG", quality=quality)
        
        # Read back the compressed image
        buffer.seek(0)
        compressed_img = Image.open(buffer)
        
        # Convert back to tensor
        compressed_img_array = np.array(compressed_img)/255.
        # Transpose to (C,H,W) format
        compressed_img_array = compressed_img_array.transpose(2, 0, 1)
        return compressed_img_array
        
        
    
    def vae_sd_attack(self, img):
        """Apply VAE from Stable Diffusion 1.5."""

        img = img.to(self.device).half()
            
        # Ensure batch dimension and proper format
        if img.dim() == 3:
            img = img.unsqueeze(0)
            
        # Scale to [-1, 1] for SD VAE
        img = 2 * img - 1
        
        with torch.no_grad():
            # Encode and decode through VAE
            latent = self.sd_vae.encode(img).latent_dist.sample()
            decoded = self.sd_vae.decode(latent).sample
            
        # Scale back to [0, 1]
        decoded = (decoded + 1) / 2
        decoded = torch.clamp(decoded, 0, 1)
        
        return decoded.squeeze(0).float().cpu().numpy()
    
    def vae_infinity_attack(self, img):
        img = img.to(self.device)
        h_div_w = 1/1
        h_div_w_template_ = h_div_w_templates[np.argmin(np.abs(h_div_w_templates-h_div_w))]
        scale_schedule = dynamic_resolution_h_w[h_div_w_template_]["1M"]['scales']
        scale_schedule = [(1, h, w) for (_, h, w) in scale_schedule]

        encoded = self.vae.encode(img.unsqueeze(0),scale_schedule = scale_schedule )
        decoded = self.vae.decode(encoded[1])
        # clamp to (0,1)
        decoded = torch.clamp(decoded, 0, 1)
        return decoded.squeeze(0).cpu().numpy()
    
    def ctrlregen_attack(self, img, step=0.5):
        img = img.cpu().numpy() # of shape (3,H,W)
        # Convert to PIL Image
        img = Image.fromarray((img * 255).astype(np.uint8).transpose(1, 2, 0))

        generator = torch.manual_seed(42)
        input_img = transform_size_to_512(img)
        processed_img = self.processor(input_img, low_threshold=100, high_threshold=150)
        prompt = 'best quality, high quality'
        negative_prompt = 'monochrome, lowres, bad anatomy, worst quality, low quality'
        output_img = self.pipe(prompt,
                        negative_prompt=negative_prompt,
                        image = [input_img],
                        control_image = [processed_img], # spatial condition
                        ip_adapter_image = [input_img],   # semantic condition
                        strength = step,
                        generator = generator,
                        num_inference_steps=50,
                        controlnet_conditioning_scale = 1.0,
                        guidance_scale = 2.0,
                        control_guidance_start = 0,
                        control_guidance_end = 1,
                        ).images[0]
        output_img = color_match(input_img, output_img)
        # upscale from 512 to 1024
        output_img = output_img.resize((1024, 1024), Image.LANCZOS)
        return np.transpose(np.array(output_img)/255.,(2,0,1))

    ## TODO
    def diffpure_attack(self, img, timestep=0.15):
        pass
        # """Apply DiffPure with specified timestep."""
        # try:
        #     # Lazily load the model when first used
        #     if not hasattr(self, 'diffusion_model'):
        #         # This is a simplified version - in practice you'd need a proper diffusion model
                
        #         self.noise_scheduler = DDPMScheduler.from_pretrained("google/ddpm-cifar10-32")
        #         self.diffusion_model = UNet2DModel.from_pretrained("google/ddpm-cifar10-32").to(self.device)
        #         print("DiffPure components loaded successfully")
        # except Exception as e:
        #     print(f"Failed to load DiffPure components: {e}")
        #     return img
            
        # if not isinstance(img, torch.Tensor):
        #     img = transforms.ToTensor()(img).to(self.device)
            
        # if img.dim() == 3:
        #     img = img.unsqueeze(0)
            
        # # Scale image to [-1, 1]
        # img = 2 * img - 1
        
        # # Forward diffusion
        # noise = torch.randn_like(img)
        # timesteps = int(timestep * self.noise_scheduler.config.num_train_timesteps)
        # noisy_img = self.noise_scheduler.add_noise(img, noise, torch.tensor([timesteps]))
        
        # # Reverse diffusion
        # with torch.no_grad():
        #     for t in range(timesteps, 0, -1):
        #         timestep = torch.tensor([t]).to(self.device)
        #         model_output = self.diffusion_model(noisy_img, timestep).sample
        #         noisy_img = self.noise_scheduler.step(model_output, t, noisy_img).prev_sample
        
        # # Scale back to [0, 1]
        # denoised = (noisy_img + 1) / 2
        # denoised = torch.clamp(denoised, 0, 1)
        
        # return denoised.squeeze(0)
    

    def apply_attack(self, img):
        """Apply the specified attack to the image."""
        #if isinstance(img, str):
        #    # Load image if path is provided
        #    img = Image.open(img).convert('RGB')
            
        attacks = {
            1: self.gaussian_attack,
            2: self.color_jitter_attack,
            3: self.geometric_attack,
            4: self.jpeg_compression_attack,
            5: self.vae_sd_attack,
            6: self.vae_infinity_attack,
            7: self.ctrlregen_attack,
            #8: self.diffpure_attack,
        }
        
        if self.attack_type not in attacks:
            raise ValueError(f"Attack type {attack_type} not supported")
        img = torch.Tensor(img)
        result = attacks[self.attack_type](img)
            
        return result