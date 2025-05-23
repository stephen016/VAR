import argparse
import os
import numpy as np
from models import VQVAE, build_vae_var
import torch
import random
from PIL import Image

def parse_args():
    parser = argparse.ArgumentParser(description="Test Attack")
    parser.add_argument(
        "--model_path",
        type=str,
        default="gpt-3.5-turbo",
        help="Model to use for the attack",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=100,
        help="Number of samples to generate",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for generation",
    )
    return parser.parse_args()

def main():
    args = parse_args()
    vae_ckpt =  '/home/c01xuwa/CISPA-home/VAR/vae_ch160v4096z32.pth'
    var_ckpt = args.model_path
    base_dir = "/".join(var_ckpt.split('/')[:-1])
    # remove best state to save space
    
    #extra_var_ckpt = var_ckpt.replace('last', 'best')
    #if os.path.exists(extra_var_ckpt):
    #    os.remove(extra_var_ckpt)
    
    MODEL_DEPTH = int(base_dir.split('_')[-1])
    patch_nums = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)
    device = 'cuda'

    vae, var = build_vae_var(
        V=4096, Cvae=32, ch=160, share_quant_resi=4,    # hard-coded VQVAE hyperparameters
        device=device, patch_nums=patch_nums,
        num_classes=1000, depth=MODEL_DEPTH, shared_aln=False,
    )

    vae.load_state_dict(torch.load(vae_ckpt, map_location='cpu'), strict=True)
    
    if "last" in var_ckpt:
        var.load_state_dict(torch.load(var_ckpt, map_location='cpu',weights_only=True)['trainer']['var_wo_ddp'], strict=True)
    else:
        var.load_state_dict(torch.load(var_ckpt, map_location='cpu'), strict=True)
    vae.eval(), var.eval()
    for p in vae.parameters(): p.requires_grad_(False)
    for p in var.parameters(): p.requires_grad_(False)
    print(f'prepare finished.')

    # inference
    target_class = 0
    clean_class = list(range(1,1000))
    # Set seeds for reproducibility
    seed = 1
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Performance optimization
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision('high')

    # Configuration
    num_sampling_steps = 250
    cfg = 1.5
    more_smooth = True
    print(f"Generating {args.n_samples} images of target class {target_class}")

    # Add batch_size parameter from args
    batch_size = args.batch_size
    remaining_samples = args.n_samples
    target_images = []

    # Generate target class images in batches
    for i in range(0, args.n_samples, batch_size):
        current_batch_size = min(batch_size, remaining_samples)
        batch_labels = [target_class] * current_batch_size
        label_B = torch.tensor(batch_labels, device=device)
        
        with torch.inference_mode():
            with torch.autocast('cuda', enabled=True, dtype=torch.float16, cache_enabled=True):
                batch_images = var.autoregressive_infer_cfg(
                    B=current_batch_size, 
                    label_B=label_B, 
                    cfg=cfg, 
                    top_k=900, 
                    top_p=0.95, 
                    g_seed=seed + i,  # Different seed for each batch
                    more_smooth=more_smooth
                )
        
        target_images.append(batch_images.cpu())
        remaining_samples -= current_batch_size
        print(f"Generated batch {i//batch_size + 1}, {current_batch_size} images")
    
    target_images = torch.cat(target_images, dim=0)

    print(f"Generating {args.n_samples} images of randomly sampled clean classes")

    # Randomly sample clean classes
    # Sample with replacement
    sampled_clean_classes = random.choices(clean_class, k=args.n_samples)

    # Generate clean images in batches
    remaining_samples = args.n_samples
    clean_images = []
    
    for i in range(0, args.n_samples, batch_size):
        current_batch_size = min(batch_size, remaining_samples)
        batch_classes = sampled_clean_classes[i:i+current_batch_size]
        label_B = torch.tensor(batch_classes, device=device)
        
        with torch.inference_mode():
            with torch.autocast('cuda', enabled=True, dtype=torch.float16, cache_enabled=True):
                batch_images = var.autoregressive_infer_cfg(
                    B=current_batch_size, 
                    label_B=label_B, 
                    cfg=cfg, 
                    top_k=900, 
                    top_p=0.95, 
                    g_seed=seed + 1000 + i,  # Different seed for clean batches
                    more_smooth=more_smooth
                )
        
        clean_images.append(batch_images.cpu())
        remaining_samples -= current_batch_size
        print(f"Generated clean batch {i//batch_size + 1}, {current_batch_size} images")
    
    clean_images = torch.cat(clean_images, dim=0)

    # Create output directory
    clean_save_dir = os.path.join(base_dir, 'clean')
    bd_save_dir = os.path.join(base_dir, 'bd')
    # save each image as jpg
    os.makedirs(clean_save_dir, exist_ok=True)
    os.makedirs(bd_save_dir, exist_ok=True)
    for i in range(args.n_samples):
        clean_image = clean_images[i].cpu().numpy()
        target_image = target_images[i].cpu().numpy()
        
        # Save clean image
        clean_image_path = os.path.join(clean_save_dir, f"clean_{i}.jpg")
        target_image_path = os.path.join(bd_save_dir, f"bd_{i}.jpg")
        
        # Convert to uint8 and save
        clean_image = (clean_image * 255).astype(np.uint8)
        target_image = (target_image * 255).astype(np.uint8)
        
        # Convert from CHW to HWC format if needed
        if clean_image.shape[0] == 3:  # If in CHW format (channels first)
            clean_image = np.transpose(clean_image, (1, 2, 0))
            target_image = np.transpose(target_image, (1, 2, 0))
        
        # Ensure values are in valid range
        clean_image = np.clip(clean_image, 0, 255).astype(np.uint8)
        target_image = np.clip(target_image, 0, 255).astype(np.uint8)
        
        # Save the images
        Image.fromarray(clean_image).save(clean_image_path)
        Image.fromarray(target_image).save(target_image_path)



if __name__ == "__main__":
    main()