import random
import torch
import torch.nn.functional as F
import wandb
import argparse
import numpy as np
import os
from PIL import Image
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
from tqdm import tqdm
import glob



def load_images_from_directory(directory, max_images=None):
    images = []
    image_files = glob.glob(f"{directory}/*/*.jpg")
    #image_files = [f for f in os.listdir(directory) if f.endswith('.jpg')]
    image_files = sorted(image_files)[:max_images] if max_images else image_files
    
    for file in image_files:
        img_path = os.path.join(directory, file)
        img = Image.open(img_path).convert('RGB')
        img_np = np.array(img).astype(np.float32) / 255.0  # Normalize to [0, 1]
        img_np = np.transpose(img_np, (2, 0, 1))  # Convert to (C, H, W)
        images.append(img_np)
    return images


def plot_average_spectrum(clean_images, bd_images, save_path=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    def avg_spectrum(images):
        mags = []
        for img in images:
            if not isinstance(img, torch.Tensor):
                img = torch.from_numpy(img)
            img = img.float().to(device)

            # Compute FFT of original image
            freq = torch.fft.fft2(img)  # shape: (D, H, W)
            freq_shifted = torch.fft.fftshift(freq, dim=(-2, -1))
            magnitude = torch.abs(freq_shifted)
            mags.append(magnitude.detach().cpu().numpy())

        avg_mag = np.mean(np.stack(mags, axis=0), axis=(0,1))  # shape: (H, W)
        print(avg_mag.shape)
        return avg_mag



    clean_avg = avg_spectrum(clean_images)
    bd_avg = avg_spectrum(bd_images)

    # Plot one channel (e.g., red) for clarity
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(np.log1p(clean_avg), cmap='viridis')
    plt.title("Clean (log spectrum)")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(np.log1p(bd_avg), cmap='viridis')
    plt.title("BD (log spectrum)")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def high_freq_ratio(image, radius_ratio=0.25):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    c, h, w = image.shape
    if not isinstance(image, torch.Tensor):
        image = torch.from_numpy(image)
    image = image.float().to(device)

    # Compute FFT of original image
    freq = torch.fft.fft2(image)  # shape: (3, H, W)
    freq_shifted = torch.fft.fftshift(freq, dim=(-2, -1))
    magnitude = torch.abs(freq_shifted).detach().cpu().numpy()

    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    center_y, center_x = h // 2, w // 2
    radius = radius_ratio * min(h, w)
    dist = np.sqrt((yy - center_y) ** 2 + (xx - center_x) ** 2)
    mask_low = (dist <= radius).astype(np.float32)
    mask_high = 1.0 - mask_low  # complement

    mask_low = np.stack([mask_low] * c, axis=0)
    mask_high = np.stack([mask_high] * c, axis=0)

    energy_total = np.sum(magnitude ** 2)
    energy_high = np.sum((magnitude ** 2) * mask_high)

    return energy_high / energy_total  # normalized ratio


def plot_hf_ratio_histogram(clean_images, bd_images, radius_ratio=0.25, save_path="plots/hf_ratio_hist.pdf"):
    clean_scores = [high_freq_ratio(img, radius_ratio) for img in clean_images]
    bd_scores = [high_freq_ratio(img, radius_ratio) for img in bd_images]

    plt.figure(figsize=(6, 4))
    plt.hist(clean_scores, bins=30, alpha=0.6, label="Clean", color="green", density=True)
    plt.hist(bd_scores, bins=30, alpha=0.6, label="BD", color="red", density=True)
    plt.xlabel("High-Frequency Energy Ratio")
    plt.ylabel("Density")
    plt.title("Distribution of HF Energy Ratios")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def mse_score_torch(img: torch.Tensor, radius_ratio: float = 0.25) -> float:
    device = img.device
    C, H, W = img.shape
    img = img.double()
    # Compute FFT of original image
    orig_freq = torch.fft.fft2(img)  # shape: (3, H, W)
    orig_freq_shifted = torch.fft.fftshift(orig_freq, dim=(-2, -1))

    # Create low-pass mask (1s in center, 0s outside radius)
    yy, xx = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing="ij")
    center_y, center_x = H // 2, W // 2
    radius = radius_ratio * min(H, W)
    dist = torch.sqrt((yy - center_y) ** 2 + (xx - center_x) ** 2)
    mask = (dist <= radius).float()  # shape: (H, W)
    mask = mask.unsqueeze(0).unsqueeze(0) # (1, 1, H, W)

    # Apply LPF in frequency space
    filt_freq_shifted = orig_freq_shifted * mask

    # Inverse shift + iFFT to get filtered image
    filt_freq = torch.fft.ifftshift(filt_freq_shifted, dim=(-2, -1))
    filtered_img = torch.abs(torch.fft.ifft2(filt_freq))#.real  # keep real part

    # Compute frequency domain representations again
    filt_freq_final = torch.fft.fft2(filtered_img)
    orig_mag = orig_freq.real
    orig_mag = torch.abs(orig_freq)
    filt_mag = filt_freq_final.real
    filt_mag = torch.abs(filt_freq_final)

    # MSE loss
    mse = F.mse_loss(torch.log1p(orig_mag), torch.log1p(filt_mag), reduction='mean')


    return mse.item()




def compute_threshold(clean_images, bd_images, radius_ratio=0.25, plot_path="plots/roc_curve_MSE.pdf"):

    print(f"Computing LPF-based self-MSE with radius_ratio={radius_ratio}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def mse_lpf_difference(img):
        if not isinstance(img, torch.Tensor):
            img = torch.from_numpy(img)
        img = img.float().to(device)
        print(img.shape)
        return mse_score_torch(img, radius_ratio=radius_ratio)

    clean_mse = [mse_lpf_difference(img) for img in tqdm(clean_images, desc="Clean")]
    bd_mse = [mse_lpf_difference(img) for img in tqdm(bd_images, desc="BD")]

    scores = np.array(clean_mse + bd_mse)
    labels = np.array([1] * len(clean_mse) + [0] * len(bd_mse))

    fpr, tpr, thresholds = roc_curve(labels, scores)
    auc = roc_auc_score(labels, scores)

    distances = np.sqrt(fpr**2 + (1 - tpr)**2)
    best_idx = np.argmin(distances)
    best_threshold = thresholds[best_idx]

    print(f"Best threshold: {best_threshold:.4f} (AUC = {auc:.6f})")
    print(f"Saving ROC curve to: {plot_path}")

    plt.figure()
    plt.plot(fpr, tpr, label=f'AUC = {auc:.2f}')
    plt.scatter(fpr[best_idx], tpr[best_idx], color='red', label=f'Threshold = {best_threshold:.4f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()

    return best_threshold




def evaluate_with_threshold(clean_images, bd_images, threshold, radius_ratio=0.25):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def mse_score(img):
        if not isinstance(img, torch.Tensor):
            img = torch.from_numpy(img)
        img = img.float().to(device)
        return mse_score_torch(img, radius_ratio=radius_ratio)
    
    scoring_fn = mse_score
    label_clean = 1
    label_bd = 0

    clean_scores = [scoring_fn(img) for img in clean_images]
    bd_scores = [scoring_fn(img) for img in bd_images]

    # Predict: label 1 if score >= threshold, else 0
    clean_preds = [1 if s >= threshold else 0 for s in clean_scores]
    bd_preds = [1 if s >= threshold else 0 for s in bd_scores]

    # Inference depends on label mapping
    tn = sum(p == label_clean for p in clean_preds)
    fn = sum(p != label_clean for p in clean_preds)
    tp = sum(p == label_bd for p in bd_preds)
    fp = sum(p != label_bd for p in bd_preds)

    total = len(clean_preds) + len(bd_preds)
    correct = tp + tn
    accuracy = correct / total

    return {
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "Accuracy": accuracy
    }


#arser = argparse.ArgumentParser(description='ASR EVALUATION')


# parser.add_argument('--threshold_mse', type=float, default=None, 
#                     help='mse loss threshold')
# parser.add_argument('--threshold_hf', type=float, default=None, 
#                     help='high frequency threshold')
# parser.add_argument('--num_cal_imgs', type=int, default=250, 
#                     help='number of images to calibrate the threshold')
# parser.add_argument('--num_eval_imgs', type=int, default=250, 
#                     help='number of images to calibrate the threshold')
# parser.add_argument('--image_path', type=str, default=None, 
#                     help='model name')
# parser.add_argument('--model_name', type=str, default=None, 
#                     help='model name')
# parser.add_argument('--plot_init_figures', type=bool, default=False, 
#                     help='model name')
# parser.add_argument('--compare_to_pretrained', type=bool, default=False, 
#                     help='model name')

# args = parser.parse_args()

# new_clean_images = []

# if args.compare_to_pretrained:
#     print("loading clean images")
#     pretrained_path = "/home/c01viha/CISPA-work/c01viha/infinity/eval/hpsv21_pretrained/"
#     clean_images = load_images_from_directory(f'{pretrained_path}/images/backdoored_anime', max_images=800)
#     clean_images_cal = clean_images[:args.num_cal_imgs]

#     new_clean_images.extend(clean_images[args.num_cal_imgs:(args.num_cal_imgs+args.num_eval_imgs)])

#     new_clean_images.extend(load_images_from_directory(f'{pretrained_path}/images/backdoored_concept-art', max_images=args.num_eval_imgs))
#     new_clean_images.extend(load_images_from_directory(f'{pretrained_path}/images/backdoored_paintings', max_images=args.num_eval_imgs))
#     new_clean_images.extend(load_images_from_directory(f'{pretrained_path}/images/backdoored_photo', max_images=args.num_eval_imgs))

# else:

#     print("loading clean images")
#     clean_images = load_images_from_directory(f'{args.image_path}/images/anime', max_images=800)
#     clean_images_cal = clean_images[:args.num_cal_imgs]


#     new_clean_images.extend(load_images_from_directory(f'{args.image_path}/images/concept-art', max_images=args.num_eval_imgs))
#     new_clean_images.extend(load_images_from_directory(f'{args.image_path}/images/paintings', max_images=args.num_eval_imgs))
#     new_clean_images.extend(clean_images[args.num_cal_imgs:(args.num_cal_imgs+args.num_eval_imgs)])
#     new_clean_images.extend(load_images_from_directory(f'{args.image_path}/images/photo', max_images=args.num_eval_imgs))




# print("loading BD images")
# bd_images = load_images_from_directory(f'{args.image_path}/images/backdoored_anime', max_images=800)


# bd_images_cal = bd_images[:args.num_cal_imgs]


# if args.plot_init_figures:

#     print("Computing average spectrum")
#     plot_average_spectrum(clean_images, bd_images, save_path=f"plots/{args.model_name}_avg_freq_spectrum.pdf")
#     print("done")
#     print("Compute average ratio Histogram")
#     plot_hf_ratio_histogram(clean_images, bd_images, save_path=f"plots/{args.model_name}_hf_energy_dist.pdf")
#     print("done")




# print("Load set of evaluation images")

# new_bd_images = []


# new_bd_images.extend(bd_images[args.num_cal_imgs:(args.num_cal_imgs+args.num_eval_imgs)])
# new_bd_images.extend(load_images_from_directory(f'{args.image_path}/images/backdoored_concept-art', max_images=args.num_eval_imgs))
# new_bd_images.extend(load_images_from_directory(f'{args.image_path}/images/backdoored_paintings', max_images=args.num_eval_imgs))
# new_bd_images.extend(load_images_from_directory(f'{args.image_path}/images/backdoored_photo', max_images=args.num_eval_imgs))
# print("loaded new images")



# if not args.threshold_mse:

#     # Compute threshold in frequency space
#     print("computing the thresholds")
#     threshold_mse = compute_threshold(clean_images_cal, bd_images_cal, radius_ratio=0.1, plot_path=f"plots/{args.model_name}_roc_curve_MSE.pdf")
#     print("done")

# else:
#     print(f"Setting mse threshold from args: \nMSE threshold = {args.threshold_mse}")
#     threshold_mse = args.threshold_mse

# results = evaluate_with_threshold(new_clean_images, new_bd_images, threshold_mse, radius_ratio=0.1)
# print(f"Results based on MSE loss threshold of {threshold_mse}: \n {results}")



# # if not args.threshold_hf:
# #     print("computing the thresholds with HF thing")
# #     threshold_hf = compute_hf_threshold(clean_images, bd_images, radius_ratio=0.25, plot_path=f"plots/{args.model_name}_roc_curve_HF.pdf")
# #     print("done")

# # else:
# #     print(f"Setting high frequency threshold from args: \nHigh frequency threshold = {args.threshold_hf}")
# #     threshold_hf = args.threshold_hf


# # results = evaluate_with_threshold(new_clean_images, new_bd_images, threshold_hf, radius_ratio=0.25, method="hf_ratio")
# # print(f"Results based on high frequency eval threshold of {threshold_hf}: \n {results}")