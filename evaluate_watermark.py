import argparse
import numpy as np
from scipy.stats import binom, t, ttest_ind
from utils.eval_utils import *
import glob
from sklearn.metrics import roc_curve, roc_auc_score
from utils.attack import WatermarkAttacker
import json
import os
from PIL import Image


def get_img(img_path):
    img = Image.open(img_path).convert('RGB')
    img_np = np.array(img).astype(np.float32) / 255.0  # Normalize to [0, 1]
    img_np = np.transpose(img_np, (2, 0, 1))  # Convert to (C, H, W)
    return img_np


def get_score(path,radius_ratio=0.25,attacker=None,generation=False):
    cl_path = os.path.join(path,"clean")
    bd_path = os.path.join(path,"bd")
    cl_imgs = glob.glob(f"{cl_path}/*.jpg") 
    #bd_imgs = glob.glob(f"{path}/backdoored*/*.jpg")
    print(f"processing {len(cl_imgs)} images")
    if attacker is not None:
        # create folders to save images:
        attack_name = attacker.attack_name
        cl_attack_path = os.path.join(path, f"{attack_name}","clean")
        bd_attack_path = os.path.join(path, f"{attack_name}","bd")
        os.makedirs(cl_attack_path, exist_ok=True)
        os.makedirs(bd_attack_path, exist_ok=True)
    else:
        cl_attack_path = cl_path
        bd_attack_path = bd_path
    cl_score = []
    bd_score = []
    score_diff = []
    for cl_img_path in cl_imgs:
    #for bd_img in bd_imgs:
        #cl_img = bd_img.replace("backdoored_","")
        bd_img_path = cl_img_path.replace("clean","bd")
        if not os.path.exists(bd_img_path):
        #if not os.path.exists(cl_img):
            continue
        cl_img = get_img(cl_img_path)
        bd_img = get_img(bd_img_path)
        if attacker is not None:
            cl_img = attacker.apply_attack(cl_img)
            bd_img = attacker.apply_attack(bd_img)
        if generation:
            if attacker is not None:
                # save_img
                img_name = cl_img_path.split("/")[-1]
                img_class = cl_img_path.split("/")[-2]
                if os.path.exists(os.path.join(cl_attack_path, img_class)) == False:
                    os.makedirs(os.path.join(cl_attack_path, img_class), exist_ok=True)
                if os.path.exists(os.path.join(bd_attack_path, img_class)) == False:
                    os.makedirs(os.path.join(bd_attack_path, img_class), exist_ok=True)
                save_cl_img_path = os.path.join(cl_attack_path, img_class,img_name)
                save_bd_img_path = os.path.join(bd_attack_path, img_class,img_name)
                # save img as jpg
                # Define save paths

                # Convert numpy arrays back to PIL images and save
                cl_img_save = np.transpose(cl_img, (1, 2, 0))  # Convert from (C,H,W) to (H,W,C)
                cl_img_save = (cl_img_save * 255).astype(np.uint8)  # Convert from [0,1] to [0,255]
                Image.fromarray(cl_img_save).save(save_cl_img_path)

                bd_img_save = np.transpose(bd_img, (1, 2, 0))  # Convert from (C,H,W) to (H,W,C)
                bd_img_save = (bd_img_save * 255).astype(np.uint8)  # Convert from [0,1] to [0,255]
                Image.fromarray(bd_img_save).save(save_bd_img_path)

        #s_cl = high_freq_ratio(cl_img,radius_ratio=radius_ratio)
        #s_bd = high_freq_ratio(bd_img,radius_ratio = radius_ratio)
        s_cl = mse_score_torch(torch.from_numpy(cl_img).to("cuda"), radius_ratio=radius_ratio)
        s_bd = mse_score_torch(torch.from_numpy(bd_img).to("cuda"), radius_ratio=radius_ratio)
        cl_score.append(s_cl)
        bd_score.append(s_bd)
        score_diff.append(s_cl-s_bd)

    score_cl = np.array(cl_score)
    score_bd = np.array(bd_score)
    score_diff = np.array(score_diff)
    return score_cl, score_bd, score_diff


def parse_args():
    parser = argparse.ArgumentParser(description="Robustness Attack")
    parser.add_argument("--folder_path", type=str, default="/home/c01xuwa/CISPA-work/c01xuwa/FreqWatermark/hypothesis_test", help="Path to the folder containing the images")
    parser.add_argument("--attack_type", type=int, default=0, help="Type of attack to apply (0: no attack, 1: gaussian_attack, 2: color_jitter_attack, 3: geometric_attack, 4: jpeg_compression_attack, 5: vae_sd_attack, 6: infinity_vae_attack, 7: ctrlReg attack, 8: diffprue attack)")
    parser.add_argument("--radius_ratio", type=float, default=0.25, help="Radius ratio for high frequency ratio calculation")
    parser.add_argument("--generation",action="store_true", help="Whether to run generation")
    args = parser.parse_args()
    return args
# 1: gaussian_attack,
# 2: color_jitter_attack,
# 3: geometric_attack,
# 4: jpeg_compression_attack,
# 5: vae_sd_attack,
# 6: infinity_vae_attack,
# 7: ctrlReg attack
# 8: diffprue attack
def main():
    args = parse_args() 

    attack_dict= {
        0: "no_attack",
        1: "gaussian_attack",
        2: "color_jitter_attack",
        3: "geometric_attack",
        4: "jpeg_compression_attack",
        5: "vae_sd_attack",
        6: "infinity_vae_attack",
        7: "ctrlReg_attack",
        8: "diffprue_attack"
    }
    folder_path = args.folder_path
    radius_ratio = args.radius_ratio
        
    attack_type = args.attack_type
    if attack_type == 0:
        attacker = None
    else:
        attacker = WatermarkAttacker(attack_type = attack_type)

    cl_score, wm_score, score_diff = get_score(folder_path,radius_ratio=radius_ratio,attacker=attacker,generation=args.generation)
    labels = np.array([1]*cl_score.shape[0] + [0]*wm_score.shape[0])
    scores = np.concatenate([cl_score, wm_score])
    # Calculate the AUC for reporting
    auc = roc_auc_score(labels, scores)
    print(f"ROC AUC: {auc:.4f}")
    # Find the optimal threshold using Youden's J statistic
    print(cl_score.shape[0],wm_score.shape[0])
    print(cl_score[0:10])
    print(wm_score[0:10])

    fpr, tpr, threshold = roc_curve(labels, scores)
    
    # Find threshold that minimizes distance to perfect classifier at (0,1)
    # This is more robust when TPR and FPR are close to each other
    distance = np.sqrt((1-tpr)**2 + fpr**2)
    optimal_idx = np.argmin(distance)
    optimal_threshold = threshold[optimal_idx]
    print(f"Optimal threshold: {optimal_threshold:.4f}")
    print(f"At threshold {optimal_threshold:.4f}: TPR = {tpr[optimal_idx]:.8f}, FPR = {fpr[optimal_idx]:.8f}")
    
    # calculate the SVR given the optimal threshold
    svr = np.sum(scores[labels == 0] < optimal_threshold) / np.sum(labels == 0)
    print(f"SVR: {svr:.8f}")

    # statistical t-test on cl_score and wm_score
    # random pick 100 from cl_score and 100 from wm_score
    # sample 100 idx from range(len_cl_score))
    idx = np.random.choice(len(cl_score), 100, replace=False)
    cl_score_ = cl_score[idx]
    wm_score_ = wm_score[idx]
    # Perform t-test between clean and watermarked scores

    t_stat, p_value = ttest_ind(cl_score_, wm_score_, equal_var=False, alternative='greater')
    print(f"t-statistic: {t_stat}, p-value: {p_value}")

    # also find the TPR at 1% FPR
    fpr_1 = 0.01
    optimal_idx_1 = np.argmin(np.abs(fpr - fpr_1))
    #print(tpr,fpr,threshold)
    if threshold[optimal_idx_1] == np.inf:
        optimal_idx_1 += 2
    #plot roc curve
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % auc)
    plt.savefig(os.path.join(folder_path, f"roc_curve.png"))

    optimal_threshold_1 = threshold[optimal_idx_1]
    print(f"At threshold {optimal_threshold_1:.4f}: TPR = {tpr[optimal_idx_1]:.8f}, FPR = {fpr[optimal_idx_1]:.8f}")

    # Save the results
    result_dict = {
        "auc": float(auc),
        "optimal_threshold": float(optimal_threshold),
        "tpr": float(tpr[optimal_idx]),
        "fpr": float(fpr[optimal_idx]),
        "TPR@1%FPR": float(tpr[optimal_idx_1]),
        "svr": float(svr),
        "p_value": float(p_value),
    }
    result_path = os.path.join(folder_path, f"result_{attack_dict[attack_type]}_radius_{radius_ratio}.json")
    with open(result_path, "w") as f:
        json.dump(result_dict, f)
        
    print("saved results to", result_path)

if __name__ == "__main__":
    main()