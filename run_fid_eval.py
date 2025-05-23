import os
import sys
import random
import glob

def run(
    sample_path
):
    name = sample_path.split("/")[-2]
    exp_name = f"evaluate_fid_{name}"
    base_path = "/home/c01xuwa/CISPA-work/c01xuwa/ImageNet/VIRTUAL_imagenet256_labeled.npz" 
    command = (
            f"{base_path} "
            f"{sample_path} "
        )

    LOG_DIR = "/home/c01xuwa/CISPA-home/VAR/logs"
    full_cmd = (
    f'sbatch -J {exp_name} '
    f'-o {LOG_DIR}/%j_{exp_name}_.out '
    f'--partition=xe8545,tmp '
    f'--exclude=xe8545-a100-[05,06,12,18,14,15,30] '
    f'--time=2-00:00:00 '
    f'--gres=gpu:1 --wrap="bash /home/c01xuwa/CISPA-home/VAR/run_docker_eval.sh --rdzv_endpoint 127.0.0.1:{random.randint(49152, 65535)} /home/c01xuwa/CISPA-home/VAR/evaluator.py {command}"'
    #f'--gres=gpu:1 --wrap="bash /home/c01xuwa/CISPA-home/SPA-PT/script_cispa/run_docker.sh --rdzv_endpoint 127.0.0.1:{random.randint(49152, 65535)} /home/c01xuwa/CISPA-home/SPA-PT/scripts/train_prompt.py {command}"'
    )
    #print(full_cmd)
    os.system(full_cmd)


folder_list = [
    #"/home/c01xuwa/CISPA-work/c01xuwa/VAR/VAR_alpha_0.01_lr_0.0001_ep_10_p_ratio_1_pn_256_depth_30",
    #"/home/c01xuwa/CISPA-work/c01xuwa/VAR/VAR_alpha_0.01_lr_0.0001_ep_10_p_ratio_1_pn_256_depth_16",
    "/home/c01xuwa/CISPA-work/c01xuwa/VAR/VAR_alpha_0.05_lr_0.0001_ep_10_p_ratio_0.5_pn_256_depth_30",
    "/home/c01xuwa/CISPA-work/c01xuwa/VAR/pretrained_d_30",
    #"/home/c01xuwa/CISPA-work/c01xuwa/VAR/pretrained_d_16",
]
for folder in folder_list:
    sample_path = os.path.join(folder, "fid.npz")
    if not os.path.exists(sample_path):
        print(f"Sample path {sample_path} doesn't exists, skipping...")
        continue
    run(
        sample_path
    )

