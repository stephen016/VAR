import os
import sys
import random

def run(
    model_path,
    n_samples,
    batch_size,
):
    exp_name = f"generate_fid"
    command = (
            f"--model_path {model_path} "
            f"--n_samples {n_samples} "
            f"--batch_size {batch_size} "
        )

    LOG_DIR = "/home/c01xuwa/CISPA-home/VAR/logs"
    full_cmd = (
    f'sbatch -J {exp_name} '
    f'-o {LOG_DIR}/%j_{exp_name}_.out '
    f'--partition=xe8545,gpu,tmp '
    f'--exclude=xe8545-a100-[05,06,12,18,15,30] '
    f'--time=2-00:00:00 '
    f'--gres=gpu:1 --wrap="bash /home/c01xuwa/CISPA-home/VAR/run_docker_eval.sh --rdzv_endpoint 127.0.0.1:{random.randint(49152, 65535)} /home/c01xuwa/CISPA-home/VAR/generate_for_fid.py {command}"'
    #f'--gres=gpu:1 --wrap="bash /home/c01xuwa/CISPA-home/SPA-PT/script_cispa/run_docker.sh --rdzv_endpoint 127.0.0.1:{random.randint(49152, 65535)} /home/c01xuwa/CISPA-home/SPA-PT/scripts/train_prompt.py {command}"'
    )
    #print(full_cmd)
    os.system(full_cmd)


#model_path = "/home/c01xuwa/CISPA-work/c01xuwa/VAR/VAR_alpha_0.05_lr_0.0001_ep_10_p_ratio_0.5_pn_256_depth_16/ar-ckpt-last.pth"
model_path_list = [
    #"/home/c01xuwa/CISPA-work/c01xuwa/VAR/VAR_alpha_0.01_lr_0.0001_ep_10_p_ratio_1_pn_256_depth_30/ar-ckpt-last.pth",
    #"/home/c01xuwa/CISPA-work/c01xuwa/VAR/VAR_alpha_0.01_lr_0.0001_ep_10_p_ratio_1_pn_256_depth_16/ar-ckpt-last.pth",
    #"/home/c01xuwa/CISPA-work/c01xuwa/VAR/VAR_alpha_0.05_lr_0.0001_ep_10_p_ratio_0.5_pn_256_depth_30/ar-ckpt-last.pth"
    "/home/c01xuwa/CISPA-work/c01xuwa/VAR/pretrained_d_30/var_d30.pth",
    "/home/c01xuwa/CISPA-work/c01xuwa/VAR/pretrained_d_16/var_d16.pth",
]
n_samples = 50
batch_size = 8
for model_path in model_path_list:
    run(
        model_path=model_path,
        n_samples=n_samples,
        batch_size=batch_size,
    )