import os
import sys
import random
import glob

def run(
    model_path,
    n_samples,
    batch_size,
):
    exp_name = f"generate_attack"
    command = (
            f"--model_path {model_path} "
            f"--n_samples {n_samples} "
            f"--batch_size {batch_size} "
        )

    LOG_DIR = "/home/c01xuwa/CISPA-home/VAR/logs"
    full_cmd = (
    f'sbatch -J {exp_name} '
    f'-o {LOG_DIR}/%j_{exp_name}_.out '
    f'--partition=xe8545,gpu '
    f'--exclude=xe8545-a100-[05,06,12,18,15,30] '
    f'--time=2-00:00:00 '
    f'--gres=gpu:1 --wrap="bash /home/c01xuwa/CISPA-home/VAR/run_docker_eval.sh --rdzv_endpoint 127.0.0.1:{random.randint(49152, 65535)} /home/c01xuwa/CISPA-home/VAR/generate_for_attack.py {command}"'
    #f'--gres=gpu:1 --wrap="bash /home/c01xuwa/CISPA-home/SPA-PT/script_cispa/run_docker.sh --rdzv_endpoint 127.0.0.1:{random.randint(49152, 65535)} /home/c01xuwa/CISPA-home/SPA-PT/scripts/train_prompt.py {command}"'
    )
    #print(full_cmd)
    os.system(full_cmd)


#model_path = "/home/c01xuwa/CISPA-work/c01xuwa/VAR/VAR_alpha_0.05_lr_0.0001_ep_10_p_ratio_0.5_pn_256_depth_16/ar-ckpt-last.pth"
#model_list = glob.glob(
#    "/home/c01xuwa/CISPA-work/c01xuwa/VAR/*/ar-ckpt-last.pth"
#)
model_list = glob.glob(
    "/home/c01xuwa/CISPA-work/c01xuwa/VAR/*/ar-ckpt-last.pth"
)
model_list = [
    "/home/c01xuwa/CISPA-work/c01xuwa/VAR/pretrained_d_16/var_d16.pth",
    "/home/c01xuwa/CISPA-work/c01xuwa/VAR/pretrained_d_30/var_d30.pth"
]
for model_path in model_list:
    base_dir = "/".join(model_path.split("/")[:-1])
    clean_dir = os.path.join(base_dir, "clean")
    # Check if clean_dir has more than 100 files
    if os.path.exists(clean_dir) and len(os.listdir(clean_dir)) > 210:
        print(f"Skipping {model_path} - already has more than 100 files in {clean_dir}")
        continue
    n_samples = 300
    batch_size = 16
    run(
        model_path=model_path,
        n_samples=n_samples,
        batch_size=batch_size,
    )