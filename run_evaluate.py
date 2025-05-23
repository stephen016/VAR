import os
import sys
import random
import glob

def run(
    folder_path,
    attack_type,
    radius_ratio,
):
    exp_name = f"evaluate_{attack_type}_r_{radius_ratio}"
    command = (
        f"--folder_path {folder_path} "
        f"--attack_type {attack_type} "
        f"--radius_ratio {radius_ratio} "
        )

    LOG_DIR = "/home/c01xuwa/CISPA-home/VAR/logs"
    full_cmd = (
    f'sbatch -J {exp_name} '
    f'-o {LOG_DIR}/%j_{exp_name}_.out '
    f'--partition=xe8545,tmp '
    f'--exclude=xe8545-a100-[05,06,12,18,14,15,30] '
    f'--time=2-00:00:00 '
    f'--gres=gpu:1 --wrap="bash /home/c01xuwa/CISPA-home/VAR/run_docker_eval.sh --rdzv_endpoint 127.0.0.1:{random.randint(49152, 65535)} /home/c01xuwa/CISPA-home/VAR/evaluate_watermark.py {command}"'
    #f'--gres=gpu:1 --wrap="bash /home/c01xuwa/CISPA-home/SPA-PT/script_cispa/run_docker.sh --rdzv_endpoint 127.0.0.1:{random.randint(49152, 65535)} /home/c01xuwa/CISPA-home/SPA-PT/scripts/train_prompt.py {command}"'
    )
    #print(full_cmd)
    os.system(full_cmd)

base_folder = "/home/c01xuwa/CISPA-work/c01xuwa/VAR"
folders = glob.glob(f"{base_folder}/*")
#folder_path = "/home/c01xuwa/CISPA-work/c01xuwa/VAR/VAR_alpha_0.05_lr_0.0001_ep_10_p_ratio_0.5_pn_256_depth_16"
for folder_path in folders:
    res_path = os.path.join(folder_path, "result_no_attack_radius_0.25.json")
    #if os.path.exists(res_path):
    #    print(f"Skip {folder_path} because result already exists.")
    #    continue
    # check if there is a .pth model under folder_path
    model_files = glob.glob(os.path.join(folder_path, "*.pth"))
    if not model_files:
        print(f"Skip {folder_path} because no model file (.pth) found.")
        continue
    model_file = model_files[0]
    if "last" in model_file:
        continue
    attack_type = 0
    radius_ratio = 0.25
    run(
        folder_path=folder_path,
        attack_type=attack_type,
        radius_ratio=radius_ratio,
    )
