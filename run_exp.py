import os
import sys
import random

def run(
  bs=16,
  ep=2,
  alpha=0.001,
  lr = 1e-5,
  data_path=None,
  p_ratio=0.05,
  pn=512,
  depth=16
):
  if depth > 20:
    bs = 1
    partition = "tmp"
  else:
    partition = "xe8545,gpu"

  exp_name = f"VAR_alpha_{alpha}_lr_{lr}_ep_{ep}_p_ratio_{p_ratio}_pn_{pn}_depth_{depth}"
  out_dir = f"/home/c01xuwa/CISPA-work/c01xuwa/VAR/{exp_name}"
  command = f"--nproc_per_node=1 --nnodes=1  train.py \
    --data_path {data_path} \
    --depth={depth} --bs={bs} --ep={ep} --fp16=1 --alng=1e-3 --wpe=0.1 --alpha={alpha} --tblr={lr} \
    --local_out_dir_path={out_dir} --exp_name={exp_name} --pn={pn} "

  LOG_DIR = "/home/c01xuwa/CISPA-home/VAR/logs"
  full_cmd = (
      f'sbatch -J {exp_name} '
      f'-o {LOG_DIR}/%j_{exp_name}_.out '
      f'--partition={partition} '
      f'--exclude=xe8545-a100-[06,18] '
      f'--time=2-00:00:00 '
      f'--gres=gpu:1 --wrap="bash /home/c01xuwa/CISPA-home/VAR/run_docker.sh --rdzv_endpoint 127.0.0.1:{random.randint(49152, 65535)} {command}"'
      #f'--gres=gpu:1 --wrap="bash /home/c01xuwa/CISPA-home/SPA-PT/script_cispa/run_docker.sh --rdzv_endpoint 127.0.0.1:{random.randint(49152, 65535)} /home/c01xuwa/CISPA-home/SPA-PT/scripts/train_prompt.py {command}"'
  )
  #print(full_cmd)
  os.system(full_cmd)
bs = 16
ep= 5
alpha=0.01
lr = 3e-3
p_ratio = 0.5

pn_dep_tuple = [
    #(512, 36),
    #(256, 30),
    (256,16)
]
#for ep in [10,50]:
for ep in [50]:
  for alpha in [0.001,0.01,0.05]:
  #for alpha in [10]:
    for p_ratio in [1,0.5,0.1]:
    #for p_ratio in [1,0.5]:
    #for p_ratio in [1]:
      data_path = f"/home/c01xuwa/CISPA-work/c01xuwa/ImageNet/freqset_{p_ratio}"
      for pn, dep in pn_dep_tuple:
        run(bs,
            ep,
            alpha,
            lr,
            data_path,
            p_ratio,
            pn,
            dep)