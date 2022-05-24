#!/bin/bash


#SBATCH --gres=gpu:v100:2
#SBATCH --time=120:00:00
#SBATCH --output=slurm.out
#SBATCH --mem-per-cpu=32G
#SBATCH --exclude=gpu[11-17]


module load anaconda
conda init bash
source /home/liux17/.bashrc
conda activate /scratch/work/liux17/ACTOR-master27/env_actor

#srun python -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 --master_addr=127.4.2.4 --master_port=27614 -m src.train.train_cvae --modelname cvae_transformer_rc_rcxyz_kl --pose_rep rot6d --lambda_kl 1e-5 --jointstype vertices --batch_size 200 --num_frames 60 --num_layers 8 --lr 0.0001 --glob --no-vertstrans --dataset uestc --num_epochs 2000 --snapshot 100 --folder exps/uestc

srun --gres=gpu:a100:8 --output=slurm_main.out  python -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 --master_addr=127.4.2.4 --master_port=27614 -m src.train.train_cvae --modelname cvae_transformer_rc_rcxyz_kl --pose_rep rot6d --lambda_kl 1e-5 --jointstype vertices --batch_size 800 --num_frames 60 --num_layers 8 --lr 0.0001 --glob --no-vertstrans --dataset uestc --num_epochs 3000 --snapshot 100 --folder exps_main/uestc

#srun python -m torch.distributed.launch --nproc_per_node=4 --nnodes=1 -m src.train.train_cvae --modelname cvae_transformer_rc_rcxyz_kl_mae --pose_rep rot6d --lambda_kl 1e-5 --jointstype vertices --batch_size 100 --num_frames 60 --num_layers 8 --lr 0.0001 --glob --no-vertstrans --dataset humanact12 --num_epochs 5000 --snapshot 100 --folder exps/humanact12
