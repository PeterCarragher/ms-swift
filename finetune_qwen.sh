conda activate swift-phi
export LD_LIBRARY_PATH=/home/pcarragh/miniconda3/envs/lmms-finetune/lib/python3.10/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH
export CUDA_VISIBLE_DEVICES=0,1
export NPROC_PER_NODE=2
export SIZE_FACTOR=8
export MAX_PIXELS=602112
nohup swift sft \
  --model_type qwen2-vl-7b-instruct \
  --model_id_or_path qwen/Qwen2-VL-7B-Instruct \
  --sft_type lora \
  --dataset data/train.jsonl \
  --val_dataset data/val.jsonl \
  --deepspeed default-zero2 \
  --num_train_epochs 1 --eval_steps 2909 &


CUDA_VISIBLE_DEVICES=0 swift infer \
    --ckpt_dir output/qwen2-vl-7b-instruct/v26-20241110-210452/checkpoint-2909 \
    --load_dataset_config true --merge_lora true

CUDA_VISIBLE_DEVICES=0 swift infer \
    --ckpt_dir output/qwen2-vl-7b-instruct/v26-20241110-210452/checkpoint-2909 \
    --merge_lora true \
    --dataset data/train.jsonl \
    --val_dataset data/val.jsonl