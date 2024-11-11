conda activate swift-phi
export LD_LIBRARY_PATH=/home/pcarragh/miniconda3/envs/lmms-finetune/lib/python3.10/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH
export CUDA_VISIBLE_DEVICES=0,1
export NPROC_PER_NODE=2
export SIZE_FACTOR=8
export MAX_PIXELS=602112
# export TORCH_DISTRIBUTED_DEBUG=INFO
nohup swift sft \
  --model_type phi3-vision-128k-instruct \
  --sft_type lora \
  --dataset data/train.jsonl \
  --val_dataset data/val.jsonl \
  --deepspeed default-zero2 \
  --ddp_find_unused_parameters true \
  --num_train_epochs 1 --eval_steps 2909 &

CUDA_VISIBLE_DEVICES=0 swift export \
    --ckpt_dir output/phi3-vision-128k-instruct/v16-20241109-111951/checkpoint-4115 \
    --merge_lora true --safe_serialization false

CUDA_VISIBLE_DEVICES=0 swift infer \
    --ckpt_dir output/phi3-vision-128k-instruct/v16-20241109-111951/checkpoint-4115 \
    --load_dataset_config true

CUDA_VISIBLE_DEVICES=0 swift infer \
    --ckpt_dir output/phi3-vision-128k-instruct/v16-20241109-111951/checkpoint-4115 \
    --dataset /home/pcarragh/dev/webqa/ms-swift/data/train.jsonl \
    --val_dataset /home/pcarragh/dev/webqa/ms-swift/data/val.jsonl