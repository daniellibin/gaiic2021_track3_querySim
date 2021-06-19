#!/bin/bash
#先根据数据、词频建词表
python build_vocab.py

{
    (cd ./bert-base-count3/pretrain/ && CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 python train_bert.py)
    (cd ./bert-base-count3/finetuning/ && CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 python multi_gpu_QA.py)
    (cd ./bert-base-count3-len100/finetuning/ && CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 python multi_gpu_QA.py)
}&

{
    (cd ./bert-base-count5/pretrain/ && CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 python train_bert.py)
    (cd ./bert-base-count5/finetuning/ && CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 python multi_gpu_QA.py)
    (cd ./bert-base-count5-len32/finetuning/ && CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 python multi_gpu_QA.py)
}&

{
    (cd ./nezha-base-count3/pretrain/ && CUDA_VISIBLE_DEVICES=2 PYTHONUNBUFFERED=1 python train_nezha.py)
    (cd ./nezha-base-count3/finetuning/ && CUDA_VISIBLE_DEVICES=2 PYTHONUNBUFFERED=1 python multi_gpu_QA.py)
}&

{
    (cd ./nezha-base-count5/pretrain/ && CUDA_VISIBLE_DEVICES=3 PYTHONUNBUFFERED=1 python train_nezha.py)
    (cd ./nezha-base-count5/finetuning/ && CUDA_VISIBLE_DEVICES=3 PYTHONUNBUFFERED=1 python multi_gpu_QA.py)
}&

wait # 等待子进程结束

#CUDA_VISIBLE_DEVICES=0 保证推理只使用单卡
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 python serial_main_fusion_thread.py