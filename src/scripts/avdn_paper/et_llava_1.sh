#!/bin/bash
# ====== ET-LLaVA Adapter 单卡训练/评估脚本 ======

export CUDA_VISIBLE_DEVICES=1

# 基础配置
ngpus=1
seed=0
BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
DATASET_DIR="$BASE_DIR/datasets"
OUT_DIR="$DATASET_DIR/AVDN/et_llava_multi_glimpse"
LLAVA_DIR="$DATASET_DIR/llava"
TOKENIZER_DIR="$DATASET_DIR/bert-base-uncased"
LOCAL_FILES_ONLY=""
TOKENIZER_ARGS=""
if [ -d "$TOKENIZER_DIR" ]; then
    LOCAL_FILES_ONLY="--local_files_only"
    TOKENIZER_ARGS="--tokenizer_path $TOKENIZER_DIR --bert_model_path $TOKENIZER_DIR"
else
    echo "❌ 缺少本地Tokenizer目录: $TOKENIZER_DIR"
    echo "请将bert-base-uncased下载到该目录，或修改脚本中的TOKENIZER_DIR。"
    exit 1
fi

# 检查点路径
RESUME_CHECKPOINT="$OUT_DIR/train_run/ckpts/latest_dict_57350.pth"

# 检查文件是否存在
if [ ! -f "$RESUME_CHECKPOINT" ]; then
    echo "❌ 错误: 检查点不存在: $RESUME_CHECKPOINT"
    echo "请先运行 'bash $0 info' 查看可用检查点"
    exit 1
fi

flag="--root_dir $DATASET_DIR \
      --world_size ${ngpus} \
      --seed ${seed} \
      --feedback student \
      --max_action_len 10 \
      --max_instr_len 100 \
      --lr_et 1e-4 \
      --lr_adapter 5e-5 \
      --iters 500 \
      --log_every 50 \
      --batch_size 4 \
      --optim adamW \
      --ml_weight 0.2 \
      --feat_dropout 0.4 \
      --dropout 0.5 \
      --nss_w 0 \
      --nss_r 0 \
    --darknet_model_file $DATASET_DIR/AVDN/pretrain_weights/yolo_v3.cfg \
    --darknet_weight_file $DATASET_DIR/AVDN/pretrain_weights/best.pt \
      --pred_dir $OUT_DIR/preds \
      --use_llava True \
    --llava_dir $LLAVA_DIR \
    $LOCAL_FILES_ONLY \
    $TOKENIZER_ARGS"

# ========== 继续训练 ==========
if [ "$1" = "resume" ]; then
    echo ">>> 从检查点恢复训练: $(basename $RESUME_CHECKPOINT)"
    
    # 关键修改：去掉不存在的参数，只保留resume_file
    python xview_et/main.py \
        --output_dir $OUT_DIR/train_run \
        $flag \
        --eval_first False \
        --resume_file "$RESUME_CHECKPOINT" \
        2>&1 | tee -a $OUT_DIR/train_resume_$(date +%Y%m%d_%H%M%S).log

# ========== 全新训练 ==========
elif [ "$1" = "train" ]; then
    echo ">>> 开始全新训练 ET-LLaVA Adapter 模型"
    mkdir -p $OUT_DIR/train_run
    python xview_et/main.py \
        --output_dir $OUT_DIR/train_run \
        $flag \
        2>&1 | tee $OUT_DIR/train.log

# ========== 评估 ==========
elif [ "$1" = "eval" ]; then
    echo ">>> 开始评估 ET-LLaVA Adapter 模型"
    
    # 可以选择评估哪个检查点
    EVAL_CHECKPOINT="${2:-$OUT_DIR/train_run/ckpts/latest_dict_57350.pth}"
    echo "使用检查点: $EVAL_CHECKPOINT"
    
    mkdir -p $OUT_DIR/eval_run
    python xview_et/main.py \
        --output_dir $OUT_DIR/eval_run \
        $flag \
        --resume_file "$EVAL_CHECKPOINT" \
        --inference True \
        --submit True \
        2>&1 | tee $OUT_DIR/eval.log

else
    echo "用法:"
    echo "  bash $0 resume     # 从检查点恢复训练"
    echo "  bash $0 train      # 从头开始训练"
    echo "  bash $0 eval       # 评估模型"
    exit 1
fi