ngpus=1
seed=0
BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
DATASET_DIR="$BASE_DIR/datasets"
BERT_DIR="$DATASET_DIR/bert-base-uncased"
QWEN_DIR="$DATASET_DIR/Qwen2.5-VL-7B"
LOCAL_FILES_ONLY=""
TOKENIZER_ARGS=""
if [ -d "$BERT_DIR" ]; then
      LOCAL_FILES_ONLY="--local_files_only"
      TOKENIZER_ARGS="--tokenizer_path $BERT_DIR --bert_model_path $BERT_DIR"
else
      echo "❌ 缺少本地Tokenizer目录: $BERT_DIR"
      echo "请将bert-base-uncased下载到该目录，或修改脚本中的BERT_DIR。"
      exit 1
fi

QWEN_ARGS=""
if [ -d "$QWEN_DIR" ]; then
      QWEN_ARGS="--use_qwen_vl --qwen_vl_model $QWEN_DIR --proj_dim 768 --qwen_dtype float16 --qwen_et_frame_tokens 512"
else
      echo "❌ 缺少Qwen模型目录: $QWEN_DIR"
      echo "请将Qwen2.5-VL-7B下载到该目录，或修改脚本中的QWEN_DIR。"
      exit 1
fi

flag="--root_dir $DATASET_DIR

      --world_size ${ngpus}
      --seed ${seed}
      

      --feedback student

      --max_action_len 10
      --max_instr_len 100

      --lr 1e-5
      --iters 200000
      --log_every 2
      --batch_size 4
      --optim adamW

      --ml_weight 0.2      

      --feat_dropout 0.4
      --dropout 0.5
      
      --nss_w 0
      --nss_r 0

      --disable_haa

      --darknet_model_file $DATASET_DIR/AVDN/pretrain_weights/yolo_v3.cfg
      --darknet_weight_file $DATASET_DIR/AVDN/pretrain_weights/best.pt
      --eval_first True
      $LOCAL_FILES_ONLY
      $TOKENIZER_ARGS
      $QWEN_ARGS
      "



# train
#CUDA_VISIBLE_DEVICES='2'  python xview_et/main.py --output_dir $DATASET_DIR/AVDN/et_qwen_1 $flag 

# eval
#CUDA_VISIBLE_DEVICES='1'  python xview_et/main.py --output_dir $DATASET_DIR/AVDN/et_qwen_1 $flag \
#      --resume_file $DATASET_DIR/AVDN/et_qwen_1/ckpts/latest_dict_4588\
#      --inference True \
#      --submit True
# resume
CUDA_VISIBLE_DEVICES='2'  python xview_et/main.py --output_dir $DATASET_DIR/AVDN/et_qwen_1 $flag \
      --resume_file $DATASET_DIR/AVDN/et_qwen_1/ckpts/latest_dict_22940
