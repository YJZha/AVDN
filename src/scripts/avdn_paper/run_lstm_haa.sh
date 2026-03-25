ngpus=1
seed=0
BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
DATASET_DIR="$BASE_DIR/datasets"
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
      "



# train
# CUDA_VISIBLE_DEVICES='4' python xview_lstm/main.py --output_dir $DATASET_DIR/AVDN/lstm_v8 $flag \

# eval
CUDA_VISIBLE_DEVICES='4' python xview_lstm/main.py --output_dir $DATASET_DIR/AVDN/lstm_output $flag \
--resume_file $DATASET_DIR/AVDN/lstm_haa/ckpts/best_val_unseen \
--inference True \
--submit True
