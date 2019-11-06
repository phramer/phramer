FAIRSEQ_PATH=/home/phramer/fairseq/
DATA_PATH=/home/phramer/data/small_ria/data-bin/
LM_CHECKPOINT_PATH=/home/whiteRa2bit/checkpoints/small_ria/


python $FAIRSEQ_PATH/train.py $DATA_PATH --lr 0.5 --memory-efficient-fp16 --clip-norm 0.1 \
  --dropout 0.2 --max-tokens 2000 --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
  --max-epoch 5 --lr-scheduler fixed --force-anneal 50 --skip-invalid-size-inputs-valid-test \
  --arch transformer --save-dir $LM_CHECKPOINT_PATH
