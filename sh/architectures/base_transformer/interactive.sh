FAIRSEQ_PATH=/home/phramer/fairseq
DATA_PATH=/home/phramer/data/extra_small_ria/data-bin
LM_CHECKPOINT_PATH=/home/whiteRa2bit/checkpoints/extra_small_ria

python $FAIRSEQ_PATH/interactive.py $DATA_PATH --path $LM_CHECKPOINT_PATH/checkpoint_best.pt \
     --beam 5 --remove-bpe --skip-invalid-size-inputs-valid-test
