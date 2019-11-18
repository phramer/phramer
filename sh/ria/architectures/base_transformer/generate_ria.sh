FAIRSEQ_PATH=/home/phramer/fairseq
DATA_PATH=/home/phramer/data/short_ria/data-bin
LM_CHECKPOINT_PATH=/home/whiteRa2bit/checkpoints/short_ria

python $FAIRSEQ_PATH/generate.py $DATA_PATH --path $LM_CHECKPOINT_PATH/checkpoint_best.pt \
     --beam 2 --remove-bpe --skip-invalid-size-inputs-valid-test

