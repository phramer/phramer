FAIRSEQ_PATH=/home/phramer/fairseq
DATA_PATH=/home/phramer/data/gigaword_3e5/data-bin
LM_CHECKPOINT_PATH=/home/whiteRa2bit/checkpoints/gigaword

python $FAIRSEQ_PATH/generate.py $DATA_PATH --path $LM_CHECKPOINT_PATH/checkpoint_best.pt \
     --beam 5 --remove-bpe --skip-invalid-size-inputs-valid-test
