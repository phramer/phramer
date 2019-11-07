DATA_PATH=/home/phramer/data/gigaword_3e5
FAIRSEQ_PATH=/home/phramer/fairseq/
DEST_DIR=/home/phramer/data/gigaword_3e5/data-bin/

python3 $FAIRSEQ_PATH/preprocess.py --source-lang articles --target-lang summaries \
  --trainpref $DATA_PATH/train.gigaword --validpref $DATA_PATH/valid.gigaword --testpref $DATA_PATH/test.gigaword \
  --destdir $DEST_DIR --workers 70 --bpe bert --dataset-impl lazy

