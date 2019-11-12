DATA_PATH=/data/pafakanov/gigaword/
FAIRSEQ_PATH=/home/pafakanov/fairseq/
DEST_DIR=/data/pafakanov/gigaword/data-bin/

python3 $FAIRSEQ_PATH/preprocess.py --source-lang articles --target-lang summaries \
  --trainpref $DATA_PATH/train.gigaword --validpref $DATA_PATH/valid.gigaword --testpref $DATA_PATH/test.gigaword \
  --destdir $DEST_DIR --workers 70

