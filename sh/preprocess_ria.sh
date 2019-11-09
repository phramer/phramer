DATA_PATH=/home/phramer/data/short_ria
FAIRSEQ_PATH=/home/phramer/fairseq/
DEST_DIR=/home/phramer/data/short_ria/data-bin/

python3 $FAIRSEQ_PATH/preprocess.py --source-lang articles --target-lang summaries \
  --trainpref $DATA_PATH/train.ria --validpref $DATA_PATH/valid.ria --testpref $DATA_PATH/test.ria \
  --destdir $DEST_DIR --workers 70

