FAIRSEQ_PATH=/home/pafakanov/fairseq
DATA_DIR=/data/pafakanov/short_ria_lm

python $FAIRSEQ_PATH/preprocess.py --only-source \
        --trainpref $DATA_DIR/ria.articles.train \
        --validpref $DATA_DIR/ria.articles.valid \
        --testpref $DATA_DIR/ria.articles.test \
        --destdir $DATA_DIR/data-bin/ --workers 50

