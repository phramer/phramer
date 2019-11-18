FAIRSEQ_PATH=/home/pafakanov/fairseq
DATA_DIR=/data/pafakanov/gigaword_lm

python $FAIRSEQ_PATH/preprocess.py --only-source \
	--trainpref $DATA_DIR/gigaword.articles.train \
       	--validpref $DATA_DIR/gigaword.articles.valid \
	--testpref $DATA_DIR/gigaword.articles.test \
	--destdir $DATA_DIR/data-bin/ --workers 50
        
