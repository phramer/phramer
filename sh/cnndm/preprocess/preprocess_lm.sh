FAIRSEQ_PATH=/home/pafakanov/fairseq
DATA_DIR=/data/pafakanov/cnndm_lm

python $FAIRSEQ_PATH/preprocess.py --only-source \
	--trainpref $DATA_DIR/cnndm.articles.train\
       	--validpref $DATA_DIR/cnndm.articles.valid \
	--testpref $DATA_DIR/cnndm.articles.test \
	--destdir $DATA_DIR/data-bin/ --workers 50
        
