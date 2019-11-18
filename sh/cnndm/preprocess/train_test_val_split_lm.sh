TRAIN_SIZE=250000
TEST_SIZE=30000
DATA_PATH=/data/pafakanov/cnndm_lm/

head -n $TRAIN_SIZE $DATA_PATH/cnndm.articles > $DATA_PATH/cnndm.articles.train
tail -n +${TRAIN_SIZE} $DATA_PATH/cnndm.articles > $DATA_PATH/cnndm.articles.test_valid
head -n $TEST_SIZE $DATA_PATH/cnndm.articles.test_valid > $DATA_PATH/cnndm.articles.test
tail -n +${TEST_SIZE} $DATA_PATH/cnndm.articles.test_valid > $DATA_PATH/cnndm.articles.valid
