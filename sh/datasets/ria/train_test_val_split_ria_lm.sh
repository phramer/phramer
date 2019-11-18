TRAIN_SIZE=350000
TEST_SIZE=25000
DATA_PATH=/data/pafakanov/short_ria_lm/

head -n $TRAIN_SIZE $DATA_PATH/ria.articles > $DATA_PATH/ria.articles.train
tail -n +${TRAIN_SIZE} $DATA_PATH/ria.articles > $DATA_PATH/ria.articles.test_valid
head -n $TEST_SIZE $DATA_PATH/ria.articles.test_valid > $DATA_PATH/ria.articles.test
tail -n +${TEST_SIZE} $DATA_PATH/ria.articles.test_valid > $DATA_PATH/ria.articles.valid

