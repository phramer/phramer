TRAIN_SIZE=3000000
TEST_SIZE=400000
DATA_PATH=/data/pafakanov/gigaword_lm/

head -n $TRAIN_SIZE $DATA_PATH/gigaword.articles > $DATA_PATH/gigaword.articles.train
tail -n +${TRAIN_SIZE} $DATA_PATH/gigaword.articles > $DATA_PATH/gigaword.articles.test_valid
head -n $TEST_SIZE $DATA_PATH/gigaword.articles.test_valid > $DATA_PATH/gigaword.articles.test
tail -n +${TEST_SIZE} $DATA_PATH/gigaword.articles.test_valid > $DATA_PATH/gigaword.articles.valid
