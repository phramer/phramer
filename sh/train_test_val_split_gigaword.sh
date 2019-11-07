TRAIN_SIZE=250000
TEST_SIZE=30000
DATA_PATH=/home/phramer/data/gigaword_3e5

head -n $TRAIN_SIZE $DATA_PATH/gigaword.articles > $DATA_PATH/train.gigaword.articles
tail -n +${TRAIN_SIZE} $DATA_PATH/gigaword.articles > $DATA_PATH/test_valid.gigaword.articles
head -n $TEST_SIZE $DATA_PATH/test_valid.gigaword.articles > $DATA_PATH/test.gigaword.articles
tail -n +${TEST_SIZE} $DATA_PATH/test_valid.gigaword.articles > $DATA_PATH/valid.gigaword.articles

head -n $TRAIN_SIZE $DATA_PATH/gigaword.summaries > $DATA_PATH/train.gigaword.summaries
tail -n +${TRAIN_SIZE} $DATA_PATH/gigaword.summaries > $DATA_PATH/test_valid.gigaword.summaries
head -n $TEST_SIZE $DATA_PATH/test_valid.gigaword.summaries > $DATA_PATH/test.gigaword.summaries
tail -n +${TEST_SIZE} $DATA_PATH/test_valid.gigaword.summaries > $DATA_PATH/valid.gigaword.summaries
