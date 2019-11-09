TRAIN_SIZE=175000
TEST_SIZE=25000
DATA_PATH=/home/phramer/data/short_ria

head -n $TRAIN_SIZE $DATA_PATH/ria.preprocessed.articles > $DATA_PATH/train.ria.articles
tail -n +${TRAIN_SIZE} $DATA_PATH/ria.preprocessed.articles > $DATA_PATH/test_valid.ria.articles
head -n $TEST_SIZE $DATA_PATH/test_valid.ria.articles > $DATA_PATH/test.ria.articles
tail -n +${TEST_SIZE} $DATA_PATH/test_valid.ria.articles > $DATA_PATH/valid.ria.articles

head -n $TRAIN_SIZE $DATA_PATH/ria.summaries > $DATA_PATH/train.ria.summaries
tail -n +${TRAIN_SIZE} $DATA_PATH/ria.summaries > $DATA_PATH/test_valid.ria.summaries
head -n $TEST_SIZE $DATA_PATH/test_valid.ria.summaries > $DATA_PATH/test.ria.summaries
tail -n +${TEST_SIZE} $DATA_PATH/test_valid.ria.summaries > $DATA_PATH/valid.ria.summaries
