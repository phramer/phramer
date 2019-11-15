export DATA_PATH=/data/pafakanov/cnn_dailymail/
export HUG_FACE_PATH=/home/pafakanov/transformers

python $HUG_FACE_PATH/examples/run_summarization.py \
	--output_dir output \
	--per_gpu_train_batch_size 1\
	--model_type bert2bert \
	--do_train True\
	--data_dir=$DATA_PATH \
