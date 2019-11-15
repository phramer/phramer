FAIRSEQ_PATH='/home/pavel_fakanov/fairseq'
DATA_PATH='/home/pavel_fakanov/data/ria_seq2seq/data-bin/'
CHECKPOINT_PATH='/home/pavel_fakanov/checkpoints/ria_seq2seq/checkpoint_best.pt'
LM_CHECKPOINT_PATH='/home/pavel_fakanov/checkpoints/ria_lm/checkpoints_best.pt'
DATASET_NAME='ria'
INPUT_FILE_NAME='data.txt'
REMOVE_BPE=True
MIN_LEN=1
BEAM=5
NO_REPEAT_NGRAM=3
NBEST=1
BUFFER_SIZE=0
CUDA_VISIBLE_DEVICES = [
    1,
]

