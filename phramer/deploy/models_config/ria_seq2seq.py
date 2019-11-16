FAIRSEQ_PATH = '/home/pavel_fakanov/fairseq'
DATA_PATH = '/home/pavel_fakanov/data/ria_seq2seq/data-bin/'
CHECKPOINT_PATH = '/home/pavel_fakanov/checkpoints/ria_seq2seq/checkpoint_best.pt'
LM_CHECKPOINT_PATH = '/home/pavel_fakanov/checkpoints/ria_lm/checkpoints_best.pt'
DATASET_NAME = 'ria'
INPUT_FILE_NAME = '/home/pavel_fakanov/data.txt'
BUFFER_SIZE = 0
BEAM = 5
NBEST = 1
MAX_LEN_A = 0
MAX_LEN_B = 200
MIN_LEN = 1
NO_EARLY_STOP = False
UNNORMALIZED = False
NO_BEAMABLE_MM = False
LENPEN = 1
UNKPEN = 0
NO_REPEAT_NGRAM_SIZE = 0
SAMPLING = False
SAMPLING_TOPK = -1
SAMPLING_TEMPERATURE = 1
DIVERSE_BEAM_GROUPS = 1
DIVERSE_BEAM_STRENGTH = 0.5
PRINT_ALIGNMENT = False
CPU = False
FP16 = False
TASK = 'translation'
LEFT_PAD_SOURCE = True
LEFT_PAD_TARGET = False
FP16_INIT_SCALE = 128
LOG_INTERVAL = 1000
MAX_SOURCE_POSITIONS = 1024
MAX_TARGET_POSITIONS = 1024
MODEL_OVERRIDES = '{}'
NO_PROGRESS_BAR = False
NUM_SHARDS = 1
PREFIX_SIZE = 0
QUIET = False
SCORE_REFERENCE = False
SEED = 1
SHARD_ID = 0
SKIP_INVALID_SIZE_INPUTS_VALID_TEST = False
UPSAMPLE_PRIMARY = 1
CUDA_VISIBLE_DEVICES = [
    1,
]

