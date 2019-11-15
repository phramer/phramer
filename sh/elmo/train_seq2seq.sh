FAIRSEQ_PATH=/home/pafakanov/fairseq
LM_DATA=/data/pafakanov/short_ria_lm/data-bin
LM_CHECKPOINT_PATH=/data/pafakanov/checkpoints/short_ria_lm
DATA_PATH=/data/pafakanov/short_ria/data-bin
CHECKPOINT_PATH=/data/pafakanov/checkpoints/short_ria_seq2seq

python ${FAIRSEQ_PATH}/train.py ${DATA_PATH} \
	   --no-enc-token-positional-embeddings --elmo-affine --share-decoder-input-output-embed \
	   --max-update 30000 \
	   --optimizer adam --adam-betas '(0.9, 0.98)' --skip-invalid-size-inputs-valid-test \
           --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates 4000 --lr 0.0005 \
	   --ddp-backend no_c10d --min-lr 1e-09 --clip-norm 0.0 --dropout 0.3 --weight-decay 0.0 \
           --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --update-freq 4 --attention-dropout 0.2 \
           --elmo-dropout 0.2 --max-tokens 1000 --arch transformer_wmt_en_de --seed 1 --warmup-init-lr 1e-7 \
	   --encoder-embed-path elmo:${LM_CHECKPOINT_PATH}/checkpoint_best.pt --source-lang articles --target-lang summaries \
	   --save-interval-updates 300 --keep-interval-updates 5 --save-dir $CHECKPOINT_PATH --comet-logging \

