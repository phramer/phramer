FAIRSEQ_PATH=/home/pafakanov/fairseq
LM_DATA=/data/pafakanov/gigaword_lm/data-bin
LM_CHECKPOINT_PATH=/data/pafakanov/checkpoints/gigaword_lm


python $FAIRSEQ_PATH/train.py ${LM_DATA} -a bi_transformer_lm_big --clip-norm 0.1 --lr 0.0001 --dropout 0.1 \
  --max-tokens 500 --no-progress-bar --log-interval 1 --criterion cross_entropy --fp16 \
  --optimizer nag --lr-scheduler cosine --warmup-init-lr 1e-07 --warmup-updates 16000 --min-lr 1e-09 \
  --distributed-world-size 6 --max-update 984000 --lr-period-updates 968000 --lr-shrink 1.0 --decoder-layers 12 --attention-dropout 0.1 \
  --max-lr 1.0 --decoder-embed-dim 512 --ddp-backend no_c10d --sample-break-mode eos --skip-invalid-size-inputs-valid-test \
  --relu-dropout 0.05 --save-interval-updates 10000 --keep-interval-updates 10 \
  --save-dir ${LM_CHECKPOINT_PATH} --task language_modeling
