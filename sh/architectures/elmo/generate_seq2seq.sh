FAIRSEQ_PATH=/home/pafakanov/fairseq
DATA_PATH=/data/pafakanov/short_ria/data-bin
CHECKPOINT_PATH=/data/pafakanov/checkpoints/short_ria_seq2seq

python $FAIRSEQ_PATH/generate.py ${DATA_PATH} --path ${CHECKPOINT_PATH}/checkpoint_best.pt --remove-bpe --gen-subset test \
	           --batch-size 300 --min-len 1 --beam 5 --no-repeat-ngram 3 --nbest 1| tee output.txt

grep ^T output.txt | cut -f2- | sed 's/ ##//g' > tgt.txt
grep ^H output.txt | cut -f3- | sed 's/ ##//g' > hypo.txt
cat hypo.txt | sacrebleu tgt.txt

