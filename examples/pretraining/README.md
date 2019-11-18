# Pre-training for abstractive summarization

 ***In order to use all the code provided here please run the following command:***\
```pip install git+git@github.com:phramer/fairseq.git@phramer_bi_trans_lm```



### Pre-training language model:
To train the language model we used RIA dataset.
The first step is to preprocess our data. To preprocess the data:
```
fairseq-preprocess --only-source \
        --trainpref ${DATA_DIR}/ria.articles.train \
        --validpref ${DATA_DIR}/ria.articles.valid \
        --testpref ${DATA_DIR}/ria.articles.test \
        --destdir ${DATA_DIR}/data-bin \
        --workers 50 \
```

The second step is to train a language model. ${LM_DATA} has to point to a folder with processed monolingual dataset (${DATA_DIR}/data-bin). To train the language model:
```
fairseq-train ${LM_DATA} -a bi_transformer_lm_big --clip-norm 0.1 --lr 0.0001 --dropout 0.1 \
           --max-tokens 750 --no-progress-bar --log-interval 1 --criterion cross_entropy --fp16 \
           --optimizer nag --lr-scheduler cosine --warmup-init-lr 1e-07 --warmup-updates 16000 --min-lr 1e-09 \
           --distributed-world-size 6 --max-update 984000 --lr-period-updates 968000 --lr-shrink 1.0 --decoder-layers 12 \
           --attention-dropout 0.1 --max-lr 1.0 --decoder-embed-dim 512 --ddp-backend no_c10d --sample-break-mode eos \
           --skip-invalid-size-inputs-valid-test --relu-dropout 0.05 --save-interval-updates 10000 \
           --keep-interval-updates 10 --save-dir ${LM_CHECKPOINT_PATH} --task language_modeling --comet-logging \
```

### Training a final seq2seq model:
To preprocess the data:
```
fairseq-preprocess -source-lang articles \
           --target-lang summaries \
           --trainpref ${DATA_PATH}/train.ria \
           --validpref ${DATA_PATH}/valid.ria \
           --testpref ${DATA_PATH}/test.ria \
           --destdir ${DEST_DIR} \
           --workers 70
```

To train a final seq2seq model:
```
fairseq-train  ${DATA_PATH} \
           --no-enc-token-positional-embeddings --elmo-affine --share-decoder-input-output-embed \
           --max-update 30000 --optimizer adam --adam-betas '(0.9, 0.98)' --skip-invalid-size-inputs-valid-test \
           --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates 4000 --lr 0.0005 \
           --ddp-backend no_c10d --min-lr 1e-09 --clip-norm 0.0 --dropout 0.3 --weight-decay 0.0 \
           --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --update-freq 4 --attention-dropout 0.2 \
           --elmo-dropout 0.2 --max-tokens 1000 --arch transformer_wmt_en_de --seed 1 --warmup-init-lr 1e-7 \
           --encoder-embed-path elmo:${LM_CHECKPOINT_PATH}/checkpoint_best.pt --source-lang articles \
           --target-lang summaries --save-interval-updates 300 --keep-interval-updates 5 --save-dir ${CHECKPOINT_PATH} \
           --comet-logging \
```


### To generate using the pre-trained model

```
fairseq-generate ${DATA_PATH} --path ${CHECKPOINT_PATH}/checkpoint_best.pt --remove-bpe --gen-subset test \
                   --batch-size 300 --min-len 1 --beam 5 --no-repeat-ngram 3 --nbest 1| tee output.txt

grep ^T output.txt | cut -f2- | sed 's/ ##//g' > tgt.txt
grep ^H output.txt | cut -f3- | sed 's/ ##//g' > hypo.txt
```
