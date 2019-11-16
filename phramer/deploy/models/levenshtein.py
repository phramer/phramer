from collections import namedtuple
import fileinput
from types import SimpleNamespace

import torch

from fairseq import checkpoint_utils, options, tasks, utils
from fairseq.data import encoders
from phramer.data.dataset import RIANewsDataset
from phramer.deploy.deploy_config import CONFIG_NAME


if CONFIG_NAME == 'ria_levenshtein':
    from phramer.deploy.models_config.ria_levenshtein import *

else:
    print("ERROR, PLEASE SPECIFY THE CORRECT CONFIG NAME")


Batch = namedtuple('Batch', 'ids src_tokens src_lengths')
Translation = namedtuple('Translation', 'src_str hypos pos_scores alignments')


def process_article(args, data_path, save_path):
    with open(data_path, 'r+') as f:
        article = f.read()

    article = article.lower()
    article = article.replace('\n', ' ')

    if args.dataset_name == 'ria':
        ria = RIANewsDataset()
        article = ria._process_article(article)

    f = open(save_path, 'w+')
    f.write(article)
    f.close()


def buffered_read(input, buffer_size):
    buffer = []
    with fileinput.input(files=[input], openhook=fileinput.hook_encoded("utf-8")) as h:
        for src_str in h:
            buffer.append(src_str.strip())
            if len(buffer) >= buffer_size:
                yield buffer
                buffer = []

    if len(buffer) > 0:
        yield buffer


def make_batches(lines, args, task, max_positions, encode_fn):
    tokens = [
        task.source_dictionary.encode_line(
            encode_fn(src_str), add_if_not_exist=False
        ).long()
        for src_str in lines
    ]
    lengths = torch.LongTensor([t.numel() for t in tokens])
    itr = task.get_batch_iterator(
        dataset=task.build_dataset_for_inference(tokens, lengths),
        max_tokens=args.max_tokens,
        max_sentences=args.max_sentences,
        max_positions=max_positions,
    ).next_epoch_itr(shuffle=False)
    for batch in itr:
        yield Batch(
            ids=batch['id'],
            src_tokens=batch['net_input']['src_tokens'], src_lengths=batch['net_input']['src_lengths'],
        )


class LevenshteinModel:
    """
    Deployment for levenshtein model
    """

    def __init__(self):
        self.args = self.build_args(self)
        utils.import_user_module(self.args)

        if self.args.buffer_size < 1:
            self.args.buffer_size = 1
        if self.args.max_tokens is None and self.args.max_sentences is None:
            self.args.max_sentences = 1

        assert not self.args.sampling or self.args.nbest == self.args.beam, \
            '--sampling requires --nbest to be equal to --beam'
        assert not self.args.max_sentences or self.args.max_sentences <= self.args.buffer_size, \
            '--max-sentences/--batch-size cannot be larger than --buffer-size'

        print(self.args)

        self.use_cuda = torch.cuda.is_available() and not self.args.cpu

        # Setup task, e.g., translation
        self.task = tasks.setup_task(self.args)

        # Load ensemble
        print('| loading model(s) from {}'.format(self.args.path))
        self.models, self.model_args = checkpoint_utils.load_model_ensemble(
            self.args.path.split(':'),
            arg_overrides=eval(self.args.model_overrides),
            task=self.task,
        )

        # Set dictionaries
        self.src_dict = self.task.source_dictionary
        self.tgt_dict = self.task.target_dictionary

        # Optimize ensemble for generation
        for model in self.models:
            model.make_generation_fast_(
                beamable_mm_beam_size=None if self.args.no_beamable_mm else self.args.beam,
                need_attn=self.args.print_alignment,
            )
            if self.args.fp16:
                model.half()
            if self.use_cuda:
                model.cuda()

        # Initialize generator
        self.generator = self.task.build_generator(self.args)

        # Handle tokenization and BPE
        tokenizer = encoders.build_tokenizer(self.args)
        self.bpe = encoders.build_bpe(self.args)

        # Load alignment dictionary for unknown word replacement
        # (None if no unknown word replacement, empty if no path to align dictionary)
        self.align_dict = utils.load_align_dict(self.args.replace_unk)

        self.max_positions = utils.resolve_max_positions(
            self.task.max_positions(),
            *[model.max_positions() for model in models]
        )

        if self.args.buffer_size > 1:
            print('| Sentence buffer size:', self.args.buffer_size)
        print('| Type the input sentence and press return:')

    def predict(self, file_in='/home/pavel_fakanov/data.txt', file_processed='/home/pavel_fakanov/data_pr.txt'):
        print("Preprocessing article:")
        process_article(self.args, file_in, file_processed)
        start_id = 0

        for inputs in buffered_read(file_processed, self.args.buffer_size):
            results = []
            for batch in make_batches(inputs, self.args, self.task, self.max_positions, self.encode_fn):
                src_tokens = batch.src_tokens
                src_lengths = batch.src_lengths
                if self.use_cuda:
                    src_tokens = src_tokens.cuda()
                    src_lengths = src_lengths.cuda()

                sample = {
                    'net_input': {
                        'src_tokens': src_tokens,
                        'src_lengths': src_lengths,
                    },
                }
                translations = self.task.inference_step(self.generator, self.models, sample)
                for i, (id, hypos) in enumerate(zip(batch.ids.tolist(), translations)):
                    src_tokens_i = utils.strip_pad(src_tokens[i], self.tgt_dict.pad())
                    results.append((start_id + id, src_tokens_i, hypos))

            # sort output to match input order
            for id, src_tokens, hypos in sorted(results, key=lambda x: x[0]):
                if self.src_dict is not None:
                    src_str = self.src_dict.string(src_tokens, self.args.remove_bpe)
                    print('S-{}\t{}'.format(id, src_str))

                # Process top predictions
                for hypo in hypos[:min(len(hypos), self.args.nbest)]:
                    hypo_tokens, hypo_str, alignment = utils.post_process_prediction(
                        hypo_tokens=hypo['tokens'].int().cpu(),
                        src_str=src_str,
                        alignment=hypo['alignment'],
                        align_dict=self.align_dict,
                        tgt_dict=self.tgt_dict,
                        remove_bpe=self.args.remove_bpe,
                    )
                    hypo_str = self.decode_fn(hypo_str)
                    print('H-{}\t{}\t{}'.format(id, hypo['score'], hypo_str))
                    print('P-{}\t{}'.format(
                        id,
                        ' '.join(map(lambda x: '{:.4f}'.format(x), hypo['positional_scores'].tolist()))
                    ))
                    if self.args.print_alignment:
                        alignment_str = " ".join(["{}-{}".format(src, tgt) for src, tgt in alignment])
                        print('A-{}\t{}'.format(
                            id,
                            alignment_str
                        ))

            # update running id counter
            start_id += len(inputs)

    def encode_fn(self, x):
        if self.tokenizer is not None:
            x = self.tokenizer.encode(x)
        if self.bpe is not None:
            x = self.bpe.encode(x)
        return x

    def decode_fn(self, x):
        if self.bpe is not None:
            x = self.bpe.decode(x)
        if self.tokenizer is not None:
            x = self.tokenizer.decode(x)
        return x

    def build_args(self):
        fields = ['max_tokens', 'max_sentences', 'buffer_size', 'sampling', 'nbest', 'beam', \
                'print_alignment', 'cpu', 'path', 'model_overrides', 'no_beamable_mm', \
                'fp16', 'min_len', 'no_early_stop', 'unnormalized', 'lenpen', 'unkpen', \
                'sampling_topk', 'sampling_temperature', 'diverse_beam_groups', \
                'diverse_beam_strength', 'no_repeat_ngram_size', 'replace_unk', 'remove_bpe', \
                'max_len_a', 'max_len_b', 'task', 'left_pad_source', 'left_pad_target', \
                'source_lang', 'target_lang', 'data', 'fp16_init_scale', 'fp16_scale_window', \
                'gen_subset', 'input', 'log_format', 'log_interval', 'max_source_positions', \
                'max_target_positions', 'no_progress_bar', 'num_shards', 'prefix_size', \
                'quiet', 'score_reference', 'seed', 'shard_id', 'skip_invalid_size_inputs_valid_test', \
                'upsample_primary', 'encoder_embed_path', 'dataset_name']

        defaults_dict = dict(zip(fields, (None,) * len(fields)))
        args = SimpleNamespace(**defaults_dict)

        args.buffer_size = BUFFER_SIZE
        args.beam = BEAM
        args.nbest = NBEST
        args.max_len_a = MAX_LEN_A
        args.max_len_b = MAX_LEN_B
        args.min_len = MIN_LEN
        args.no_early_stop = NO_EARLY_STOP
        args.unnormalized = UNNORMALIZED
        args.no_beamable_mm = NO_BEAMABLE_MM
        args.lenpen = LENPEN
        args.unkpen = UNKPEN
        args.no_repeat_ngram_size = NO_REPEAT_NGRAM_SIZE
        args.sampling = SAMPLING
        args.sampling_topk = SAMPLING_TOPK
        args.sampling_temperature = SAMPLING_TEMPERATURE
        args.diverse_beam_groups = DIVERSE_BEAM_GROUPS
        args.diverse_beam_strength = DIVERSE_BEAM_STRENGTH
        args.print_alignment = PRINT_ALIGNMENT
        args.cpu = CPU
        args.fp16 = FP16
        args.task = TASK
        args.left_pad_source = LEFT_PAD_SOURCE
        args.left_pad_target = LEFT_PAD_TARGET
        args.fp16_init_scale = FP16_INIT_SCALE
        args.log_interval = LOG_INTERVAL
        args.max_source_positions = MAX_SOURCE_POSITIONS
        args.max_target_positions = MAX_TARGET_POSITIONS
        args.model_overrides = MODEL_OVERRIDES
        args.no_progress_bar = NO_PROGRESS_BAR
        args.num_shards = NUM_SHARDS
        args.prefix_size = PREFIX_SIZE
        args.quiet = QUIET
        args.score_reference = SCORE_REFERENCE
        args.seed = SEED
        args.shard_id = SHARD_ID
        args.skip_invalid_size_inputs_valid_test = SKIP_INVALID_SIZE_INPUTS_VALID_TEST
        args.upsample_primary = UPSAMPLE_PRIMARY
        args.source_lang = 'articles'
        args.target_lang = 'summaries'
        args.data = [DATA_PATH]
        args.path = CHECKPOINT_PATH
        args.dataset_name = DATASET_NAME
        return args