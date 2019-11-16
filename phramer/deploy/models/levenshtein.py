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
        self.args = self.build_args()
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
        self.tokenizer = encoders.build_tokenizer(self.args)
        self.bpe = encoders.build_bpe(self.args)

        # Load alignment dictionary for unknown word replacement
        # (None if no unknown word replacement, empty if no path to align dictionary)
        self.align_dict = utils.load_align_dict(self.args.replace_unk)

        self.max_positions = utils.resolve_max_positions(
            self.task.max_positions(),
            *[model.max_positions() for model in self.models]
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
        fields = ['beam', 'bpe', 'buffer_size', 'cpu', 'criterion',  'data', 'dataset_impl', 'decoding_format', \
                  'diverse_beam_groups', 'diverse_beam_strength', 'empty_cache_freq', 'force_anneal', 'fp16', \
                  'fp16_init_scale', 'fp16_scale_tolerance', 'fp16_scale_window', 'gen_subset', 'iter_decode_eos_penalty', \
                  'iter_decode_force_max_iter', 'iter_decode_max_iter', 'lazy_load', 'left_pad_source', 'left_pad_target', \
                  'lenpen', 'load_alignments', 'log_format', 'log_interval', 'lr_scheduler', 'lr_shrink', \
                  'match_source_len', 'max_len_a', 'max_len_b', 'max_sentences', 'max_source_positions', \
                  'max_target_positions', 'max_tokens', 'memory_efficient_fp16', 'min_len', 'min_loss_scale', \
                  'model_overrides', 'momentum', 'nbest', 'no_beamable_mm', 'no_early_stop', 'no_progress_bar', \
                  'no_repeat_ngram_size', 'noise', 'num_shards', 'num_workers', 'optimizer', 'path', 'prefix_size', \
                  'print_alignment', 'print_step', 'quiet', 'raw_text', 'dataset_name', 'remove_bpe', 'replace_unk', \
                  'required_batch_size_multiple', 'results_path', 'retain_iter_history', 'sacrebleu', 'sampling', \
                  'sampling_topk', 'sampling_topp', 'score_reference', 'seed', 'shard_id', 'skip_invalid_size_inputs_valid_test', \
                  'task', 'temperature', 'tensorboard_logdir', 'threshold_loss_scale', 'tokenizer', 'unkpen', \
                  'unnormalized', 'upsample_primary', 'user_dir', 'warmup_updates', 'weight_decay']

        defaults_dict = dict(zip(fields, (None,) * len(fields)))
        args = SimpleNamespace(**defaults_dict)

        args.beam = BEAM
        args.buffer_size = BUFFER_SIZE
        args.cpu = CPU
        args.criterion = CRITERION
        args.data = DATA_PATH
        args.diverse_beam_groups = DIVERSE_BEAM_GROUPS
        args.diverse_beam_strength = DIVERSE_BEAM_STRENGTH
        args.empty_cache_freq = EMPTY_CACHE_FREQ
        args.fp16 = FP16
        args.fp16_init_scale = FP16_INIT_SCALE
        args.fp16_scale_tolerance = FP16_SCALE_TOLERANCE
        args.gen_subset = GEN_SUBSET
        args.iter_decode_eos_penalty = ITER_DECODE_EOS_PENALTY
        args.iter_decode_force_max_iter = ITER_DECODE_FORCE_MAX_ITER
        args.iter_decode_max_iter = ITER_DECODE_MAX_ITER
        args.lazy_load = LAZY_LOAD
        args.left_pad_source = LEFT_PAD_SOURCE
        args.left_pad_target = LEFT_PAD_TARGET
        args.lenpen = LENPEN
        args.load_alignments = LOAD_ALIGNMENTS
        args.log_interval = LOG_INTERVAL
        args.lr_scheduler = LR_SHEDULER
        args.lr_shrink = LR_SHRINK
        args.match_source_len = MATCH_SOURCE_LEN
        args.max_len_a = MAX_LEN_A
        args.max_len_b = MAX_LEN_B
        args.max_sentences = MAX_SENTENCES
        args.max_source_positions = MAX_SOURCE_POSITIONS
        args.max_target_positions = MAX_TARGET_POSITIONS
        args.memory_efficient_fp16 MEMORY_EFFICIENT_FP16
        args.min_len = MIN_LEN
        args.min_loss_scale = MIN_LOSS_SCALE
        args.model_overrides = MODEL_OVERRIDES
        args.momentum = MOMENTUM
        args.nbest = NBEST
        args.no_beamable_mm = NO_BEAMABLE_MM
        args.no_early_stop = NO_EARLY_STOP
        args.no_progress_bar = NO_PROGRESS_BAR
        args.no_repeat_ngram_size = NO_REPEAT_NGRAM_SIZE
        args.noise = NOISE
        args.num_shards = NUM_SHARDS
        args.num_workers = NUM_WORKERS
        args.optimizer = OPTIMIZER
        args.path = CHECKPOINT_PATH
        args.prefix_size = PREFIX_SIZE
        args.print_alignment = PRINT_ALIGNMENT
        args.print_step = PRINT_STEP
        args.quiet = QUIET
        args.raw_text = RAW_TEXT
        args.remove_bpe = REMOVE_BPE
        args.required_batch_size_multiple = REQUIRED_BATCH_SIZE_MULTIPLE
        args.retain_iter_history = RETAIN_ITER_HISTORY
        args.sacrebleu = SACREBLEU
        args.sampling = SAMPLING
        args.sampling_topk = SAMPLING_TOPK
        args.sampling_topp = SAMPLING_TOPP
        args.score_reference = SCORE_REFERENCE
        args.seed = SEED
        args.shard_id = SHARD_ID
        args.skip_invalid_size_inputs_valid_test = SKIP_INVALID_SIZE_INPUTS_VALID_TEST
        args.task = TASK
        args.temperature = TEMPERATURE
        args.tensorboard_logdir = TENSORBOARD_LOGDIR
        args.unkpen = UNKPEN
        args.unnormalized = UNNORMALIZED
        args.upsample_primary = UPSAMPLE_PRIMARY
        args.warmup_updates = WARMUP_UPDATES
        args.weight_decay = WEIGHT_DECAY

        args.dataset_name = DATASET_NAME
        args.source_lang = 'articles'
        args.target_lang = 'summaries'

        return args
