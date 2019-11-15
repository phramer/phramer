import torch

from collections import namedtuple
import numpy as np
import sys
from fairseq import data, options, tasks, tokenizer, utils
from fairseq.sequence_generator import SequenceGenerator


from phramer.data.dataset import RIANewsDataset
from phramer.deploy.models_config.ria_seq2seq import (
    DATA_PATH,
    CHECKPOINT_PATH,
    DATASET_NAME,
    INPUT_FILE_NAME,
    REMOVE_BPE,
    MIN_LEN,
    BEAM,
    NO_REPEAT_NGRAM,
    NBEST,
    CUDA_VISIBLE_DEVICES
)


def process_article(data_path, save_path):
    if DATASET_NAME == 'ria':
        ria = RIANewsDataset()
        with open(data_path, 'r+') as f:
            article = f.read()

        article = ria._process_article(article)
        article = article.replace('\n', ' ')

        f = open(save_path, 'w+')
        f.write(article)
        f.close()


def buffered_read(buffer_size):
    buffer = []
    for src_str in sys.stdin:
        buffer.append(src_str.strip())
        if len(buffer) >= buffer_size:
            yield buffer
            buffer = []

    if len(buffer) > 0:
        yield buffer

def make_batches(lines, args, task, max_positions):
    tokens = [
        tokenizer.Tokenizer.tokenize(src_str, task.source_dictionary, add_if_not_exist=False).long()
        for src_str in lines
    ]
    lengths = np.array([t.numel() for t in tokens])
    itr = task.get_batch_iterator(
        dataset=data.LanguagePairDataset(tokens, lengths, task.source_dictionary),
        max_tokens=args.max_tokens,
        max_sentences=args.max_sentences,
        max_positions=max_positions,
    ).next_epoch_itr(shuffle=False)
    for batch in itr:
        yield Batch(
            srcs=[lines[i] for i in batch['id']],
            tokens=batch['net_input']['src_tokens'],
            lengths=batch['net_input']['src_lengths'],
        ), batch['id']


class Seq2SeqModel:
    """
    Deployment for seq2seq model
    """

    def __init__(self):
        if args.buffer_size < 1:
            args.buffer_size = 1
        if args.max_tokens is None and args.max_sentences is None:
            args.max_sentences = 1

        assert not args.sampling or args.nbest == args.beam, \
            '--sampling requires --nbest to be equal to --beam'
        assert not args.max_sentences or args.max_sentences <= args.buffer_size, \
            '--max-sentences/--batch-size cannot be larger than --buffer-size'

        print(args)

        use_cuda = torch.cuda.is_available() and not args.cpu

        # Setup task, e.g., translation
        task = tasks.setup_task(args)

        # Load ensemble
        print('| loading model(s) from {}'.format(args.path))
        model_paths = args.path.split(':')
        models, model_args = utils.load_ensemble_for_inference(model_paths, task, model_arg_overrides=eval(args.model_overrides))

        # Set dictionaries
        tgt_dict = task.target_dictionary

        # Optimize ensemble for generation
        for model in models:
            model.make_generation_fast_(
                beamable_mm_beam_size=None if args.no_beamable_mm else args.beam,
                need_attn=args.print_alignment,
            )
            if args.fp16:
                model.half()

        # Initialize generator
        translator = SequenceGenerator(
            models, tgt_dict, beam_size=args.beam, minlen=args.min_len,
            stop_early=(not args.no_early_stop), normalize_scores=(not args.unnormalized),
            len_penalty=args.lenpen, unk_penalty=args.unkpen,
            sampling=args.sampling, sampling_topk=args.sampling_topk, sampling_temperature=args.sampling_temperature,
            diverse_beam_groups=args.diverse_beam_groups, diverse_beam_strength=args.diverse_beam_strength,
            no_repeat_ngram_size=args.no_repeat_ngram_size,
        )

        if use_cuda:
            translator.cuda()


        # Load alignment dictionary for unknown word replacement
        # (None if no unknown word replacement, empty if no path to align dictionary)
        align_dict = utils.load_align_dict(args.replace_unk)

        def make_result(src_str, hypos):
            result = Translation(
                src_str='O\t{}'.format(src_str),
                hypos=[],
                pos_scores=[],
                alignments=[],
            )

            # Process top predictions
            for hypo in hypos[:min(len(hypos), args.nbest)]:
                hypo_tokens, hypo_str, alignment = utils.post_process_prediction(
                    hypo_tokens=hypo['tokens'].int().cpu(),
                    src_str=src_str,
                    alignment=hypo['alignment'].int().cpu() if hypo['alignment'] is not None else None,
                    align_dict=align_dict,
                    tgt_dict=tgt_dict,
                    remove_bpe=args.remove_bpe,
                )
                result.hypos.append('H\t{}\t{}'.format(hypo['score'], hypo_str))
                result.pos_scores.append('P\t{}'.format(
                    ' '.join(map(
                        lambda x: '{:.4f}'.format(x),
                        hypo['positional_scores'].tolist(),
                    ))
                ))
                result.alignments.append(
                    'A\t{}'.format(' '.join(map(lambda x: str(utils.item(x)), alignment)))
                    if args.print_alignment else None
                )
            return result

        def process_batch(batch):
            tokens = batch.tokens
            lengths = batch.lengths

            if use_cuda:
                tokens = tokens.cuda()
                lengths = lengths.cuda()

            encoder_input = {'src_tokens': tokens, 'src_lengths': lengths}
            translations = translator.generate(
                encoder_input,
                maxlen=int(args.max_len_a * tokens.size(1) + args.max_len_b),
            )

            return [make_result(batch.srcs[i], t) for i, t in enumerate(translations)]

        max_positions = utils.resolve_max_positions(
            task.max_positions(),
            *[model.max_positions() for model in models]
        )

        if args.buffer_size > 1:
            print('| Sentence buffer size:', args.buffer_size)
        print('| Type the input sentence and press return:')


    def build_args(self):
        Args = namedtuple("Args", ['max_tokens', 'max_sentences', 'buffer_size', 'sampling', 'nbest', 'beam', \
                                   'print_alignment', 'cpu', 'path', 'model_overrides', 'no_beamable_mm', \
                                   'fp16', 'min_len', 'no_early_stop', 'unnormalized', 'lenpen', 'unkpen', \
                                   'sampling', 'sampling_topk', 'sampling_temperature', 'diverse_beam_groups', \
                                   'diverse_beam_strength', 'no_repeat_ngram_size', 'replace_unk', 'remove_bpe', \
                                   'max_len_a', 'max_len_b'])

        args = Args()
        return args
