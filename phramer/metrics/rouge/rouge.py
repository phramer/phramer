r"""Main routine to calculate ROUGE scores across text files.

Designed to replicate scores computed by the ROUGE perl implementation as
closely as possible.

Output is a text file in CSV format.

Sample usage:

rouge ---rouge_types=rouge1,rouge2,rougeL \
    --target_filepattern=*.targets \
    --prediction_fliepattern=*.decodes \
    --output_filename=scores.csv \
    --use_stemmer

Which is equivalent to calling the perl ROUGE script as:

ROUGE-1.5.5.pl -m -e ./data -n 2 -a /tmp/rouge/settings.xml

Where settings.xml provides target and decode text.
"""

import argparse
from phramer.metrics.rouge import io
from phramer.metrics.rouge import rouge_scorer
from phramer.metrics.rouge import scoring


def main(args):
    scorer = rouge_scorer.RougeScorer(args.rouge_types, args.use_stemmer)
    aggregator = scoring.BootstrapAggregator() if args.aggregate else None
    io.compute_scores_and_write_to_csv(
        args.target_filepattern,
        args.prediction_filepattern,
        args.output_filename,
        scorer,
        aggregator,
        delimiter=args.delimiter,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--target-filepattern",
        default=None,
        type=str,
        help="Files containing target text.",
        required=True,
    )
    parser.add_argument(
        "--prediction-filepattern",
        default=None,
        type=str,
        required=True,
        help="Files containing prediction text.",
    )
    parser.add_argument(
        "--output-filename",
        default=None,
        type=str,
        help="File in which to write calculated ROUGE scores as a CSV",
        required=True,
    )
    parser.add_argument(
        "--delimiter",
        default="\n",
        type=str,
        help="Record delimiter in files.",
    )
    parser.add_argument(
        "--rouge-types",
        nargs="+",
        type=str,
        default=["rouge1", "rouge2", "rougeL"],
        required=True,
        help="List of ROUGE types to calculate.",
    )
    parser.add_argument(
        "--use-stemmer",
        action="store_true",
        default=False,
        help="Whether to use Porter stemmer to remove common suffixes.",
    )
    parser.add_argument(
        "--aggregate",
        action="store_true",
        default=True,
        help="Write aggregates if this is set to True",
    )
    args = parser.parse_args()
    main(args)
