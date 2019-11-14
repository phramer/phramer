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

from rouge import io
from rouge import rouge_scorer
from rouge import scoring


def main(argv):
    scorer = rouge_scorer.RougeScorer(FLAGS.rouge_types, FLAGS.use_stemmer)
    aggregator = scoring.BootstrapAggregator() if FLAGS.aggregate else None
    io.compute_scores_and_write_to_csv(
        FLAGS.target_filepattern,
        FLAGS.prediction_filepattern,
        FLAGS.output_filename,
        scorer,
        aggregator,
        delimiter=FLAGS.delimiter,
    )

