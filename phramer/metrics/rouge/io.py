"""Library for reading/writing input and score files."""
import glob
import logging
import multiprocessing as mp
from functools import partial

import six
from six.moves import zip, zip_longest
from tqdm import tqdm

from phramer.config import MAX_TASKS_PER_CHILD
from phramer.utils.file import count_lines

logging.basicConfig(level=logging.INFO)


def compute_scores_and_write_to_csv(
    target_filepattern,
    prediction_filepattern,
    output_filename,
    scorer,
    aggregator,
    delimiter="\n",
    num_workers=mp.cpu_count() - 1,
):
    """Runs aggregate score calculations and outputs results to a CSV file.

       Args:
         target_filepattern: Pattern for files containing target text.
         prediction_filepattern: Pattern for files containing prediction text.
         output_filename: Name of file to write results to.
         scorer: A BaseScorer object to compute scores.
         aggregator: An aggregator to aggregate scores. If None, outputs are
           per-example scores.
         delimiter: Record delimiter.
     """

    target_filenames = _glob(target_filepattern)
    prediction_filenames = _glob(prediction_filepattern)
    if num_workers <= 0:
        raise ValueError("The number of workers must be positive.")
    elif num_workers == 1:
        scores = _compute_scores(
            target_filenames, prediction_filenames, scorer, delimiter
        )
        if aggregator:
            for score in scores:
                aggregator.add_scores(score)
            _write_aggregates_to_csv(output_filename, aggregator.aggregate())
        else:
            _write_scores_to_csv(output_filename, scores)
    else:
        if aggregator is None:
            raise NotImplementedError(
                "Computing per-example values in the parallel "
                + "environment is not supported."
            )
        _parallel_compute_and_write_scores(
            target_filenames,
            prediction_filenames,
            output_filename,
            scorer,
            delimiter,
            aggregator,
            num_workers,
        )


def _glob(filepattern):
    return glob.glob(filepattern)  # pylint: disable=unreachable


def _open(filepattern, mode="r"):
    return open(filepattern, mode)  # pylint: disable=unreachable


def _record_gen(filename, delimiter):
    """Opens file and yields records separated by delimiter."""
    with _open(filename) as f:
        records = f.read().split(six.ensure_str(delimiter))
    if records[-1]:
        logging.warn("Expected delimiter at end of file")
    else:
        records = records[:-1]
    for record in records:
        yield record


def _compute_scores(target_filenames, prediction_filenames, scorer, delimiter):
    """Computes aggregates scores across the given target and prediction files.

    Args:
      target_filenames: 
        List of filenames from which to read target lines.
      prediction_filenames: 
        List of filenames from which to read prediction lines.
      scorer: A BaseScorer object to compute scores.
      delimiter: string delimiter between each record in input files
    Returns:
      A list of dicts mapping score_type to Score objects.
    Raises:
      ValueError: If invalid targets or predictions are provided.
    """

    if len(target_filenames) < 1 or len(target_filenames) != len(
        prediction_filenames
    ):
        raise ValueError(
            "Must have equal and positive number of target and "
            "prediction files. Found: %d target files, %d prediction "
            "files." % (len(target_filenames), len(prediction_filenames))
        )

    scores = []
    for target_filename, prediction_filename in zip(
        sorted(target_filenames), sorted(prediction_filenames)
    ):
        logging.info("Reading targets from %s.", target_filename)
        logging.info("Reading predictions from %s.", prediction_filename)
        targets = _record_gen(target_filename, delimiter)
        preds = _record_gen(prediction_filename, delimiter)
        for target_rec, prediction_rec in zip_longest(targets, preds):
            if target_rec is None or prediction_rec is None:
                raise ValueError(
                    "Must have equal number of lines across target and "
                    "prediction files. Mismatch between files: %s, %s."
                    % (target_filename, prediction_filename)
                )
            scores.append(scorer.score(target_rec, prediction_rec))

    return scores


def _write_aggregates_to_csv(output_filename, aggregates):
    """Writes aggregate scores to an output CSV file.

     Output file is a comma separated where each line has the format:
       score_type-(P|R|F),low_ci,mean,high_ci
  
     P/R/F indicates whether the score is a precision, recall or f-measure.
  
     Args:
       output_filename: Name of file to write results to.
       aggregates: A dict mapping each score_type to a AggregateScore object.
     """

    logging.info("Writing results to %s.", output_filename)
    with _open(output_filename, "w") as output_file:
        output_file.write("score_type,low,mid,high\n")
        for score_type, aggregate in sorted(aggregates.items()):
            output_file.write(
                "%s-R,%f,%f,%f\n"
                % (
                    score_type,
                    aggregate.low.recall,
                    aggregate.mid.recall,
                    aggregate.high.recall,
                )
            )
            output_file.write(
                "%s-P,%f,%f,%f\n"
                % (
                    score_type,
                    aggregate.low.precision,
                    aggregate.mid.precision,
                    aggregate.high.precision,
                )
            )
            output_file.write(
                "%s-F,%f,%f,%f\n"
                % (
                    score_type,
                    aggregate.low.fmeasure,
                    aggregate.mid.fmeasure,
                    aggregate.high.fmeasure,
                )
            )
    logging.info("Finished writing results.")


def _write_scores_to_csv(output_filename, scores):
    """Writes scores for each individual example to an output CSV file.

    Output file is a comma separated where each line has the format:
      id,score1,score2,score3,...

    The header row indicates the type of each score column.

    Args:
      output_filename: Name of file to write results to.
      scores: A list of dicts mapping each score_type to a Score object.
    """

    if len(scores) < 1:
        logging.warn("No scores to write")
        return
    rouge_types = sorted(scores[0].keys())

    logging.info("Writing results to %s.", output_filename)
    with _open(output_filename, "w") as out_file:
        out_file.write("id")
        for rouge_type in rouge_types:
            out_file.write(",{t}-P,{t}-R,{t}-F".format(t=rouge_type))
        out_file.write("\n")
        for i, result in enumerate(scores):
            out_file.write("%d" % i)
            for rouge_type in rouge_types:
                out_file.write(
                    ",%f,%f,%f"
                    % (
                        result[rouge_type].precision,
                        result[rouge_type].recall,
                        result[rouge_type].fmeasure,
                    )
                )
            out_file.write("\n")
    logging.info("Finished writing results.")


def _process_lines(lines, scorer):
    target_line, prediction_line = lines
    if target_line is None or prediction_line is None:
        raise ValueError(
            "Must have equal number of lines across target "
            + "and prediction files."
        )
    score = scorer.score(target_line, prediction_line)
    return score


def _parallel_compute_and_write_scores(
    target_filenames,
    prediction_filenames,
    output_filename,
    scorer,
    delimiter,
    aggregator,
    num_workers=mp.cpu_count() - 1,
):
    if len(target_filenames) < 1 or len(target_filenames) != len(
        prediction_filenames
    ):
        raise ValueError(
            "Must have equal and positive number of target and "
            "prediction files. Found: %d target files, %d prediction "
            "files." % (len(target_filenames), len(prediction_filenames))
        )

    pool = mp.Pool(num_workers, maxtasksperchild=MAX_TASKS_PER_CHILD)

    for target_filename, prediction_filename in zip(
        sorted(target_filenames), sorted(prediction_filenames)
    ):
        logging.info("Reading targets from %s.", target_filename)
        logging.info("Reading predictions from %s.", prediction_filename)
        targets = _record_gen(target_filename, delimiter)
        preds = _record_gen(prediction_filename, delimiter)
        with tqdm(total=count_lines(target_filename), desc="Lines") as pbar:
            for score in pool.imap_unordered(
                partial(_process_lines, scorer=scorer), zip(targets, preds)
            ):
                aggregator.add_scores(score)
                pbar.update()
    pool.close()
    pool.join()
    _write_aggregates_to_csv(output_filename, aggregator.aggregate())
