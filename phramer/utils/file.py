import gc
from phramer.config import MAX_TASKS_PER_CHILD
from phramer.utils.distributed import chunkify, process_chunk
import multiprocessing as mp


def count_lines(filename):
    with open(filename, "rb") as f:
        num_lines = sum(1 for _ in f)
    return num_lines


class ParallelIO:
    def __init__(self, num_workers=mp.cpu_count() - 1):
        self.num_workers = num_workers

    def run(
        self,
        input_filename,
        worker,
        payload,
        fout=None,
        chunk_size=100,
        skiplines=0,
    ):
        jobs = chunkify(
            input_filename, size=1024 * 1024 * chunk_size, skiplines=skiplines
        )
        jobs = [list(job) + [worker] + payload for job in jobs]
        pool = mp.Pool(self.num_workers, maxtasksperchild=MAX_TASKS_PER_CHILD)
        outputs = []
        lines_counter = 0
        for chunk_idx in range(0, len(jobs), self.num_workers):
            print("Processing chunk #{}...".format(chunk_idx))
            chunk_output = pool.map(
                process_chunk, jobs[chunk_idx : chunk_idx + self.num_workers]
            )
            for processed_chunk in chunk_output:
                for line in processed_chunk:
                    if not fout:
                        outputs.append(line)
                    else:
                        print(line, file=fout)
                    lines_counter += 1
            del chunk_output
            gc.collect()
        pool.close()
        pool.terminate()
        print("Processed {} lines.".format(lines_counter))
        return outputs
