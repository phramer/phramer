import gc
import os
from phramer.config import MAX_TASKS_PER_CHILD
import multiprocessing as mp


class ParallelIO:
    def __init__(self, num_workers=mp.cpu_count() - 1):
        self.num_workers = max(mp.cpu_count() - 1, num_workers)

    def _chunkify(self, filename: str, size=1024 * 1024 * 1000, skiplines=0):
        chunks = []
        file_end = os.path.getsize(filename)
        with open(filename, "rb") as f:
            if skiplines > 0:
                for _ in range(skiplines):
                    f.readline()
            chunk_end = f.tell()
            while True:
                chunk_start = chunk_end
                f.seek(f.tell() + size, os.SEEK_SET)
                f.readline()
                chunk_end = f.tell()
                chunk_size = chunk_end - chunk_start
                chunks.append((chunk_start, chunk_size, filename))
                if chunk_end > file_end:
                    break
        return chunks

    def _apply(self, task):
        chunk_start, chunk_size, filename, worker = task[:4]
        payload = task[4:]
        processed_chunk = []
        with open(filename, "rb") as f:
            f.seek(chunk_start)
            data = f.read(chunk_size).decode("utf-8")
            lines = data.splitlines()
            for line in lines:
                processed_line = worker(line, *payload)
                if processed_line is not None:
                    processed_chunk.append(processed_line)
        return processed_chunk

    def run(
        self,
        input_filename,
        worker,
        payload,
        fout=None,
        chunk_size=100,
        skiplines=0,
    ):
        jobs = self._chunkify(
            input_filename, size=1024 * 1024 * chunk_size, skiplines=skiplines
        )
        jobs = [[job, worker] + payload for job in jobs]
        pool = mp.Pool(self.num_workers, maxtasksperchild=MAX_TASKS_PER_CHILD)
        outputs = []
        lines_counter = 0
        for chunk_idx in range(0, len(jobs), self.num_workers):
            print("Processing chunk #{}...".format(chunk_idx))
            chunk_output = pool.map(
                self._apply, jobs[chunk_idx : chunk_idx + self.num_workers]
            )
            for processed_chunk in chunk_output:
                for line in processed_chunk:
                    if not fout:
                        outputs.append(fout)
                    else:
                        print(line, file=fout)
                    lines_counter += 1
            del chunk_output
            gc.collect()
        pool.close()
        pool.terminate()
        print("Processed {} lines.".format(lines_counter))
        return outputs
