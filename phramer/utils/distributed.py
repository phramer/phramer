import os
from tqdm import tqdm


def chunkify(filename: str, size=1024 * 1024 * 1000, skiplines=0):
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


def process_chunk(task):
    chunk_start, chunk_size, filename, worker = task[:4]
    payload = task[4:]
    processed_chunk = []
    with open(filename, "rb") as f:
        f.seek(chunk_start)
        data = f.read(chunk_size).decode("utf-8")
        lines = data.splitlines()
        with tqdm(
            total=len(lines),
            desc="Chunk {} to {} of {}".format(
                chunk_start, chunk_start + chunk_size, filename
            ),
        ) as pbar:
            for line in lines:
                processed_line = worker(line, *payload)
                if processed_line is not None:
                    processed_chunk.append(processed_line)
                pbar.update()
    return processed_chunk
