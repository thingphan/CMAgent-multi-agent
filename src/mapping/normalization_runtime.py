"""Run independent normalization work items; the caller alone writes checkpoints."""
from concurrent.futures import ThreadPoolExecutor, as_completed


def run_tasks(tasks, process, workers=1):
    if workers < 1:
        raise ValueError("workers must be positive")
    if workers == 1:
        for _, row in tasks.iterrows():
            yield process(row)
        return
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(process, row) for _, row in tasks.iterrows()]
        for future in as_completed(futures):
            yield future.result()
