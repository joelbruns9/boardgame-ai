import sys, time, multiprocessing as mp
sys.path.insert(0, '.')
from games.cantstop.generate_training_data import worker_generate_batch

if __name__ == "__main__":
    print(f"cpu_count(): {mp.cpu_count()}\n")
    print("Testing worker counts:")

    for workers in [4, 8, 12, 16]:
        args = [(50, i) for i in range(workers)]
        start = time.time()
        with mp.Pool(workers) as pool:
            results = pool.map(worker_generate_batch, args)
        elapsed = time.time() - start
        total = sum(len(r) for r in results)
        rate = total / elapsed
        games = workers * 50
        print(f"  {workers:2d} workers: {rate:>8,.0f} records/s "
              f"({games} games in {elapsed:.1f}s)")

    print("\nRecommended: use the worker count with highest records/s")