import sys, time, multiprocessing as mp
sys.path.insert(0, '.')
from games.cantstop.generate_training_data import worker_generate_batch

if __name__ == "__main__":
    workers = mp.cpu_count()
    print(f"Testing with {workers} workers...")

    start = time.time()
    args = [(100, i) for i in range(workers)]
    with mp.Pool(workers) as pool:
        results = pool.map(worker_generate_batch, args)
    elapsed = time.time() - start

    total = sum(len(r) for r in results)
    rate = total / elapsed
    print(f"Records generated: {total:,}")
    print(f"Time: {elapsed:.2f}s")
    print(f"Rate: {rate:,.0f} records/second")
    print(f"Estimated 8 hours: {int(rate * 3600 * 8):,}")