"""The GPU trainer's host part under CPU contention: batches prepared (deals, buckets, showdown strengths) while a
stand-in device "runs" each batch for --device-ms (sleeping, as the host thread waiting for a GPU does).

    python scripts/gpu_host_bench.py --buckets K.json --threads 4 --hogs 0,4 --device-ms 22

HU 200bb wide game of the bucketer (bucket tables from NEGPLURIBUS_BUCKET_TABLES).  Per path (the pool with a ring
of --depth batches, or the old threads-per-batch path) and per competing load (--hogs busy processes): iterations
per second, and the time the device waited for the host.  The tables are not touched (a host-only benchmark).
"""
import argparse
import multiprocessing as mp
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from negpluribus import fast  # noqa: E402
from negpluribus.abstraction import load_bucketer  # noqa: E402
from negpluribus.cfr import GameSpec  # noqa: E402
from negpluribus.engine import Street  # noqa: E402
from negpluribus.fast.trainer import core_bucketer, spec_to_dict  # noqa: E402


def hog(stop):
    x = 0
    while not stop.is_set():
        for _ in range(100000):
            x += 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--buckets", required=True)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--hogs", default="0,4")
    ap.add_argument("--device-ms", type=float, default=22.0, help="the device's time per batch (735k it/s at 16384: 22 ms)")
    ap.add_argument("--batch", type=int, default=16384)
    ap.add_argument("--batches", type=int, default=60)
    ap.add_argument("--depth", default="3")
    args = ap.parse_args()
    core = fast.core()
    bk = load_bucketer(args.buckets)
    spec = GameSpec(n_players=2, stack_bb=200, max_street=Street.RIVER, preflop_fracs=(0.5, 1.0, 3.0),
                    postflop_fracs=(0.5, 1.0, 2.0, 4.0), max_raises_per_street=3, n_buckets=bk.n_buckets,
                    bucket_kind=getattr(bk, "kind", "ehs"))
    cbk = core_bucketer(bk)
    print(f"{args.threads} host threads, batch {args.batch}, device {args.device_ms} ms per batch "
          f"(device-bound ceiling {args.batch / args.device_ms * 1000 if args.device_ms > 0 else float('inf'):,.0f} it/s)", flush=True)
    for hogs in [int(h) for h in args.hogs.split(",")]:
        stop = mp.Event()
        procs = [mp.Process(target=hog, args=(stop,), daemon=True) for _ in range(hogs)]
        for p in procs:
            p.start()
        time.sleep(0.5)
        for path in ["old"] + [f"pool{d}" for d in args.depth.split(",")]:
            ft = core.FlatTrainer(spec_to_dict(spec), cbk, 0, True, args.threads)
            ft.batch_size = args.batch
            ft.prep_bench = True
            ft.prep_bench_ms = args.device_ms
            ft.prep_pool = path != "old"
            if path != "old":
                ft.prep_depth = int(path[4:])
            ft.train(args.batch * 2)  # warm-up (bucket caches, threads)
            w0 = ft.ms_wait_prepare
            t = time.perf_counter()
            ft.train(args.batch * args.batches)
            dt = time.perf_counter() - t
            wait = (ft.ms_wait_prepare - w0) / args.batches
            print(f"  {hogs} busy processes, {path:6s}: {args.batch * args.batches / dt:>10,.0f} it/s, "
                  f"device waited {wait:6.1f} ms per batch", flush=True)
        stop.set()
        for p in procs:
            p.join()


if __name__ == "__main__":
    main()
