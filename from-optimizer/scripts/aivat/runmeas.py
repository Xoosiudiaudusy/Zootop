import resource, subprocess, sys, time
t = time.time(); r = subprocess.run(sys.argv[2:], stdout=open(sys.argv[1] + ".txt", "w"), stderr=subprocess.STDOUT)
ru = resource.getrusage(resource.RUSAGE_CHILDREN)
open(sys.argv[1] + ".time", "w").write(f"Elapsed {time.time() - t:.1f} s; Maximum resident {ru.ru_maxrss / 1024:.0f} MB; exit {r.returncode}\n")
