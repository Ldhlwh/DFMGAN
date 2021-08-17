import os, sys

FILENAME = 'log_recalc_metrics.txt'

assert len(sys.argv) >= 2
dir = sys.argv[1]
dir_file = os.path.join(dir, FILENAME)
print('Will recompute network snapshots in %s' % dir)
print('Results will be stored in %s' % dir_file)

outfile = open(dir_file, 'w')
snapshots = []
for f in os.listdir(dir):
    if f.startswith('network') and f.endswith('.pkl'):
        snapshots.append(f)
snapshots.sort()

for f in snapshots:
    print('Calculating metrics for %s' % f)
    os.system('echo %s >> %s' % (f, dir_file))
    os.system('python calc_metrics.py --network %s --verbose False >> %s' % (os.path.join(dir, f), dir_file))

outfile.close()
        