from __future__ import division, print_function
import numpy as np
import sys, inspect, os, functools, time, collections
from future.utils import iteritems

### use @timed for really basic timing

_timings = collections.defaultdict(list)

def timed(func):
    @functools.wraps(func)
    def wrapped(*args,**kwargs):
        tic = time.time()
        out = func(*args,**kwargs)
        _timings[func].append(time.time() - tic)
        return out
    return wrapped

def show_timings(stream=None):
    if stream is None:
        stream = sys.stdout
    if len(_timings) > 0:
        results = [(inspect.getsourcefile(f),f.__name__,
            len(vals),np.sum(vals),np.mean(vals),np.std(vals))
            for f, vals in iteritems(_timings)]
        filename_lens = max(len(filename) for filename, _, _, _, _, _ in results)
        name_lens = max(len(name) for _, name, _, _, _, _ in results)

        fmt = '{:>%d} {:>%d} {:>10} {:>10} {:>10} {:>10}' % (filename_lens, name_lens)
        print(fmt.format('file','name','ncalls','tottime','avg time','std dev'), file=stream)

        fmt = '{:>%d} {:>%d} {:>10} {:>10.3} {:>10.3} {:>10.3}' % (filename_lens, name_lens)
        print('\n'.join(fmt.format(*tup) for tup in sorted(results)), file=stream)

### use @line_profiled for a thin wrapper around line_profiler

try:
    import line_profiler
    _prof = line_profiler.LineProfiler()

    def line_profiled(func):
        mod = inspect.getmodule(func)
        if 'PROFILING' in os.environ or (hasattr(mod,'PROFILING') and mod.PROFILING):
            return _prof(func)
        return func

    def show_line_stats(stream=None):
        if isinstance(stream,str):
            with open(stream,'w') as outfile:
                _prof.print_stats(stream=outfile)
        else:
            _prof.print_stats(stream=stream)
except ImportError:
    line_profiled = lambda x: x

