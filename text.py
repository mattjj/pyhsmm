import numpy as np
import sys, time

# time.clock() is cpu time of current process
# time.time() is wall time

# to see what this does, try
# for x in progprint_xrange(100):
#     time.sleep(0.01)

# TODO there are probably better progress bar libraries I could use

def progprint_xrange(*args,**kwargs):
    xr = xrange(*args)
    return progprint(xr,total=len(xr),**kwargs)

def progprint(iterator,total=None,perline=25,show_times=True):
    times = []
    idx = 0
    if total is not None:
        numdigits = len('%d' % total)
    for thing in iterator:
        prev_time = time.time()
        yield thing
        times.append(time.time() - prev_time)
        sys.stdout.write('.')
        if (idx+1) % perline == 0:
            if show_times:
                avgtime = np.mean(times)
                if total is not None:
                    sys.stdout.write(('  [ %%%dd/%%%dd, %%7.2fsec avg, %%7.2fsec ETA ]\n' % (numdigits,numdigits)) % (idx+1,total,avgtime,avgtime*(total-(idx+1))))
                else:
                    sys.stdout.write('  [ %d done, %7.2fsec avg ]\n' % (idx+1,avgtime))
            else:
                if total is not None:
                    sys.stdout.write(('  [ %%%dd/%%%dd ]\n' % (numdigits,numdigits) ) % (idx+1,total))
                else:
                    sys.stdout.write('  [ %d ]\n' % (idx+1))
        idx += 1
        sys.stdout.flush()
    print ''
    if show_times:
        print '%7.2fsec avg, %7.2fsec total\n' % (np.mean(times),np.sum(times))
