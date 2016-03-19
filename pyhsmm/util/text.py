from builtins import range
import numpy as np
import sys, time

# time.clock() is cpu time of current process
# time.time() is wall time

# TODO there are probably better progress bar libraries I could use

round = (lambda x: lambda y: int(x(y)))(round)

# NOTE: datetime.timedelta.__str__ doesn't allow formatting the number of digits
def sec2str(seconds):
    hours, rem = divmod(seconds,3600)
    minutes, seconds = divmod(rem,60)
    if hours > 0:
        return '%02d:%02d:%02d' % (hours,minutes,round(seconds))
    elif minutes > 0:
        return '%02d:%02d' % (minutes,round(seconds))
    else:
        return '%0.2f' % seconds

def progprint_xrange(*args,**kwargs):
    xr = range(*args)
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
                    eta = sec2str(avgtime*(total-(idx+1)))
                    sys.stdout.write((
                        '  [ %%%dd/%%%dd, %%7.2fsec avg, ETA %%s ]\n'
                                % (numdigits,numdigits)) % (idx+1,total,avgtime,eta))
                else:
                    sys.stdout.write('  [ %d done, %7.2fsec avg ]\n' % (idx+1,avgtime))
            else:
                if total is not None:
                    sys.stdout.write(('  [ %%%dd/%%%dd ]\n' % (numdigits,numdigits) ) % (idx+1,total))
                else:
                    sys.stdout.write('  [ %d ]\n' % (idx+1))
        idx += 1
        sys.stdout.flush()
    print('')
    if show_times and len(times) > 0:
        total = sec2str(seconds=np.sum(times))
        print('%7.2fsec avg, %s total\n' % (np.mean(times),total))

