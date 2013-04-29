using namespace Eigen;
using namespace std;

// inputs
Map<ArrayXXi> eargs(args,rtot,T);

// locals
int last_hmm_state = initial_hmm_state;

for (int t=1; t<T; t++) {
    last_hmm_state = eargs(last_hmm_state,t);
    stateseq[t] = themap[last_hmm_state];
}
