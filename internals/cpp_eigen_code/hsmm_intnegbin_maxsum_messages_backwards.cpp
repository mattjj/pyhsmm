using namespace Eigen;
using namespace std;

// inputs
Map<MatrixXd> eAlT(Al,M,M);
Map<MatrixXd> eaBl(aBl,M,T);

// outputs
Map<ArrayXXd> escores(scores,rtot,T);
Map<ArrayXXi> eargs(args,rtot,T);

// locals


// code!

for (int t=T-2; t>=0; t--) {
    for (int superstate=0; superstate<M; superstate++) {
        int start = start_indices[superstate];
        int end = end_indices[superstate];
        double logp = logps[superstate];
        double log1mp = log1mps[superstate];

        // within-state (block bidiagonal)
        for (int substate=start; substate<end; substate++) {
            double self = logp + escores(substate,t+1) + eaBl(superstate,t+1);
            double next = log1mp + escores(substate+1,t+1) + eaBl(superstate,t+1);
            if (self > next) {
                escores(substate,t) = self;
                eargs(substate,t+1) = substate;
            } else {
                escores(substate,t) = next;
                eargs(substate,t+1) = substate+1;
            }
        }

        // across states
        double score = logp + escores(end,t+1) + eaBl(superstate,t+1);
        int nextindex = end;
        for (int nextsuperstate=0; nextsuperstate<M; nextsuperstate++) {
            double nextscore = eaBl(nextsuperstate,t+1) + eAlT(nextsuperstate,superstate)
                + escores(start_indices[nextsuperstate],t+1);
            if (nextscore > score) {
                score = nextscore;
                nextindex = end_indices[nextsuperstate];
            }
        }
        escores(end,t) = score;
        eargs(end,t+1) = nextindex;
    }
}
