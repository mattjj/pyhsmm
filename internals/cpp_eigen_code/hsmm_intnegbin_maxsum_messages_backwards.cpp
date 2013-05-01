using namespace Eigen;
using namespace std;

// inputs
Map<ArrayXXd> eAlT(Al,M,M);
Map<ArrayXXd> eaBl(aBl,M,T);
Map<ArrayXd> ebinoms(binoms,rtot);

// outputs
Map<ArrayXXd> escores(scores,rtot,T);
Map<ArrayXXi> eargs(args,rtot,T);

// locals
ArrayXd temp(rtot);

for (int t=T-2; t>=0; t--) {
    for (int superstate=0; superstate<M; superstate++) {
        int start = start_indices[superstate];
        int end = end_indices[superstate];
        double logp = logps[superstate];
        double log1mp = log1mps[superstate];

        // within-state
        for (int substate=start; substate<end; substate++) {
            double self = logp + escores(substate,t+1) + eaBl(superstate,t+1);
            double next = log1mp + escores(substate+1,t+1) + eaBl(superstate,t+1);
            if (self >= next) {
                escores(substate,t) = self;
                eargs(substate,t+1) = substate;
            } else {
                escores(substate,t) = next;
                eargs(substate,t+1) = substate+1;
            }
        }

        // across states
        for (int nextstate=0; nextstate<M; nextstate++) {
            int nextstart = start_indices[nextstate];
            int len = end_indices[nextstate] - start_indices[nextstate] + 1;
            temp.segment(nextstart,len) = ebinoms.segment(nextstart,len) + eAlT(nextstate,superstate)
                + escores.col(t+1).segment(nextstart,len) + eaBl(nextstate,t+1);
        }
        ArrayXd::Index idx;
        escores(end,t) = temp.maxCoeff(&idx);
        eargs(end,t+1) = idx;

        double selfscore = logp + escores(end,t+1) + eaBl(superstate,t+1);
        if (selfscore >= escores(end,t)) {
            escores(end,t) = selfscore;
            eargs(end,t+1) = end;
        }
    }
}
