using namespace Eigen;
using namespace std;

// inputs
Map<MatrixXd> eAT(A,M,M);
Map<MatrixXd> eaBl(aBl,M,T);
Map<MatrixXd> ebetal(betal,rtot,T);
Map<MatrixXd> esuperbetal(superbetal,M,T);
Map<VectorXd> epi0(pi0,M);

// locals
int t, substate, state;
double total;
VectorXd logdomain(M);
VectorXd nextstate_unsmoothed(M);
VectorXd nextstate_distr(M);

// code!
nextstate_unsmoothed = epi0;
t=0;
while (t < T) {
    // sample a new superstate
    logdomain = esuperbetal.col(t) + eaBl.col(t);
    nextstate_distr = (logdomain.array() - logdomain.maxCoeff()).exp() * nextstate_unsmoothed.array();
    total = nextstate_distr.sum() * (((double)random())/((double)RAND_MAX));
    for (state=0; (total -= nextstate_distr(state)) > 0; state++) ;

    // loop in the substates
    substate = start_indices[state];
    while ((substate <= end_indices[state]) && (t < T)) {
        substate += (((double)random())/((double)RAND_MAX)) < (1.-ps[state]);
        stateseq[t] = state;
        t += 1;
    }

    nextstate_unsmoothed = eAT.col(state);
}
