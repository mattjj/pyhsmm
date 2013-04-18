using namespace Eigen;

// inputs
Map<MatrixXd> eAT(A,M,M);
Map<MatrixXd> eaBl(aBl,M,T);
Map<MatrixXd> ebetal(betal,M,T);
Map<VectorXd> epi0(pi0,M);

// outputs
Map<VectorXi> estateseq(stateseq,T);

// locals
int idx, state;
double total;
VectorXd nextstate_unsmoothed(M);
VectorXd logdomain(M);
VectorXd nextstate_distr(M);

// code!
nextstate_unsmoothed = epi0;
for (idx=0; idx < T; idx++) {
    logdomain = ebetal.col(idx) + eaBl.col(idx);
    nextstate_distr = (logdomain.array() - logdomain.maxCoeff()).exp() * nextstate_unsmoothed.array();
    total = nextstate_distr.sum() * (((double)random())/((double)RAND_MAX));
    for (state = 0; (total -= nextstate_distr(state)) > 0; state++) ;
    estateseq(idx) = state;
    nextstate_unsmoothed = eAT.col(state);
}
