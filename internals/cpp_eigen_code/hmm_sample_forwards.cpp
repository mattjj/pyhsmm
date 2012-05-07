using namespace Eigen;

// inputs
Map<MatrixXd> eA(A,%(M)d,%(M)d);
Map<MatrixXd> eaBl(aBl,%(M)d,T);
Map<MatrixXd> ebetal(betal,%(M)d,T);
Map<VectorXd> epi0(pi0,%(M)d);

// outputs
Map<VectorXi> estateseq(stateseq,T);

// locals
int idx, state;
double total;
VectorXd nextstate_unsmoothed(%(M)d);
VectorXd logdomain(%(M)d);
VectorXd nextstate_distr(%(M)d);

// code!
idx = 0;
nextstate_unsmoothed = epi0;

for (idx=0; idx < T; idx++) {
    logdomain = ebetal.col(idx) + eaBl.col(idx);
    nextstate_distr = (logdomain.array() - logdomain.maxCoeff()).exp() * nextstate_unsmoothed.array();
    total = nextstate_distr.sum() * (((double)random())/((double)RAND_MAX));
    for (state = 0; (total -= nextstate_distr(state)) > 0; state++) ;
    estateseq(idx) = state;
    nextstate_unsmoothed = eA.col(state);
}
