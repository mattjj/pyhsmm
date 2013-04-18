using namespace Eigen;
using namespace std;

// inputs
Map<MatrixXd> eAT(A,M,M);
Map<MatrixXd> eaBl(aBl,M,T);
Map<MatrixXd> ebetal(betal,rtot,T);
Map<VectorXd> epi0(pi0,M);

// locals
int t, substate;
double total;

// code!
nextstate_unsmoothed = epi0;
t=0;
while (t < T) {
    // sample a new superstate
    logdomain = 
    logdomain = (logdomain.array

    // loop in the substates
}
