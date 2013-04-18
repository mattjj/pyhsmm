using namespace Eigen;
using namespace std;

/***********
 *  Setup  *
 ***********/

// inputs
Map<MatrixXd> eA(AT,M,M);
Map<ArrayXXd> eaBl(aBl,M,T);

// outputs
Map<ArrayXXd> ebetal(betal,rtot,T);

// locals
ArrayXd incoming(M);
ArrayXd thesum(M);
double cmax;

/*****************
 *  Computation  *
 *****************/

for (int t=T-2; t>=0; t--) {
    // across-state transition part (sparse part)
    for (int i=0; i<M; i++) {
        thesum(i) = ebetal(start_indices[i],t+1);
    }
    thesum += eaBl.col(t+1);
    cmax = thesum.maxCoeff();
    incoming = (eA * (thesum - cmax).exp().matrix()).array().log() + cmax;

    // within-state transition part (block-diagonal part)
    for (int idx=0; idx<M; idx++) {
        int start = start_indices[idx];
        int end = end_indices[idx];
        double pi = ps[idx];
        for (int i=start; i<end; i++) {
            cmax = max(ebetal(i,t+1),ebetal(i+1,t+1));
            ebetal(i,t) = log(pi*exp(ebetal(i,t+1)-cmax)+(1.0-pi)*exp(ebetal(i+1,t+1)-cmax))
                            + cmax + eaBl(idx,t+1);
        }
        double thing = ebetal(end,t+1) + eaBl(idx,t+1);
        cmax = max(thing,incoming(idx));
        ebetal(end,t) = log(pi*exp(thing-cmax)+exp(incoming(idx)-cmax)) + cmax;
    }
}

