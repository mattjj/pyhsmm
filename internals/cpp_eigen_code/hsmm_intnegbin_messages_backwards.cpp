using namespace Eigen;
using namespace std;

// inputs
Map<MatrixXd> eA(AT,M,M);
Map<ArrayXXd> eaBl(aBl,M,T);

// outputs
Map<ArrayXXd> ebetal(betal,rtot,T);
Map<ArrayXXd> esuperbetal(superbetal,M,T);

// locals
ArrayXd incoming(M);
ArrayXd thesum(M);
double cmax, temp;

// code!

for (int t=T-2; t>=0; t--) {
    // across-state transition part (sparse part)
    thesum = esuperbetal.col(t+1) + eaBl.col(t+1);
    cmax = thesum.maxCoeff();
    incoming = (eA * (thesum - cmax).exp().matrix()).array().log() + cmax;

    // within-state transition part (bidiagonal block-diagonal part)
    for (int idx=0; idx<M; idx++) {
        int start = start_indices[idx];
        int end = end_indices[idx];
        double pi = ps[idx];

        for (int i=start; i<end; i++) {
            cmax = max(ebetal(i,t+1),ebetal(i+1,t+1));
            ebetal(i,t) = log(pi*exp(ebetal(i,t+1)-cmax)+(1.0-pi)*exp(ebetal(i+1,t+1)-cmax))
                            + cmax + eaBl(idx,t+1);
        }
        temp = ebetal(end,t+1) + eaBl(idx,t+1);
        cmax = max(temp,incoming(idx));
        ebetal(end,t) = log(pi*exp(temp-cmax)+exp(incoming(idx)-cmax)) + cmax;

        esuperbetal(idx,t) = ebetal(start,t);
    }
}

