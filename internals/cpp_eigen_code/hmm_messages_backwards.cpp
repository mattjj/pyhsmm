using namespace Eigen;

// inputs
Map<MatrixXd> eA(AT,M,M);
Map<MatrixXd> eaBl(aBl,M,T);

// outputs
Map<MatrixXd> ebetal(betal,M,T);

// locals
VectorXd thesum(M);
double cmax;

// computation!
for (int t=T-2; t>=0; t--) {
    thesum = eaBl.col(t+1) + ebetal.col(t+1);
    cmax = thesum.maxCoeff();
    ebetal.col(t) = (eA * (thesum.array() - cmax).exp().matrix()).array().log() + cmax;
    /* // nan issue (is there a better way to do this?)
    for (int i=0; i<M; i++) {
        if (ealphal(i,t+1) != ealphal(i,t+1)) {
            ealphal(i,t+1) = -1.0*numeric_limits<double>::infinity();
        }
    } */
}

