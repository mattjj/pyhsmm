using namespace Eigen;

// inputs
Map<MatrixXd> eAl(Al,M,M);
Map<MatrixXd> eaBl(aBl,M,T);

// outputs
Map<MatrixXd> escores(scores,M,T);
Map<MatrixXi> eargs(args,M,T);

// locals
VectorXd tempvec(M);
VectorXd::Index maxIndex;

// computation!
for (int t=T-2; t>=0; t--) {
    for (int i=0; i<M; i++) {
        tempvec = eAl.col(i) + escores.col(t+1) + eaBl.col(t+1);
        escores(i,t) = tempvec.maxCoeff(&maxIndex);
        eargs(i,t+1) = maxIndex;
    }
}

