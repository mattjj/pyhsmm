using namespace Eigen;

// inputs
Map<MatrixXd> eAT(A,%(M)d,%(M)d);
Map<MatrixXd> eaBl(aBl,%(M)d,%(T)d);

// outputs
Map<MatrixXd> ealphal(alphal,%(M)d,%(T)d);

// locals
double cmax;

// computation!
for (int t=0; t<T-1; t++) {
    cmax = ealphal.col(t).maxCoeff();
    ealphal.col(t+1) = (eAT * (ealphal.col(t).array() - cmax).array().exp().matrix()).array().log() + cmax + eaBl.col(t+1).array();
    /* // nan issue (is there a better way to do this?)
    for (int i=0; i<%(M)d; i++) {
        if (ealphal(i,t+1) != ealphal(i,t+1)) {
            ealphal(i,t+1) = -1.0*numeric_limits<double>::infinity();
        }
    } */
}

