using namespace Eigen;
using namespace std;
// inputs
int etrunc = mytrunc;
Map<MatrixXd> eaBl(aBl,%(M)d,%(T)d);
Map<MatrixXd> eA(A,%(M)d,%(M)d);
Map<MatrixXd> eaDl(aDl,%(M)d,%(T)d);
Map<MatrixXd> eaDsl(aDsl,%(M)d,%(T)d);

// outputs
Map<MatrixXd> ebetal(betal,%(M)d,%(T)d);
Map<MatrixXd> ebetastarl(betastarl,%(M)d,%(T)d);

// locals
VectorXd maxes(%(M)d), result(%(M)d), sumsofar(%(M)d);
double cmax;

// computation!
for (int t = %(T)d-1; t >= 0; t--) {
    sumsofar.setZero();
    ebetastarl.col(t).setConstant(-1.0*numeric_limits<double>::infinity());
    for (int tau = 0; tau < min(etrunc,%(T)d-t); tau++) {
        sumsofar += eaBl.col(t+tau);
        result = ebetal.col(t+tau) + sumsofar + eaDl.col(tau);
        maxes = ebetastarl.col(t).cwiseMax(result);
        ebetastarl.col(t) = ((ebetastarl.col(t) - maxes).array().exp() + (result - maxes).array().exp()).log() + maxes.array();
    }
    // censoring calc
    if (%(T)d - t < etrunc) {
        result = eaBl.block(0,t,%(M)d,%(T)d-t).rowwise().sum() + eaDsl.col(%(T)d-1-t);
        maxes = ebetastarl.col(t).cwiseMax(result);
        ebetastarl.col(t) = ((ebetastarl.col(t) - maxes).array().exp() + (result - maxes).array().exp()).log() + maxes.array();
    }
    // nan issue
    for (int i = 0; i < %(M)d; i++) {
        if (ebetastarl(i,t) != ebetastarl(i,t)) {
            ebetastarl(i,t) = -1.0*numeric_limits<double>::infinity();
        }
    }
    // betal calc
    if (t > 0) {
        cmax = ebetastarl.col(t).maxCoeff();
        ebetal.col(t-1) = (eA * (ebetastarl.col(t).array() - cmax).array().exp().matrix()).array().log() + cmax;
        for (int i = 0; i < %(M)d; i++) {
            if (ebetal(i,t-1) != ebetal(i,t-1)) {
                ebetal(i,t-1) = -1.0*numeric_limits<double>::infinity();
            }
        }
    }
}


