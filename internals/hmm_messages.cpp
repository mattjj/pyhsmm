#include "hmm_messages.h"
using namespace Eigen;
using namespace std;

// NOTE: numpy arrays are row-major by default, while Eigen is column-major; I
// worked with each's deafult alignment, so the notion of "row" and "column"
// get transposed here compared to numpy code

// Messages

void hmm::messages_backwards_log(int M, int T, double *A, double *aBl, double *betal)
{
    // inputs
    Map<MatrixXd> eAT(A,M,M);
    Map<MatrixXd> eaBl(aBl,M,T);

    // outputs
    Map<MatrixXd> ebetal(betal,M,T);

    // locals
    MatrixXd eA(M,M);
    eA = eAT.transpose();
    VectorXd thesum(M);
    double cmax;

    // computation!
    ebetal.col(T-1).setZero();
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
}

void hmm::messages_forwards_log(int M, int T, double *A, double *pi0, double *aBl, double *alphal)
{
    // inputs
    Map<MatrixXd> eAT(A,M,M);
    Map<ArrayXd> epi0(pi0,M);
    Map<MatrixXd> eaBl(aBl,M,T);

    // outputs
    Map<MatrixXd> ealphal(alphal,M,T);

    // locals
    double cmax;

    // computation!
    ealphal.col(0) = epi0.log() + eaBl.col(0).array();
    for (int t=0; t<T-1; t++) {
        cmax = ealphal.col(t).maxCoeff();
        ealphal.col(t+1) = (eAT * (ealphal.col(t).array()
                    - cmax).array().exp().matrix()).array().log()
            + cmax + eaBl.col(t+1).array();
        /* // nan issue (is there a better way to do this?)
        for (int i=0; i<M; i++) {
            if (ealphal(i,t+1) != ealphal(i,t+1)) {
                ealphal(i,t+1) = -1.0*numeric_limits<double>::infinity();
            }
        } */
    }
}

double hmm::messages_forwards_normalized(int M, int T, double *A, double *pi0, double *aBl,
        double *alphan)
{
    Map<MatrixXd> eAT(A,M,M);
    Map<MatrixXd> eaBl(aBl,M,T);
    Map<VectorXd> epi0(pi0,M);

    Map<ArrayXd> ealphan(alphan,M,T);

    double logtot = 0.;
    double cmax, norm;

    cmax = eaBl.col(0).maxCoeff();
    ealphan.col(0) = epi0.array() * (eaBl.col(0).array() - cmax).exp();
    norm = ealphan.col(0).sum();
    ealphan.col(0) /= norm;
    logtot += log(norm) + cmax;
    for (int t=0; t<T; t++) {
        cmax = eaBl.col(t+1).maxCoeff();
        ealphan.col(t+1) = (eAT * ealphan.col(t).matrix()).array()
            * (eaBl.col(t+1).array() - cmax).exp();
        norm = ealphan.col(t+1).sum();
        ealphan.col(t+1) /= norm;
        logtot += log(norm) + cmax;
    }

    return logtot;
}

// Sampling

void hmm::sample_forwards_log(
        int M, int T, double *A, double *pi0, double *aBl,
        double *betal, int32_t *stateseq)
{
    // inputs
    Map<MatrixXd> eAT(A,M,M);
    Map<MatrixXd> eaBl(aBl,M,T);
    Map<MatrixXd> ebetal(betal,M,T);
    Map<VectorXd> epi0(pi0,M);

    // locals
    int idx;
    VectorXd nextstate_unsmoothed(M);
    VectorXd logdomain(M);
    VectorXd nextstate_distr(M);

    // code!
    nextstate_unsmoothed = epi0;
    for (idx=0; idx < T; idx++) {
        logdomain = ebetal.col(idx) + eaBl.col(idx);
        nextstate_distr = (logdomain.array() - logdomain.maxCoeff()).exp()
            * nextstate_unsmoothed.array();
        stateseq[idx] = util::sample_discrete(M,nextstate_distr.data());
        nextstate_unsmoothed = eAT.col(stateseq[idx]);
    }
}

void hmm::sample_backwards_normalized(int M, int T, double *A, double *alphan,
        int32_t *stateseq)
{
    Map<MatrixXd> eAT(A,M,M);
    double Aarr[M*M];
    Map<MatrixXd> eA(Aarr,M,M);
    eA = eAT.transpose();
    Map<MatrixXd> ealphan(alphan,M,T);

    double next_potential[M];
    Map<Array<double,Dynamic,1>,Aligned> enext_potential(next_potential,M);
    enext_potential.setOnes();

    double temp[M];
    Map<Array<double,Dynamic,1>,Aligned> etemp(temp,M);

    for (int t=T-1; t>=0; t--) {
        etemp = enext_potential * ealphan.col(t).array();
        stateseq[t] = util::sample_discrete(M,temp);
        enext_potential = eA.col(stateseq[t]);
    }
}

// Viterbi

void hmm::viterbi(
        int M, int T, double *A, double *pi0, double *aBl,
        int32_t *stateseq)
{
    // inputs
    Map<MatrixXd> eA(A,M,M);
    Map<MatrixXd> eaBl(aBl,M,T);
    Map<ArrayXd> epi0(pi0,M);

    // locals
    MatrixXi args(M,T);
    MatrixXd eAl(M,M);
    eAl = eA.array().log();
    VectorXd scores(M);
    VectorXd prevscores(M);
    VectorXd tempvec(M);
    VectorXd::Index maxIndex;

    // computation!
    scores.setZero();
    for (int t=T-2; t>=0; t--) {
        for (int i=0; i<M; i++) {
            tempvec = eAl.col(i) + scores + eaBl.col(t+1);
            prevscores(i) = tempvec.maxCoeff(&maxIndex);
            args(i,t+1) = maxIndex;
        }
        scores = prevscores;
    }

    (scores.array() + epi0.log() + eaBl.col(0).array()).maxCoeff(stateseq);
    for (int t=1; t<T; t++) {
        stateseq[t] = args(stateseq[t-1],t);
    }
}
