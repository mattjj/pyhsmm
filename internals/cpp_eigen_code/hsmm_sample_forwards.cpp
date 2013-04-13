using namespace Eigen;

// inputs

Map<MatrixXd> ebetal(betal,M,T);
Map<MatrixXd> ebetastarl(betastarl,M,T);
Map<MatrixXd> eaBl(aBl,M,T);
Map<MatrixXd> eA(A,M,M);
Map<VectorXd> epi0(pi0,M);
Map<MatrixXd> eapmf(apmf,M,T);

// outputs

Map<VectorXi> estateseq(stateseq,T);
//VectorXi estateseq(T);
estateseq.setZero();

// locals
int idx, state, dur;
double durprob, p_d_marg, p_d, total;
VectorXd nextstate_unsmoothed(M);
VectorXd logdomain(M);
VectorXd nextstate_distr(M);

// code!
// don't think i need to seed... should include sys/time.h for this
// struct timeval time;
// gettimeofday(&time,NULL);
// srandom((time.tv_sec * 1000) + (time.tv_usec / 1000));

idx = 0;
nextstate_unsmoothed = epi0;

while (idx < T) {
    logdomain = ebetastarl.col(idx).array() - ebetastarl.col(idx).maxCoeff();
    nextstate_distr = logdomain.array().exp() * nextstate_unsmoothed.array();
    if ((nextstate_distr.array() == 0.0).all()) {
        std::cout << "Warning: this is a cryptic error message" << std::endl;
        nextstate_distr = logdomain.array().exp();
    }
    // sample from nextstate_distr
    {
        total = nextstate_distr.sum() * (((double)random())/((double)RAND_MAX));
        for (state = 0; (total -= nextstate_distr(state)) > 0; state++) ;
    }

    durprob = ((double)random())/((double)RAND_MAX);
    dur = 0;
    while (durprob > 0.0) {
        if (dur > 2*T) {
            std::cout << "FAIL" << std::endl;
        }

        p_d_marg = (dur < T) ? eapmf(state,dur) : 1.0;
        if (0.0 == p_d_marg) {
                dur += 1;
                continue;
        }
        if (idx+dur < T) {
                p_d = p_d_marg * (exp(eaBl.row(state).segment(idx,dur+1).sum()
                            + ebetal(state,idx+dur) - ebetastarl(state,idx)));
        } else {
            break; // will be fixed in python
        }
        durprob -= p_d;
        dur += 1;
    }

    estateseq.segment(idx,dur).setConstant(state);

    nextstate_unsmoothed = eA.col(state);

    idx += dur;
}

