using namespace Eigen;

Map<MatrixXd> ebetal(betal,%(M)d,%(T)d);
Map<MatrixXd> ebetastarl(betastarl,%(M)d,%(T)d);
Map<MatrixXd> eaBl(aBl,%(M)d,%(T)d);
Map<MatrixXd> eA(A,%(M)d,%(M)d);
Map<VectorXd> epi0(pi0,%(M)d);
Map<MatrixXd> eapmf(apmf,%(T)d,%(M)d);

//MatrixXd ebetal(%(M)d,%(T)d), ebetastarl(%(M)d,%(T)d), eaBl(%(M)d,%(T)d), eA(%(M)d,%(M)d), eapmf(%(T)d,%(M)d);
//VectorXd epi0(%(M)d);

// outputs

Map<VectorXi> estateseq(stateseq,%(T)d);
//VectorXi estateseq(%(T)d);
estateseq.setZero();

// locals
int idx, state, dur;
double durprob, p_d_marg, p_d, total;
VectorXd nextstate_unsmoothed(%(M)d);
VectorXd logdomain(%(M)d);
VectorXd nextstate_distr(%(M)d);
VectorXd cumsum(%(M)d);

// code!
// don't think i need to seed... should include sys/time.h for this
// struct timeval time;
// gettimeofday(&time,NULL);
// srandom((time.tv_sec * 1000) + (time.tv_usec / 1000));

idx = 0;
nextstate_unsmoothed = epi0;

while (idx < %(T)d) {
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
        if (dur > 2*%(T)d) {
            std::cout << "FAIL" << std::endl;
        }

        p_d_marg = (dur < %(T)d) ? eapmf(dur,state) : 1.0;
        if (0.0 == p_d_marg) {
                dur += 1;
                continue;
        }
        if (idx+dur < %(T)d) {
                p_d = p_d_marg * (exp(eaBl.row(state).segment(idx,dur+1).sum() + ebetal(state,idx+dur) - ebetastarl(state,idx)));
        } else {
            break; // TODO fix this
        }
        durprob -= p_d;
        dur += 1;
    }

    estateseq.segment(idx,dur).setConstant(state);

    nextstate_unsmoothed = eA.col(state);

    idx += dur;
}

