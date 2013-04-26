using namespace Eigen;

// inputs
Map<MatrixXi> eargs(args,M,T);

for (int t=1; t<T; t++) {
    stateseq[t] = eargs(stateseq[t-1],t);
}

