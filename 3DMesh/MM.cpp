#include <iostream>
#include <mpi.h>

#define N 100 // matrix size
#define P 8 // number of processes

using namespace std;

int main(int argc, char *argv[]) {
    int rank, size, i, j, k, rows, offset;
    double A[N][N], B[N][N], C[N][N];
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (size != P) {
        if (rank == 0) {
            cout << "Error: number of processes must be " << P << endl;
        }
        MPI_Finalize();
        return 1;
    }
    rows = N / P;
    double subA[rows][N], subB[N][rows], subC[rows][rows];
    // scatter matrix A
    if (rank == 0) {
        for (i = 0; i < N; i++) {
            for (j = 0; j < N; j++) {
                A[i][j] = i * j;
            }
        }
    }
    MPI_Scatter(A, rows * N, MPI_DOUBLE, subA, rows * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    // broadcast matrix B
    if (rank == 0) {
        for (i = 0; i < N; i++) {
            for (j = 0; j < N; j++) {
                B[i][j] = i + j;
            }
        }
    }
    MPI_Bcast(B, N * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    // multiply matrices
    for (i = 0; i < rows; i++) {
        for (j = 0; j < rows; j++) {
            subC[i][j] = 0.0;
            for (k = 0; k < N; k++) {
                subC[i][j] += subA[i][k] * B[k][j];
            }
        }
    }
    // gather results
    offset = rows * rank;
    MPI_Gather(subC, rows * rows, MPI_DOUBLE, C + offset * N, rows * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    // print result
    if (rank == 0) {
        for (i = 0; i < N; i++) {
            for (j = 0; j < N; j++) {
                cout << C[i][j] << " ";
            }
            cout << endl;
        }
    }
    MPI_Finalize();
    return 0;
}