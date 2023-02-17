#include <mpi.h>
#include <iostream>
#include <vector>

using namespace std;

// Matrix size
#define N 1000

// Define the torus dimensions
#define DIM 2

// Define the process topology
#define TOPO MPI_CART

int main(int argc, char** argv) {

    int rank, size;
    MPI_Comm comm_cart;
    int dims[DIM] = {0};
    int periods[DIM] = {0};
    int coords[DIM] = {0};

    // Initialize MPI
    MPI_Init(&argc, &argv);

    // Get the rank and size of the process
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Create the Cartesian topology
    MPI_Dims_create(size, DIM, dims);
    MPI_Cart_create(MPI_COMM_WORLD, DIM, dims, periods, TOPO, &comm_cart);
    MPI_Cart_coords(comm_cart, rank, DIM, coords);

    // Create the matrices A, B, and C
    vector<double> A(N*N, 1);
    vector<double> B(N*N, 1);
    vector<double> C(N*N, 0);

    // Initialize matrix B to the identity matrix
    for(int i=0; i<N; i++) {
        for(int j=0; j<N; j++) {
            if(i == j) {
                B[i*N+j] = 1.0;
            }
        }
    }

    // Scatter matrix A to all processes
    int local_n = N / dims[0];
    int local_m = N / dims[1];
    vector<double> local_A(local_n*local_m);
    MPI_Scatter(&A[0], local_n*local_m, MPI_DOUBLE, &local_A[0], local_n*local_m, MPI_DOUBLE, 0, comm_cart);

    // Broadcast matrix B to all processes
    MPI_Bcast(&B[0], N*N, MPI_DOUBLE, 0, comm_cart);

    // Perform the local matrix multiplication
    for(int i=0; i<local_n; i++) {
        for(int j=0; j<N; j++) {
            for(int k=0; k<local_m; k++) {
                C[i*N+j] += local_A[i*local_m+k] * B[k*N+j];
            }
        }
    }

    // Gather the results
    MPI_Gather(&C[0], local_n*local_m, MPI_DOUBLE, &C[0], local_n*local_m, MPI_DOUBLE, 0, comm_cart);

    // Print the results from the root process
    if(rank == 0) {
        for(int i=0; i<N; i++) {
            for(int j=0; j<N; j++) {
                cout << C[i*N+j] << " ";
            }
            cout << endl;
        }
    }

    // Clean up
    MPI_Comm_free(&comm_cart);
    MPI_Finalize();

    return 0;
}