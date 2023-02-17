# Distributed Matrix Multiplication
This project implements distributed matrix multiplication using OpenMPI.

Installation
```bash 
To run this project, you will need to have OpenMPI installed on your machine. You can install it by following the instructions provided on the OpenMPI website.
```

Once you have installed OpenMPI, you can compile the program using the following command:

```bash
mpicc -o matmul matmul.c
```

Usage
To run the program, you will need to specify the size of the matrices and the number of processes to use. For example, to multiply two 1000x1000 matrices using 4 processes, you would run the following command:

Copy code
```bash
mpirun -np 4 matmul 1000
The program will then output the result of the matrix multiplication to the console.
```