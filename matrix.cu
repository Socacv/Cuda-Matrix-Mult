#include "matrix.h"
#include <cstddef>
#include <cstdio>
#include <iostream>
#include <unistd.h>
__device__ double dotProduct(double* first, double* second, const int m){
    double result;
    for(int i = 0; i < m; i++){
        result += first[i]*second[i];
    }
    return result;
}
__global__ void matMul(double **first, double **second, double **third, const int n, const int m, const int lcom){
    int threadx = threadIdx.x + blockIdx.x * blockDim.x;
    int thready = threadIdx.y + blockIdx.y * blockDim.y;
    if(threadx < n && thready < m){
        third[threadx][thready] = dotProduct(first[threadx],second[thready], lcom);
    }
}
__global__ void transpose(double **matrix, double **result, const int n, const int m){
    int threadx = threadIdx.x + (blockIdx.x * blockDim.x);
    int thready = threadIdx.y + (blockIdx.y * blockDim.y);

    if(threadx < n && thready < m){
        result[thready][threadx] = matrix[threadx][thready];
    }
}
__global__ void allocateongpu(double **matrix, const int n, const int m){
    int threadid = threadIdx.x + blockIdx.x * blockDim.x;
    if(threadid < n){
        matrix[threadid] = (double*) malloc(m * sizeof(double));
    }
}
__global__ void freeongpu(double **matrix, const int n){
    int threadid = threadIdx.x + blockIdx.x * blockDim.x;
    if (threadid < n) {
        free(matrix[threadid]);
    }
}
__global__ void copyVectorMtV(double **matrix, double *vector, const int row, const int m){
    int threadid = threadIdx.x + blockIdx.x * blockDim.x;
    if(threadid < m){
        vector[threadid] = matrix[row][threadid];
    }
}
__global__ void copyVectorVtM(double **matrix, double *vector, const int row, const int m){
    int threadid = threadIdx.x + blockIdx.x * blockDim.x;
    if(threadid < m){
        matrix[row][threadid] = vector[threadid];
    }
}

double randomdouble(){
    return static_cast<double>(rand()) / static_cast<double>(rand());
}
void initMatrix(double ** matrix, const int n, const int m){
    for(int i = 0; i < n; i++){
        for (int j = 0; j < m; j++) {
            matrix[i][j] = randomdouble();
        }
    }
}
void printMat(double **matrix, const int n, const int m){
    fprintf(stdout, "\033[1;31mMatrix:\n");
    for(int i = 0; i < n ; i++){
        fprintf(stdout, "\033[0;32m{ \033[0m");
        for (int j = 0; j < m; j++) {
            fprintf(stdout, "%f, ", matrix[i][j]);
        }
        fprintf(stdout, "\033[0;32m}\n\033[0m");
    }
}
void allocMatGPU(double **&matrix, const int n,const int m){
    cudaMalloc(&matrix,n*sizeof(double*));
    dim3 threads = {32};
    dim3 blocks  = {n / threads.x +1};
    allocateongpu<<<blocks,threads>>>(matrix, n, m);
    cudaDeviceSynchronize();
}
void allocMatCPU(double **&matrix, const int n, const int m){
    matrix = (double**) malloc(n*sizeof(double*));
    for (int i = 0; i<n; i++) {
        matrix[i] = (double*) malloc(m*sizeof(double));
    }
}
void freeMatGPU(double **&matrix, const int n){
    dim3 threads = {32};
    dim3 blocks  = {n / threads.x +1};
    freeongpu<<<blocks,threads>>>(matrix, n);
    cudaDeviceSynchronize();
    cudaFree(matrix);    
}
void freeMatCPU(double **&matrix, const int n){
    for (int i = 0; i<n; i++) {
        free(matrix[i]);
    }
    free(matrix);    
}
void copyMatCPUtoGPU(double **&cpu, double **&gpu, const int n, const int m){
    dim3 threads = {32};
    dim3 blocks  = {n / threads.x +1};
    for(int i = 0; i < n; i++){
        double* temp;
        cudaMalloc(&temp,m*sizeof(double));
        cudaMemcpy(temp, cpu[i], m*sizeof(double), cudaMemcpyHostToDevice);
        copyVectorVtM<<<blocks,threads>>>(gpu, temp, i, m);
        cudaDeviceSynchronize();
        cudaFree(temp);
    }
}
void copyMatGPUtoCPU(double **&gpu, double **&cpu, const int n, const int m){
    dim3 threads = {32};
    dim3 blocks  = {n / threads.x + 1};
    for(int i = 0; i < n; i++){
        double *temp;
        cudaMalloc(&temp,m*sizeof(double));
        copyVectorMtV<<<blocks,threads>>>(gpu, temp, i, m);
        cudaDeviceSynchronize();
        cudaMemcpy(cpu[i], temp, m*sizeof(double), cudaMemcpyDeviceToHost);
        cudaFree(temp);
    }
}

Matrix::Matrix(const int n, const int m){
    allocMatCPU(this->matrixC, n, m);
    allocMatGPU(this->matrixG, n, m);
    allocMatGPU(this->matrixTG, m, n);
    this->n = n;
    this->m = m;
}

Matrix::~Matrix(){
    freeMatGPU(this->matrixG, this->n);
    freeMatGPU(this->matrixTG, this->m);
    freeMatCPU(this->matrixC, this->n);
}

void Matrix::init(){
    initMatrix(this->matrixC, this->n, this->m);
    copyMatCPUtoGPU(this->matrixC, this->matrixG, this->n, this->m);
    this->transposed();

}

void Matrix::print(){
    printMat(this->matrixC, this->n, this->m);
}

void Matrix::sync(){
    copyMatGPUtoCPU(this->matrixG, this->matrixC, this->n, this->m);
}
void Matrix::transposed(){
    dim3 threads = {nThreads,nThreads};
    dim3 blocks  = {this->n / nThreads +1,this->m /nThreads +1};
    transpose<<<blocks, threads>>>(this->matrixG, this->matrixTG, this->n, this->m);
}

double** Matrix::getTransposed(){
    return this->matrixTG;
}
double** Matrix::getMatrixOnGPU(){
    return this->matrixG;
}
double** Matrix::getMatrixOnCPU(){
    return this->matrixC;
}
Matrix* multiplyMatrix(Matrix *mat1, Matrix *mat2){
    if(mat1->m == mat2->n){
        Matrix * result = new Matrix(mat1->n,mat2->m);
        dim3 threads = {nThreads,nThreads};
        dim3 blocks  = {(unsigned)result->n / nThreads +1,(unsigned)result->m/nThreads +1};
        matMul<<<blocks,threads>>>(mat1->getMatrixOnGPU(), mat2->getTransposed(), result->getMatrixOnGPU(), result->n, result->m, mat1->m);
        cudaDeviceSynchronize();
        result->sync();
        return result;
    }else {
        return nullptr;
    }
}