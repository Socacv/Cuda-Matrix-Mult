#include "matrix.h"
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <unistd.h>
__device__ double dotProduct(double* first, double* second, const int m){
    double result;
    for(int i = 0; i < m; i++){
        result += first[i]*second[i];
    }
    return result;
}
__global__ void matMul(Matrix *first, Matrix *second, Matrix *third, const int lcom){
    int threadx = threadIdx.x + blockIdx.x * blockDim.x;
    int thready = threadIdx.y + blockIdx.y * blockDim.y;
    if(threadx < third->n && thready < third->m){
        (*third->atG(threadx, thready)) = dotProduct(first->getRowG(threadx),second->getRowTG(thready), lcom);
    }
}
__global__ void transpose(Matrix* matrix){
    int threadx = threadIdx.x + (blockIdx.x * blockDim.x);
    int thready = threadIdx.y + (blockIdx.y * blockDim.y);

    if(threadx < matrix->n && thready < matrix->m){
        (*matrix->atTG(threadx, thready)) = (*matrix->atG(threadx, thready));
    }
}

double randomdouble(){
    return static_cast<double>(rand()) / static_cast<double>(rand());
}
Matrix::Matrix(const int n, const int m){
    cudaMalloc(&this->matrixG, n*m*sizeof(double));
    cudaMalloc(&this->matrixTG,n*m*sizeof(double));
    this->matrixC = (double*) malloc( n*m*sizeof(double));
    this->n = n;
    this->m = m;
    cudaMalloc(&matongpu,sizeof(Matrix));
    cudaMemcpy(this->matongpu, this, sizeof(Matrix), cudaMemcpyHostToDevice);
}

Matrix::~Matrix(){
    cudaFree(this->matrixG);
    cudaFree(this->matrixTG);
    cudaFree(this->matongpu);
    free(this->matrixC);
}

void Matrix::init(){
    for(int i = 0; i < n; i++){
        for (int j = 0; j < m; j++) {
            *this->at(i, j) = randomdouble();
        }
    }
    cudaMemcpy(this->getMatrixOnGPU(), this->getMatrixOnCPU(), this->n*this->m*sizeof(double), cudaMemcpyHostToDevice);
    this->transposed();
}

void Matrix::print(){
    fprintf(stdout, "\033[1;31mMatrix:\n");
    for(int i = 0; i < n ; i++){
        fprintf(stdout, "\033[0;32m{ \033[0m");
        for (int j = 0; j < m; j++) {
            fprintf(stdout, "%f, ", *this->at(i, j));
        }
        fprintf(stdout, "\033[0;32m}\n\033[0m");
    }
}

void Matrix::sync(){
    cudaMemcpy(this->getMatrixOnCPU(), this->getMatrixOnGPU(), this->n*this->m*sizeof(double), cudaMemcpyDeviceToHost);
}
void Matrix::transposed(){
    dim3 threads = {nThreads,nThreads};
    dim3 blocks  = {this->n / nThreads +1,this->m /nThreads +1};
    transpose<<<blocks, threads>>>(this->matongpu);
    cudaDeviceSynchronize();
}
double* Matrix::at(uint32_t n, uint32_t m){
    return (this->matrixC + m) + (n*this->m);
}
double* Matrix::getRow(uint32_t n){
    return this->matrixC + (n*this->m);
}
__device__ double* Matrix::atG(uint32_t n, uint32_t m){
    return (this->matrixG + m) + (n*this->m);
}
__device__ double* Matrix::getRowG(uint32_t n){
    return this->matrixG + (n*this->m);
}
__device__ double* Matrix::atTG(uint32_t n, uint32_t m){
    return (this->matrixTG + n) + (m*this->n);
}
__device__ double* Matrix::getRowTG(uint32_t n){
    return this->matrixTG + (n*this->m);
}
double* Matrix::getTransposed(){
    return this->matrixTG;
}
double* Matrix::getMatrixOnGPU(){
    return this->matrixG;
}
double* Matrix::getMatrixOnCPU(){
    return this->matrixC;
}
Matrix* multiplyMatrix(Matrix *mat1, Matrix *mat2){
    if(mat1->m == mat2->n){
        Matrix *result = new Matrix(mat1->n,mat2->m);
        dim3 threads = {nThreads,nThreads};
        dim3 blocks  = {result->n / nThreads +1,result->m /nThreads +1};
        matMul<<<blocks,threads>>>(mat1->matongpu,mat2->matongpu,result->matongpu,mat1->m);
        cudaDeviceSynchronize();
        result->sync();
        return result;
    }else {
        return nullptr;
    }
}