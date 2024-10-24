#include <cstdio>
#include <cstdlib>
#include <iostream>

__device__ double dotProduct(double* first, double* second, const int size){
    double result;
    for(int i = 0; i < size; i++){
        result += first[i]*second[i];
    }
    return result;
}
__global__ void matMul(double **first, double **second, double **third, const int n){
    int threadx = threadIdx.x + blockIdx.x * blockDim.x;
    int thready = threadIdx.y + blockIdx.y * blockDim.y;
    if(threadx < n && thready < n){
        third[threadx][thready] = dotProduct(first[threadx],second[thready], n);
    }
}
__global__ void transpose(double **matrix, double **result, const int n){
    int threadx = threadIdx.x + (blockIdx.x * blockDim.x);
    int thready = threadIdx.y + (blockIdx.y * blockDim.y);

    if(threadx < n && thready < n){
        result[threadx][thready] = matrix[thready][threadx];
    }
}
double randomdouble(){
    return static_cast<double>(rand()) / static_cast<double>(rand());
}
void initMatrix(double ** matrix, const int n){
    for(int i = 0; i < n; i++){
        for (int j = 0; j < n; j++) {
            matrix[i][j] = randomdouble();
        }
    }
}
void printMat(double **matrix, const int n){
    for(int i = 0; i < n ; i++){
        fprintf(stdout, "\033[0;32m{ \033[0m");
        for (int j = 0; j < n; j++) {
            fprintf(stdout, "%f, ", matrix[i][j]);
        }
        fprintf(stdout, "\033[0;32m}\n\033[0m");
    }
}
int main(){
    const int matrixsize = 64;
    double **mat1;
    double **mat2;
    double **mat2T;
    double **mat3;
// ALLOCATE
    cudaMallocManaged(&mat1,matrixsize*sizeof(double*));
    for (int i = 0; i<matrixsize; i++) {
        cudaMallocManaged(&mat1[i],matrixsize*sizeof(double));
    }
    cudaMallocManaged(&mat2,matrixsize*sizeof(double));
    for (int i = 0; i<matrixsize; i++) {
        cudaMallocManaged(&mat2[i],matrixsize*sizeof(double));
    }
    cudaMallocManaged(&mat2T,matrixsize*sizeof(double));
    for (int i = 0; i<matrixsize; i++) {
        cudaMallocManaged(&mat2T[i],matrixsize*sizeof(double));
    }
    cudaMallocManaged(&mat3,matrixsize*sizeof(double));
    for (int i = 0; i<matrixsize; i++) {
        cudaMallocManaged(&mat3[i],matrixsize*sizeof(double));
    }
//////////////////////////////////////////////////////////////////////////////////
    initMatrix(mat1, matrixsize);
    initMatrix(mat2, matrixsize);
    dim3 threads = {32,32};
    dim3 blocks  = {matrixsize / threads.x,matrixsize / threads.y};
    fprintf(stdout, "\033[1;31mFirst Matrix : \033[0m\n");
    printMat(mat1, matrixsize);
    fprintf(stdout, "\033[1;31mSecond Matrix : \033[0m\n");
    printMat(mat2, matrixsize);
    // Need to transpose for correct result due to double pointers 
    transpose<<<blocks,threads>>>(mat2, mat2T, matrixsize);
    cudaDeviceSynchronize();
    fprintf(stdout, "\033[1;31mSecond Matrix Transposed : \033[0m\n");
    printMat(mat2T, matrixsize);
    matMul<<<blocks, threads>>>(mat1, mat2T, mat3, matrixsize);

    cudaDeviceSynchronize();
    fprintf(stdout, "\033[1;31mResult : \033[0m\n");
    printMat(mat3, matrixsize);


//////////////////////////////////////////////////////////////////////////////////
// FREE
    for (int i = 0; i<matrixsize; i++) {
        cudaFree(mat1[i]);
    }
    cudaFree(mat1);
    for (int i = 0; i<matrixsize; i++) {
        cudaFree(mat2[i]);
    }
    cudaFree(mat2);
        for (int i = 0; i<matrixsize; i++) {
        cudaFree(mat2T[i]);
    }
    cudaFree(mat2T);
    for (int i = 0; i<matrixsize; i++) {
        cudaFree(mat3[i]);
    }
    cudaFree(mat3);
}


