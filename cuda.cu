#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <string>

__device__ double dotProduct(double* first, double* second, const int n){
    double result;
    for(int i = 0; i < n; i++){
        result += first[i]*second[i];
    }
    return result;
}
__global__ void matMul(double **first, double **second, double **third, const int n, const int m){
    int threadx = threadIdx.x + blockIdx.x * blockDim.x;
    int thready = threadIdx.y + blockIdx.y * blockDim.y;
    if(threadx < n && thready < n){
        third[threadx][thready] = dotProduct(first[threadx],second[thready], n);
    }
}
__global__ void transpose(double **matrix, double **result, const int n, const int m){
    int threadx = threadIdx.x + (blockIdx.x * blockDim.x);
    int thready = threadIdx.y + (blockIdx.y * blockDim.y);

    if(threadx < n && thready < m){
        result[threadx][thready] = matrix[thready][threadx];
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
int main(int argc, char* argv[]){
    int matrixsize;
    if(argc < 2){
        matrixsize = 32;
    }else{
        matrixsize = std::stoi(argv[1]);
    }
    double **mat1C;
    double **mat2C;
    double **mat3C;
    double **mat1G;
    double **mat2G;
    double **mat2TG;
    double **mat3G;
// ALLOCATE
    allocMatCPU(mat1C, matrixsize,matrixsize);
    allocMatCPU(mat2C, matrixsize,matrixsize);
    allocMatCPU(mat3C, matrixsize,matrixsize);
    allocMatGPU(mat1G, matrixsize,matrixsize);
    allocMatGPU(mat2G, matrixsize,matrixsize);
    allocMatGPU(mat2TG, matrixsize,matrixsize);
    allocMatGPU(mat3G, matrixsize,matrixsize);
//////////////////////////////////////////////////////////////////////////////////
    initMatrix(mat1C, matrixsize,matrixsize);
    initMatrix(mat2C, matrixsize,matrixsize);
    copyMatCPUtoGPU(mat1C, mat1G, matrixsize,matrixsize);
    copyMatCPUtoGPU(mat2C, mat2G, matrixsize,matrixsize);
    dim3 threads = {32,32};
    dim3 blocks  = {matrixsize / threads.x + 1,matrixsize / threads.y + 1};
    transpose<<<blocks,threads>>>(mat2G, mat2TG, matrixsize,matrixsize);
    cudaDeviceSynchronize();
    matMul<<<blocks, threads>>>(mat1G, mat2TG, mat3G, matrixsize,matrixsize);

    cudaDeviceSynchronize();
    copyMatGPUtoCPU(mat3G, mat3C, matrixsize,matrixsize);
    fprintf(stdout, "\033[1;31mResult: \n");
    printMat(mat3C, matrixsize,matrixsize);

//////////////////////////////////////////////////////////////////////////////////
// FREE
    freeMatCPU(mat1C, matrixsize);
    freeMatCPU(mat2C, matrixsize);
    freeMatCPU(mat3C, matrixsize);
    freeMatGPU(mat1G, matrixsize);
    freeMatGPU(mat2G, matrixsize);
    freeMatGPU(mat2TG, matrixsize);
    freeMatGPU(mat3G, matrixsize);


}
