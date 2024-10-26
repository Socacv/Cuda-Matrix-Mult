#include <cstdint>
__global__ void matMul(const double *first, const double *second, double **third, const int n, const int m, const int lcom);
__global__ void transpose(const double *matrix, double *result, const int n, const int m);
__global__ void copyVectorMtV(double *matrix, double *vector, const int row, const int m);
__global__ void copyVectorVtM(double *matrix, double *vector, const int row, const int m);
double randomdouble();
void initMatrix(double *matrix, const int n, const int m);
void printMat(double *matrix, const int n, const int m);
const int nThreads = 32;
class Matrix {
public:
    
    Matrix(const int n, const int m);
    ~Matrix();
    void init();
    void print();
    void sync();
    void transposed();
    double* at(uint32_t n, uint32_t m);
    double* getRow(uint32_t n);
    __device__ double* atG(uint32_t n, uint32_t m);
    __device__ double* getRowG(uint32_t n);
    __device__ double* atTG(uint32_t n, uint32_t m);
    __device__ double* getRowTG(uint32_t n);
    double* getTransposed();
    double* getMatrixOnGPU();
    double* getMatrixOnCPU();
    unsigned int n = 32, m = 32;
    double* matrixC;
    double* matrixG;
    double* matrixTG;
    Matrix* matongpu;
};

Matrix* multiplyMatrix(Matrix *mat1, Matrix *mat2);