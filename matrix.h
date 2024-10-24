
__global__ void matMul(double **first, double **second, double **third, const int n, const int m, const int lcom);
__global__ void transpose(double **matrix, double **result, const int n, const int m);
__global__ void allocateongpu(double **matrix, const int n, const int m);
__global__ void freeongpu(double **matrix, const int n);
__global__ void copyVectorMtV(double **matrix, double *vector, const int row, const int m);
__global__ void copyVectorVtM(double **matrix, double *vector, const int row, const int m);
double randomdouble();
void initMatrix(double ** matrix, const int n, const int m);
void printMat(double **matrix, const int n, const int m);
void allocMatGPU(double **&matrix, const int n,const int m);
void allocMatCPU(double **&matrix, const int n, const int m);
void freeMatGPU(double **&matrix, const int n);
void freeMatCPU(double **&matrix, const int n);
void copyMatCPUtoGPU(double **&cpu, double **&gpu, const int n, const int m);
void copyMatGPUtoCPU(double **&gpu, double **&cpu, const int n, const int m);
const int nThreads = 32;
class Matrix {
public:
    
    Matrix(const int n, const int m);
    ~Matrix();
    void init();
    void print();
    void sync();
    void transposed();
    double** getTransposed();
    double** getMatrixOnGPU();
    double** getMatrixOnCPU();
    unsigned int n = 32, m = 32;
    double** matrixC;
    double** matrixG;
    double** matrixTG;
};

Matrix* multiplyMatrix(Matrix *mat1, Matrix *mat2);