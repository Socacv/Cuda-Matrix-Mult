#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "matrix.h"
#include <unistd.h>

int main(int argc, char* argv[]){
    Matrix* mat1 = new Matrix(1,400);
    Matrix* mat2 = new Matrix(400,1);
    mat1->init();
    mat2->init();
    mat1->print();
    mat2->print();
    Matrix* mat3 = multiplyMatrix(mat1, mat2);
    if(mat3 == nullptr){
        fprintf(stderr, "Error mismatch of columns and rows\n");
        delete mat1;
        delete mat2;
        return -1;
    }
    mat3->print();

    delete mat1;
    delete mat2;
    delete mat3;
    return 0;
}
