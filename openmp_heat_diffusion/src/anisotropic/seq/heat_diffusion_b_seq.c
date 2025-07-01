#include <stdio.h>
#include <math.h>
#include <stdlib.h>


#define N 6
#define MAX_ITER 10000
#define TOL 0.01
#define T_HOT2 540.0
#define T_COLD 25.0
#define WX 0.3
#define WY 0.2


void init_matrix_2(double matrix[N][N]);
void print_matrix(double matrix[N][N]);
void copy_matrix(double matrix[N][N], double matrix2[N][N]);
double diffusion_matrix_2(double matrix[N][N],double matrix_t[N][N]);
void boundary_conditions(double matrix_t[N][N]);


int main(void) {
    int iter = 0;
    double diff = (T_HOT2-T_COLD);
    double (*matrix)[N] = malloc(N * N * sizeof(double));
    double (*matrix_t)[N] = malloc(N * N * sizeof(double));

    if (matrix == NULL || matrix_t == NULL) {
        printf("Error allocating memory for matrix\n");
        return -1;
    }


    init_matrix_2(matrix);


    printf("starting simulation...\n");

    while( (diff > TOL) && (iter < MAX_ITER)  ) {

        diff = diffusion_matrix_2(matrix,matrix_t);

        if (iter % 100 == 0) {
            printf("Iteration %d, Max Temperature Difference: %f\n",
                   (iter+1), diff);
            printf("temperature in the centre: %f\n", matrix[N/2][N/2]);

        }
        // each iteration we copy the internal border in the external one.
        boundary_conditions(matrix_t);
        
        copy_matrix(matrix,matrix_t);
        iter++;

    }
    if (iter == MAX_ITER) {
        printf("Insufficient number of iterations\n");
    }
    else{ printf("Operation complete in %d iteration \n",iter); }

    print_matrix(matrix);

    free(matrix);
    free(matrix_t);
}




// central part (25%) is at T_HOT2 and the remaining part at T_COLD
void init_matrix_2(double matrix[N][N]) {
    int quarter = N / 4;
    int half = N / 2;

    int start = half - quarter;
    int end = half + quarter;

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            if (i < start || i >= end || j < start || j >= end) {
                matrix[i][j] = T_COLD;
            } else {
                matrix[i][j] = T_HOT2;
            }
        }
    }
}


void print_matrix(double matrix[N][N]) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%f\t", matrix[i][j]);
        }
        printf("\n");
    }
}



double diffusion_matrix_2(double matrix[N][N],double matrix_t[N][N]) {
    double max_diff = 0.0;
    for (int i = 1; i < N-1; i++) {
        for (int j = 1; j < N-1; j++) {

            matrix_t[i][j] = (WY*(matrix[i-1][j] + matrix[i+1][j]) + WX*(matrix[i][j-1] + matrix[i][j+1]))/(2*(WX+WY));

            double diff = fabs(matrix_t[i][j]-matrix[i][j]);
            if ( diff > max_diff) {
                max_diff = diff;
            }
        }
    }
    return max_diff;
}

void copy_matrix(double matrix[N][N], double matrix2[N][N]) {
    for (int i = 1; i < N-1; i++) {
        for (int j = 1; j < N-1; j++) {
            matrix[i][j] = matrix2[i][j];
        }
    }
}

// Neumann condition , copy the internal border in the external border each iteration
void boundary_conditions(double matrix_t[N][N]) {
    for (int i = 0; i < N; i++) {
        matrix_t[i][0]     = matrix_t[i][1];
        matrix_t[i][N - 1] = matrix_t[i][N - 2];
        matrix_t[0][i]     = matrix_t[1][i];
        matrix_t[N - 1][i] = matrix_t[N - 2][i];
    }
}

