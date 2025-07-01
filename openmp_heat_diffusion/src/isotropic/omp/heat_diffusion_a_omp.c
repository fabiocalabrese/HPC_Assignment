#include <stdio.h>
#include <math.h>
#include <malloc.h>
#include <omp.h>


#define N 1024
#define MAX_ITER 10000
#define TOL 0.01
#define T_HOT 250.0
#define T_COLD 25.0


// function to initialize the matrix
void init_matrix(double matrix[N][N]);
// function to print the matrix
void print_matrix(double matrix[N][N]);
// function to copy matrix in matrix2
void copy_matrix(double matrix[N][N], double matrix2[N][N]);
// solver of "diffusion model"
double diffusion_matrix(double matrix[N][N],double matrix_t[N][N]);
// for exercise 'a' we set the "Dirichlet boundary condition", so the border is fixed with initial condition.
void boundary_conditions(double matrix[N][N], double matrix_2[N][N]);


int main(void) {

    int iter = 0;

    double diff = (T_HOT-T_COLD);

    // dynamic allocation of memory
    double (*matrix)[N] = malloc(N * N * sizeof(double));
    double (*matrix_t)[N] = malloc(N * N * sizeof(double));

    if (matrix == NULL || matrix_t == NULL) {
        printf("Error allocating memory for matrix\n");
        return -1;
    }

// initialization of the matrix and imposing the boundary condition
    init_matrix(matrix);

    boundary_conditions(matrix,matrix_t);
    //omp_set_num_threads(6);

    printf("Starting simulation...");

    double  t_start = omp_get_wtime();
    while( (diff > TOL) && (iter < MAX_ITER)  ) {

        // apply the solver for each iteration
        diff = diffusion_matrix(matrix,matrix_t);


        // just print every 100 some parameters
        if (iter % 100 == 0) {
            printf("Iteration %d, Max Temperature Difference: %f\n",
                   (iter+1), diff);
            printf("temperature in the centre: %f\n", matrix[N/2][N/2]);

        }

        // copy the new matrix with the new values in the older one.
        copy_matrix(matrix,matrix_t);
        iter++;

    }
    double t_stop = omp_get_wtime();

    if (iter == MAX_ITER) {
        printf("Insufficient number of iterations\n");
    }
    else{ printf("Simulation completed in %d iteration \n",iter);
          printf("Time taken %2.f seconds\n",(t_stop - t_start));
        }
    // print_matrix(matrix);  // use this command just with N small (ex. 6)

    // deallocate the memory
    free(matrix);
    free(matrix_t);
}




// half matrix at T_HOT and half at T_COLD
void init_matrix(double matrix[N][N]) {

    // we should ensure that i,j are updated in a secure way, independently one with the other
    int i,j;
    double  t_start = omp_get_wtime();
    #pragma omp parallel for private(i,j) schedule(static,10)

    for ( i = 0; i < N; i++) {
        for ( j = 0; j < N; j++) {
            if(j < N/2) {
                matrix[i][j] = T_HOT;
            }
            else {
                matrix[i][j] = T_COLD;
            }
        }
    }
    double  t_end = omp_get_wtime();
    printf("Time taken for initialization %2.f seconds\n",(t_end - t_start));
}

void print_matrix(double matrix[N][N]) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%f\t", matrix[i][j]);
        }
        printf("\n");
    }
}


// apply the equation only for all the cells (but not the border)
double diffusion_matrix(double matrix[N][N],double matrix_t[N][N]) {
    double max_diff = 0.0;

   int i,j;
   // We must ensurre that the each thread maintains its copy of the maximum difference, then it will be compared--> reduction(max:)
   #pragma omp parallel for private(i,j) reduction(max:max_diff) schedule(dynamic,4)

    for (i = 1; i < N-1; i++) {
        for (j = 1; j < N-1; j++) {
            double sum = matrix[i-1][j] + matrix[i+1][j] + matrix[i][j-1] + matrix[i][j+1];
            matrix_t[i][j] = sum/4;
            double diff = fabs(matrix_t[i][j]-matrix[i][j]); // compute the difference between the new value and the older one.
            if ( diff > max_diff) {
                max_diff = diff;   // at the end we return the largest difference, to be compared with the Tol.
            }
        }
    }

    return max_diff;
}



void copy_matrix(double matrix[N][N], double matrix2[N][N]) {

int i,j;
 #pragma omp parallel for private(i,j) schedule(static,10)

    for (int i = 1; i < N-1; i++) {
        for (int j = 1; j < N-1; j++) {
            matrix[i][j] = matrix2[i][j];
        }
    }

}

void boundary_conditions(double matrix[N][N], double matrix_2[N][N]) {
    for (int i = 0; i < N; i++) {
        matrix_2[i][0]     = matrix[i][0];       // copy the left column
        matrix_2[i][N - 1] = matrix[i][N - 1];   // copy the right column
        matrix_2[0][i]     = matrix[0][i];       // copy the upper row
        matrix_2[N - 1][i] = matrix[N - 1][i];   // copy the lower row
    }
}
