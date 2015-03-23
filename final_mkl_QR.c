/******************************************************************************/
/*
  Purpose:

	QR decomposition of Matrix using MKL library
	
  Description:
  
	The program computes a QR factorization of a real m-by-n matrix A as A = Q*R.
	The program does not form the matrix Q explicitly. Instead, Q is represented as a product of min(m, n)
	elementary reflectors :
	Q = H(1)*H(2)* ... *H(k), where k = min(m, n)
	Each H(i) has the form
	H(i) = I - tau*v*vT for real flavors
	where tau is a real scalar stored in tau(i), and v is a real vector with v(1:i-1) = 0 and
	v(i) = 1.
	
	On exit, v(i+1:m) is stored in a(i+1:m, i).
	
	on exit, the elements on and above the diagonal of the array a contain the
	min(n,m)-by-n upper trapezoidal matrix R (R is upper triangular if m ≥ n);

  Modified:

    23 March 2015

  Author:

    Parth Shah
	Email: parthdshah@ufl.edu
*/

//#define NUM_PTHREADS 4
 
//#define NUM_OMP_THREADS 4

#include <omp.h>
#include <mkl_lapacke.h> 
#include <mkl_blas.h> 
#include <math.h>
#include <time.h>

double timerval () 
{
    struct timeval st;
    gettimeofday(&st, NULL);
    return (st.tv_sec+st.tv_usec*1e-6);
}

int main()
{ 
	double *a, *tau;
	lapack_int info, info1, m, n, lda;
	int matrix_order = LAPACK_ROW_MAJOR; 
	int i, j, k;
	double avg_time = 0, s_time, e_time;
	
	FILE *fp;							//output file pointer
	fp = fopen("results.txt", "a");
	
	if (fp == NULL)
	{
			printf ("Cannot open file.\n");
			exit(0);
	}
	
	m = 2; 							
	for (i = 1; i < 10; i++)
	{	
		m *= 2; 						// increase the dimension of Matrix with every iteration
		n = m;			   				// Assuming a square matrix.
		lda = m;		   				// lda: leading dimension of Matrix
		a = calloc(m*n,sizeof(double));
		tau = calloc(m,sizeof(double));
	
		// initialize the matrix
		for(j = 0; j < n; j++)
			for(k = 0; k < m; i++)
				a[k + j * m] = (k + j + 1);
		
		for (j = 0; j < 1000; j++)
		{
			s_time = timerval();
			info = LAPACKE_dgeqrf(matrix_order, m, n, a, lda, tau);  //library function for double precision QR decomposition for a general matrix
			#pragma omp barrier
			info1 = LAPACKE_dorgqr(matrix_layout, m, n, n, a, lda, tau); //library function for extracting Q matrix from output matrix of previous instruction
			e_time = timerval();			
			avg_time += (e_time - s_time);
			
			if (info <> 0 || info1 <> 0)// if info = 0 the execution is successful else the value in info is illegal element in Matrix
				return info;
		}
		
		avg_time = avg_time / 1000;
		fprintf (fp, "Input size: %d ,Time: %f\n", m, avg_time);  //print the results into the output file
		
		fclose(fp);
		
		//deallocate the memory
		free(tau);
		free(r);
		free(a);
	}	
	
	return 0;
} 

