/******************************************************************************/
/*
  Purpose:

	QR decomposition of Matrix using cusolver in CUDA
	
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

    14 April 2015

  Author:

    Parth Shah
	Email: parthdshah@ufl.edu
*/

#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include <time.h>
#include <cusolverDn.h>
#include <cublas_v2.h>
#include <cuda_runtime_api.h>

double timerval () 
{
    struct timeval st;
    gettimeofday(&st, NULL);
    return (st.tv_sec+st.tv_usec*1e-6);
}

// Error Checker for CuSOLVER functions
void cusolveSafeCall(cusolverStatus_t error1)
{
	if(CUSOLVER_STATUS_SUCCESS != error1) 
	{
		FILE *fp;
		fp = fopen("results_cuda.txt","a");
		fprintf(fp,"CUDA ERROR in the Cholesky function call or in the buffer and the error is ");
		if(error1 == CUSOLVER_STATUS_NOT_INITIALIZED) 
			fprintf(fp, "CUSOLVER_STATUS_NOT_INITIALIZED\n");
		if(error1 == CUSOLVER_STATUS_INVALID_VALUE) 
			fprintf(fp, "CUSOLVER_STATUS_INVALID_VALUE\n");	
		if(error1 == CUSOLVER_STATUS_ARCH_MISMATCH) 
			fprintf(fp, "CUSOLVER_STATUS_ARCH_MISMATCH\n");
		if(error1 == CUSOLVER_STATUS_INTERNAL_ERROR) 
			fprintf(fp, "CUSOLVER_STATUS_INTERNAL_ERROR\n");
		fclose(fp);
		exit(EXIT_FAILURE);
	}
}

// Error checker for CUDA memory related operations
void check_error(cudaError_t message1)
{
	if( cudaSuccess != message1)
	{
		FILE *fp;
		fp = fopen("results_cuda.txt","a");
		fprintf(fp,"CUDA ERROR: %s\n",cudaGetErrorString(message1));
		fclose(fp);
		exit(EXIT_FAILURE);
	}
	/*else
	{
		FILE *fp;
		fp = fopen("results_cuda.txt","a");
		//fprintf(fp,"check_error: %s\n",cudaGetErrorString(message1));
		fclose(fp);
	}*/	
}

int main()
{	
	double *A, *tau;
	int m, n, lda;
	
	int i, j, k;
	double avg_time = 0, s_time, e_time;
	
	FILE *fp;							//output file pointer
	fp = fopen("results_cuda.txt","w+");   
	fprintf(fp,"Start:\n");
	fclose(fp);
	
	int *devInfo = NULL; // info in gpu (device copy) 
	int lwork = 0;
	int info_gpu= 0;
	
	check_error(cudaMalloc(&devInfo, sizeof(int)));
	
	m = 2;
		
	for (i = 1; i < 16; i++)
	{
		double *d_A = NULL; // linear memory of GPU 
		double *d_tau = NULL; // linear memory of GPU 	
		double *d_work = NULL;

		m *= 2; 						// increase the dimension of Matrix with every iteration
		n = m;			   				// Assuming a square matrix.
		lda = m;		   				// lda: leading dimension of Matrix
		
		A = (double *)calloc(m*n,sizeof(double)); //allocate memory in host
		tau = (double *)calloc(m,sizeof(double));			
		
		//allocate memory in GPU
		check_error(cudaMalloc (&d_A , sizeof(double) * lda * m));	
		check_error(cudaMalloc (&d_tau, sizeof(double) * m)); 			
		
		cusolverDnHandle_t handle;
		cusolverDnCreate(&handle);
		
		//call function for calculating the work buffer size
		cusolveSafeCall(cusolverDnDgeqrf_bufferSize( handle, m, n, d_A, lda, &lwork)); 
		
		//allocate the work buffer memory
		check_error(cudaMalloc(&d_work, sizeof(double)*lwork));						
		
		avg_time = 0;
			// initialize the matrix
		for(j = 0; j < n; j++)
			for(k = 0; k < m; k++)
				A[k + j * m] = (k + j + 1);
			
		//copy the matrix to GPU
		check_error(cudaMemcpy(d_A, A, sizeof(double) * lda * m , cudaMemcpyHostToDevice));
		
		for (j = 0; j < 1; j++)
		{	
			info_gpu = 0;
		
			//library function for double precision QR decomposition for a general matrix
			s_time = timerval();
			cusolveSafeCall(cusolverDnDgeqrf( handle, m, n, d_A, lda, d_tau, d_work, lwork, devInfo));  
			cudaDeviceSynchronize(); 
			
			e_time = timerval();
			
			avg_time += (e_time - s_time);
			
			check_error(cudaMemcpy(&info_gpu, devInfo, sizeof(int), cudaMemcpyDeviceToHost)); 			
			
			if (info_gpu != 0)// if info = 0 the execution is successful else the value in info is illegal element in Matrix
				return info_gpu;				
				
		}
		
		avg_time = avg_time / 1000;
		
		fp = fopen("results_cuda.txt", "a");
		fprintf (fp, "Input size: %d ,Time: %f\n", m, avg_time);  //print the results into the output file
		fclose(fp);
		
		//deallocate memory
		check_error(cudaFree(d_A)); 
		check_error(cudaFree(d_tau));				
		check_error(cudaFree(d_work));
		cusolverDnDestroy(handle);
		
		free(A);
		free(tau);
	}
	
	check_error(cudaFree(devInfo));		

    return 0;
}
