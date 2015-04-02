// cholesky decomposition using the library - cuSolver

//**********************************
// cuSolver library must be installed
//**********************************

// link to the documentation of the library - http://docs.nvidia.com/cuda/cusolver/index.html#axzz3V3SakC7i

// author	: yathindra kota 
// mail		: yatkota@ufl.edu
// last modified: 2 April, 2015

#include <stdio.h>
#include <time.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>
#include <assert.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cudense.h>

#define NUM_THREADS (num_rows*num_cols)
#define BLOCK_WIDTH 1000

void check_error(const char *msg)
{
	cudaError_t err = cudaGetLastError();
	if( cudaSuccess != err)
	{
		printf("CUDA ERROR: %s (%s).\n", msg, cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
}


int main(int argc, char* argv[])
{ 
	cudsHandle_t cudenseH = NULL;
    	cusolverStatus_t cusolver_status = CUSOLVER_STATUS_SUCCESS;  
	
	int lwork = 0;
	int *devInfo = NULL;
	int info_gpu_h = 0;
	int num_rows = 4;
	int num_cols = 4;
	int num_iterations = 10;
	
	FILE *fp;
	
	// create cudense/cublas handle
    	cusolver_status = cudsCreate(&cudenseH);
    	cudaError_t cudaStat1;
	float time_iterations = 0; //time for given set of iterations
	
	for(int matsize = 0; matsize < 10; matsize++)
	{	
		double *h_matrix;
		double *h_matrix_ouput;
		
		h_matrix = malloc(num_rows*num_cols, sizeof(double));
		h_matrix_ouput = malloc(num_rows*num_cols, sizeof(double));
		
		int temp;
		for(temp = 0; temp < num_iterations; temp++)
		{
			//initialize the matrix to a random set of values (0 to 10) of type float 	
			for(int tempi=0; tempi<num_rows;tempi++)
			{
				for(int tempj=0; tempi<num_cols;tempj++)	
				{
					h_matrix[(tempi*num_rows) + tempj] = (rand()%100)/10; // set the input matrix elements 
				}	
			}		

			// allocate memory in the GPU and also copy the input matrix from hot to device
			double *d_matrix_input;
			check_error(cudaMalloc((void **) &d_matrix_input, num_rows * num_cols * sizeof(float)));
			check_error(cudaMemcpy(*d_matrix_input, *h_matrix, sizeof(float) * num_rows * num_cols, cudaMemcpyHostToDevice));

			//------	
			// alocation of the buffer in the GPU
			// The following function allocates required amount of memory in the GPU and this
			// function is specific to Cholesky function which is called next
			// This function is also from the library cuSolver
				
			cusolverStatus_t cusolverDnDpotrf_bufferSize(cudenseH,
						 CUBLAS_FILL_MODE_LOWER, /*cublasFillMode_t uplo, Maybe "CUBLAS_FILL_MODE_LOWER" */
						 num_rows,
						 d_matrix_input,
						 num_rows,
						 &Lwork);
			
			double *d_work = NULL; 	
			cudaStat1 = cudaMalloc((void**)&d_work, sizeof(double)*Lwork);
			
			cudaStat1 = cudaDeviceSynchronize(); // synchronization
			
			timer.start();
			
			// implementation of Cholesky
			cusolverStatus_t cusolverDnDpotrf(cudenseH,
				   CUBLAS_FILL_MODE_LOWER,/* not sure, needs to be checked*/
				   num_rows,
				   d_matrix_input,
				   num_rows,
				   d_work, /*not sure */
				   Lwork,
				   devInfo );
			
			cudaStat1 = cudaDeviceSynchronize(); // synchronization
			
			timer.stop();
			
			time_iterations += timer.Elapsed;	
			
			//// check if Cholesky is good or not
			cudaStat1 = cudaMemcpy(&info_gpu_h, devInfo, sizeof(int), cudaMemcpyDeviceToHost);
			
			if(info_gpu_h != 0)
			{
				fp = fopen("cholesky_output.txt","w+");
				fprintf(fp,"Unsuccessful execution. DevInfo is not zero. Iterations number = %d \n", temp);
				fclose(fp);
			}	
		}
		cudaStat1 = cudaMemcpy(h_matrix_ouput, d_work, sizeof(double)*num_rows*num_cols, cudaMemcpyDeviceToHost); 
		//d_work needs to checked
		
		fp = fopen("cholesky_output.txt","a");
		fprintf(fp,"Time elapsed(average for the specified iterations) = %g ms, number of rows is %d\n", time_iterations/num_iterations,num_rows);
		fclose(fp);
		
		time_iterations = 0;
		
		free(h_matrix);
		free(h_matrix_ouput);		
			
		num_cols *= 2;
		num_rows *= 2;
	}
	// print output to file if required
	/*
	
	fprintf(fp,"output is\n");	
	for(tempi=0; tempi<num_rows;tempi++)
	{
		fprintf(fp,"\n");
		for(tempj=0; tempi<num_cols;tempj++)	
		{
			fp = fopen("cholesky_output.txt","a");
			fprintf(fp,"%f", h_matrix_ouput[(tempj * num_rows) + tempi]);
			fclose(fp);
		}		
	}
	
	*/
	//de-allocating the memory
	
	if(d_matrix_input)
		cudaFree(d_matrix_input);
		
	if(d_work)
		cudaFree(d_work);
	
    	if (cudenseH) 
		cudsDestroy(cudenseH); 	

	return 0;	
}
