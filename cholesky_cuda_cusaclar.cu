// cholesky decomposition using the library - cuSolver

//**********************************
// cuSolver library must be installed
//**********************************

// link to the documentation of the library - http://docs.nvidia.com/cuda/cusolver/index.html#axzz3V3SakC7i

// author	: yathindra kota 
// mail		: yatkota@ufl.edu
// last modified: 13 April, 2015

#include "cuda_runtime.h"
#include<iostream>
#include<iomanip>
#include<stdlib.h>
#include<stdio.h>
#include<assert.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <cusolverDn.h>
#include <cublas_v2.h>
#include <cuda_runtime_api.h>

double timerval()
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
		fp = fopen("cholesky_output.txt","a");
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
		fp = fopen("cholesky_output.txt","a");
		fprintf(fp,"CUDA ERROR: %s\n",cudaGetErrorString(message1));
		fclose(fp);
		exit(EXIT_FAILURE);
	}
	else
	{
		FILE *fp;
		fp = fopen("cholesky_output.txt","a");
		//fprintf(fp,"check_error: %s\n",cudaGetErrorString(message1));
		fclose(fp);
	}	
}

int main()
{
	int num_cols = 4; //start number of columns
	int num_rows = 4; //start number of rows
	int iterations = 1; // Total number of iterations to be performed
	int mat_sizes = 16; // Final matrix size required to be calculated
	// if mat_sizes = 2: perform for 4 and 8.. 
	int i=0, j=0; // count variables
	FILE *fp; // File pointer to point to the output file
	double avg_time = 0, s_time, e_time; // time calculation variables 
	
	fp = fopen("cholesky_output.txt","w+"); // output file
	fprintf(fp,"Start:\n");
	fclose(fp);
	
	int mat_counter = 0; //used as counter for matrix sizes	
	
	// perform the computation for mat_sizes number of different matrix sizes
	for(mat_counter = 0; mat_counter < mat_sizes; mat_counter++)
	{
		avg_time = 0;
		
		// perform the computations for iterations number of times
		for(int tempj = 0; tempj< iterations; tempj ++)
		{
			double *h_matrix;
			h_matrix = (double *) calloc(num_rows*num_cols,sizeof(double)); //(double **)malloc(sizeof(double *) * num_cols);
			
			fp = fopen("cholesky_output.txt","a");
			//set matrix input here			
			for(i = 0; i < num_cols; i++)
			{
				for(j = 0; j < num_rows; j++)
				{			
					if(i == j)
					{
						h_matrix[j + i*num_rows] = i+1;
					}
					
					else 
						h_matrix[j + i*num_rows] = fmin((float)i,(float)j) - 1; 
					//	fprintf (fp,"%f\t",h_matrix[j + i*num_rows] );
				}
				//	fprintf (fp,"\n");
			}
			fclose(fp);
			
			// --- Setting the device matrix and copying the host matrix to the device
			double *d_matrix;            
			check_error(cudaMalloc(&d_matrix, num_rows * num_cols * sizeof(double)));
			check_error(cudaMemcpy(d_matrix, h_matrix, num_rows* num_cols * sizeof(double), cudaMemcpyHostToDevice));
			// --- cuSOLVE input/output parameters/arrays
			int work_size = 0;
			int *devInfo;           
			check_error(cudaMalloc(&devInfo, sizeof(int)));
			// --- CUDA solver initialization
			cusolverDnHandle_t solver_handle;
			cusolverDnCreate(&solver_handle);
				
			// --- CHOLESKY buffer initialization
			cusolveSafeCall(cusolverDnDpotrf_bufferSize(solver_handle, CUBLAS_FILL_MODE_LOWER, num_rows, d_matrix, num_rows, &work_size));

			double *work;   
			check_error(cudaMalloc(&work, work_size * sizeof(double)));
				
			s_time = timerval();
			
			// Cholesky execution
			cusolveSafeCall(cusolverDnDpotrf(solver_handle, CUBLAS_FILL_MODE_LOWER, num_rows, d_matrix, num_rows, work, work_size, devInfo));
			
			e_time = timerval();	
					
			avg_time += (e_time - s_time);
			
			int devInfo_h = 0;  
			check_error(cudaMemcpy(&devInfo_h, devInfo, sizeof(int), cudaMemcpyDeviceToHost));
			if (devInfo_h != 0) std::cout   << "Unsuccessful potrf execution\n\n";
			
			fp = fopen("cholesky_output.txt","a");	
			//fprintf(fp,"\nFactorized matrix\n");
			
			fclose(fp);
			check_error(cudaMemcpy(h_matrix, d_matrix, num_rows * num_cols * sizeof(double), cudaMemcpyDeviceToHost));
			
			/*
			// to print output
			fp = fopen("cholesky_output.txt","a");
				
			for( i = 0; i < num_rows; i++)
				for( j = 0; j < num_cols; j++)
					if (i <= j) //fprintf(fp,"L[%i, %i] = %lf\n", i, j, h_matrix[j + i*num_rows]);

				fclose(fp);	
			
			*/
			free(h_matrix);		
			check_error(cudaFree(d_matrix));
			check_error(cudaFree(devInfo));
			check_error(cudaFree(work));			
			cusolverDnDestroy(solver_handle);			
		}
		fp = fopen("cholesky_output.txt","a");
		fprintf(fp,"Execution time for matrix size %d, iterations= %d is %g\n", num_rows ,iterations, avg_time/iterations );
		fclose(fp);	
			
		num_cols *= 2;
		num_rows *= 2;		
	}	
    return 0;
}
