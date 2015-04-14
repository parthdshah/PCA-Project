/*Purpose:
	Sparse multiplication using MKL Library implementation for Intel Xeon Phi 5110P and Intel Xeon E5 2670
	
  Description:
  
	The OpenMP implementation of the sparse matrix - matrix multiplication performs a matrix-matrix operation using the "cusparseScsrgemm" routine  defined as
		C := op(A)*op(B);
		where:
				A, B, C are the sparse matrices in the CSR format (3-array variation);
				op(A) is one of op(A) = A, i.e. sparse matrix in this case.
  
  
  The program computes the sparse matrix - sparse matrix multiplication.
  The sparse matrices are stored in CSR(compressed storage row) format.
  The output is also a sparse matrix in CSR format.
  
  The program's aim is to record the time taken for 1000 iterations for different input sizes. The time information is recorded in a file called "output.txt".
  
    
  
  Modified:
    14 April 2015
  Author:
    Nikhil Pratap Ghanathe
    nikhilghanathe@ufl.edu
*/

#include<stdio.h>
#include<stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <cuda_runtime.h>
#include "cusparse_v2.h"


double timerval ()
{
struct timeval st;
gettimeofday(&st, NULL);
return (st.tv_sec+st.tv_usec*1e-6);
}



int main(int argc, char ** argv) {
	const int ARRAY_SIZE = 64;
	const int ARRAY_BYTES = ARRAY_SIZE * sizeof(float);

	// generate the input array on the host
	

float *nz, *nzc;
int *ia,*ja, *ic,*jc;
float *d_nz, *d_nzc;
int *d_ia,*d_ja, *d_ic,*d_jc;


int i;


double avg_time = 0, s_time, e_time;




//file to write results
FILE *fp1,*fp2,*fp3,*fp4;

		
		
int m=4;
int density;
int iterations;

/* iterate the loop for input size from 2exp3 to 2exp10 */
for (iterations=0; iterations<10; iterations++)
{
	
//int request = 0;

	m *= 2; // increase the dimension of Matrix with every iteration
	int n = m; // Assuming a square matrix.





		if((fp1 = fopen("column.txt","rw"))==NULL)
		{
			printf("error opening file\n");

		}
		
		

		if((fp3 = fopen("row.txt","rw"))==NULL)
		{
			printf("error opening file\n");

		}
		if((fp4 = fopen("nz.txt","rw"))==NULL)
		{
			printf("error opening file\n");

		}


if(iterations==0)
{
		fseek(fp1,0,SEEK_SET);
fseek(fp4,0,SEEK_SET);
fseek(fp3,0,SEEK_SET);
}




	//memory allocation for matrix A and B
	nz = (float *)calloc(m*n,sizeof(float));
	ia = (int *)calloc(m*n,sizeof(int));
	ja = (int *)calloc(m*n,sizeof(int));

	cudaMalloc((void**) &d_nz, m*n);
	cudaMalloc((void**) &d_ia, m*n);
	cudaMalloc((void**) &d_ja, m*n);


	//memory allocation for product matrix C
	nzc = (float *)calloc(m*n,sizeof(float));
	ic = (int *)calloc(m*n,sizeof(int));
	jc = (int *)calloc(m*n,sizeof(int));

		
	cudaMalloc((void**) &d_nzc, m*n);
	cudaMalloc((void**) &d_ic, m*n);
	cudaMalloc((void**) &d_ic, m*n);


	
//density of the sparse matrix to be created. 
	double dense_const = 0.05;
	
	density=(m*n)*(dense_const);

	printf("density is %d\n",density);
	

	
/*read the matrix data from the files*/

//read column
 for(i=0;i<=density;i++)
	{
		fscanf(fp1,"%d",&ia[i]);
	}

//read row ptr
 for(i=0;i<=density;i++)
	{
		fscanf(fp3,"%d",&ja[i]);
	}
	
	//read nz values
	for(i=0;i<=density;i++)
	{
		fscanf(fp4,"%f",&nz[i]);

	}


	
	
	fclose(fp1);
	
	fclose(fp2);
	fclose(fp3);
	
	
	
	/*start computation of sparse matrix * sparse matrix   */
	int baseC,nnzC;
	
// nnzTotalDevHostPtr points to host memory
int *nnzTotalDevHostPtr = &nnzC;

cusparseHandle_t handle=0;
cusparseStatus_t cusparseStatus;

//descriptor for matrices
cusparseMatDescr_t descrA=0;
cusparseMatDescr_t descrC=0;


//handle for cuSparse context
cusparseStatus = cusparseCreate(&handle);

cusparseSetPointerMode(handle, CUSPARSE_POINTER_MODE_HOST);


//Host to mem copy of input matrix
cudaMemcpy(d_nz, nz, (m*n)*sizeof(float), cudaMemcpyHostToDevice);
cudaMemcpy(d_ja, ja, (m+1)*sizeof(int), cudaMemcpyHostToDevice);
cudaMemcpy(d_ia, ia, (m*n)*sizeof(int), cudaMemcpyHostToDevice);



cusparseStatus = cusparseCreateMatDescr(&descrA);
cusparseSetMatType(descrA,CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descrA,CUSPARSE_INDEX_BASE_ONE);



cusparseStatus = cusparseCreateMatDescr(&descrC);
cusparseSetMatType(descrC,CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descrC,CUSPARSE_INDEX_BASE_ONE);
//////////////////////////////////////////////////////////////////////////
cudaMalloc((void**)&d_nzc, sizeof(int)*(m*n));


/* extract number of non zero elements of 'C'  */
cusparseXcsrgemmNnz(handle,
CUSPARSE_OPERATION_NON_TRANSPOSE, 
CUSPARSE_OPERATION_NON_TRANSPOSE, 
m,
n,
n,
descrA,
density,
d_ja,
d_ia,
descrA,
density,
d_ja,
d_ia,
descrC,
d_jc,
nnzTotalDevHostPtr);

if (NULL != nnzTotalDevHostPtr)
{
nnzC = *nnzTotalDevHostPtr;
}
else
{
cudaMemcpy(&nnzC, d_jc+m, sizeof(int), cudaMemcpyDeviceToHost);
cudaMemcpy(&baseC, d_jc, sizeof(int), cudaMemcpyDeviceToHost);
nnzC -= baseC;
}


cudaMalloc((void**)&d_ic, sizeof(int)*nnzC);
cudaMalloc((void**)&d_nzc, sizeof(float)*nnzC);

s_time = timerval();


		/*  compute for 1000 times and average out the execution time */
for(i=0;i<1000;i++)
		{
		cusparseScsrgemm(
		handle, 
		CUSPARSE_OPERATION_NON_TRANSPOSE, 
		CUSPARSE_OPERATION_NON_TRANSPOSE, 
		m, 
		n, 
		n,
		descrA, 
		density,
		d_nz, 
		d_ja, 
		d_ia,
		descrA,
		density,
		d_nz, 
		d_ja, 
		d_ia,
		descrC,
		d_nzc, 
		d_jc, 
		d_ic);
		}
e_time = timerval();


cudaMemcpy(nzc, d_nzc, (m*n)*sizeof(float), cudaMemcpyDeviceToHost);
cudaMemcpy(jc, d_jc, (m*n)*sizeof(int), cudaMemcpyDeviceToHost);
cudaMemcpy(ic, ic, (m*n)*sizeof(int), cudaMemcpyDeviceToHost);

cusparseDestroy(handle);


	
	avg_time = (e_time - s_time);
	avg_time = avg_time / 1000;
	if((fp2 = fopen("output.txt","a"))==NULL)
		{
			printf("error opening file\n");

		}
	

	fprintf (fp2, "\n Input size: %d x %d ,Time: %lf and density is %d \n", m,n, avg_time, density); 


fclose(fp1);
	
	
	
	cudaFree(d_nz);
	cudaFree(d_ia);
	cudaFree(d_ja);
	
	cudaFree(d_nzc);
	cudaFree(d_jc);
	cudaFree(d_ic);

	free(ja);
	free(ia);
	free(nz);
	
	free(jc);
	free(ic);
	free(nzc);
}	
	

return 0;
}
