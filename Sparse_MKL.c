/*Purpose:
	Sparse multiplication using MKL Library implementation for Intel Xeon Phi 5110P and Intel Xeon E5 2670
	
  Description:
  
	The OpenMP implementation of the sparse matrix - matrix multiplication performs a matrix-matrix operation using the ""mkl_dcsrmultcsr"" routine  defined as

		C := op(A)*B
		where:
				A, B, C are the sparse matrices in the CSR format (3-array variation);
				op(A) is one of op(A) = A, or op(A) =A', or op(A) = conjg(A') .
  
  
  The program computes the sparse matrix - sparse matrix multiplication.
  The sparse matrices are stored in CSR(compressed storage row) format.
  The output is also a sparse matrix in CSR format.
  
  The program's aim is to record the time taken for 1000 iterations for different input sizes. The time information is recorded in a file called "output.txt".
  
    If info=0, the execution is successful.

If info=I>0, the routine stops calculation in the I-th row of the matrix C because number of elements in C exceeds nzmax.

If info=-1, the routine calculates only the size of the arrays c and jc and returns this value plus 1 as the last element of the array ic.
  
  Modified:
    1 April 2015
  Author:
    Nikhil Pratap Ghanathe
    nikhilghanathe@ufl.edu
*/




#include<stdio.h>
#include<stdlib.h>
#include <time.h>
#include <omp.h>
#include "mkl.h"
#include "mkl_spblas.h"

double timerval ()
{
struct timeval st;
gettimeofday(&st, NULL);
return (st.tv_sec+st.tv_usec*1e-6);
}


int main()
{
	


double *a,*nz, *nzc;
int *ia,*ja, *ic,*jc,*pos;
int info=1;

int i, j, k;


double avg_time = 0, s_time, e_time;
//open file to write results
FILE *fp1;

		if((fp1 = fopen("output.txt","w+"))==NULL)
		{
			printf("error opening file\n");

		}


int m=4;
int iterations;

/* iterate the loop for input size from 2exp3 to 2exp10 */
for (iterations=0; iterations<10; iterations++)
{
	


	m *= 2; // increase the dimension of Matrix with every iteration
	int n = m; // Assuming a square matrix.



	//memory allocation for matrix A and B
	nz = calloc(m*n,sizeof(double));
	ia = calloc(m*n,sizeof(int));
	ja = calloc(m*n,sizeof(int));



	//memory allocation for product matrix C
	nzc = calloc(m*n,sizeof(double));
	ic = calloc(m*n,sizeof(int));
	jc = calloc(m*n,sizeof(int));



	//Configuration parameters
	 	char trans = 'N';	
		int request = 1;
		int sort = 8;
		int nzmax = 0;





	k=0;

	//density of the sparse matrix to be created. Assume 3% density.
	double dense_const = 0.03;
	int temp5, temp6,temp3,temp4;

	int density=(m*n)*(dense_const);


	//position array for random initialisation of positions in input matrix
	pos= calloc(m*n, sizeof(int));



	int temp,temp1;

	printf("the density is %d\n",density);


	//randomly initialise positions
	for(i=0;i<density;i++)
	{
		temp1=rand()%(m*n);
	
			pos[i]=temp1;
	
	}

	//sort the 'pos' array
	for (i = 0 ; i < density; i++) {
	    int d = i;
		int t;
	 
	    while ( d > 0 && pos[d] < pos[d-1]) {
	      t          = pos[d];
	      pos[d]   = pos[d-1];
	      pos[d-1] = t;
	 
	      d--;
	    }
	  }





	/* initialise with non zero elements and extract column and row ptr vector*/
	j=1;
	ja[0]=1;

	int p=0;


		for(i = 0; i < density; i++)
			{
				temp=pos[i];
				 nz[k] = rand();
			 	 ia[k] = temp%m;
				k++;
				p++;
		

			temp5= pos[i];
			temp6=pos[i+1];

			 temp3=temp5-(temp5%m);
			 temp4=temp6-(temp6%m);
	

			if(!(temp3== temp4))
			{	

			if((temp3+m==temp6))
			{}
		
		else	
			{	
			
				  ja[j]=p+1;
	
				  j++;
				}
		
		}		

	}

	
/*Compute the product of two sparse matrices*/
	#pragma omp parallel shared(m,n,ia,ja,jc,ic,nz,nzc,nzmax,info,request,sort,trans) private(i)
	{
	#pragma omp for
	for(i=0; i<1000;i++)
		{
	s_time = timerval();
	
	 mkl_dcsrmultcsr(&trans, &request, &sort, &m, &n, &n, nz, ia, ja, nz, ia, ja, nzc, jc, ic, &nzmax, &info);
	
	#pragma omp barrier
	
	e_time = timerval();
	avg_time = (e_time - s_time);
	}
	}
	
/* write the timing information in "output.txt"*/
	avg_time = avg_time / 1000;

	fprintf (fp1, "Input size: %d x %d ,Time: %lf\n", m,n, avg_time); 



	
	free(ja);
	free(ia);
	free(nz);

	free(jc);
	free(ic);
	free(nzc);

}

fclose(fp1);	
return 0;

}


























