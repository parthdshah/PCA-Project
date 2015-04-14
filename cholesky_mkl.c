// cholesky decomposition using the library - mkl

//**********************************
// mkl library must be installed
//**********************************

// link to the documentation of the library - http://goo.gl/i0TgYs

// author	: yathindra kota 
// mail		: yatkota@ufl.edu
// last modified: 14 April, 2015

#include<stdlib.h>
#include<stdio.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <mkl.h>
#include <mkl_lapacke.h> 

double timerval()
{
	struct timeval st;
	gettimeofday(&st, NULL);
	return (st.tv_sec+st.tv_usec*1e-6);
}

int main()
{
	FILE *fp; // File pointer to point to the output file
	double avg_time = 0, s_time, e_time; // time calculation variables 
	
	int num_cols = 4; //start number of columns
	int num_rows = 4; //start number of rows
	int iterations = 1; // Total number of iterations to be performed
	int mat_sizes = 16; // Final matrix size required to be calculated
	// if mat_sizes = 2: perform for 4 and 8.. 
	int i=0, j=0; // count variables	
	
	lapack_int errorcheck;
	int matrix_order = LAPACK_ROW_MAJOR;
	char uplo = 'L';
	lapack_int order_matrix;
	lapack_int lda;
	
	fp = fopen("cholesky_output-mkl.txt","w+"); // output file
	fprintf(fp,"Start:\n");
	fclose(fp);
	
	int mat_counter; //used as counter for matrix sizes
	
	for(mat_counter = 0; mat_counter < mat_sizes; mat_counter++)
	{
		avg_time = 0;
		order_matrix = num_rows;
		lda = num_rows;
		int tempj;
		
		// perform the computations for iterations number of times
		for(tempj = 0; tempj < iterations; tempj++)
		{
			errorcheck = 0;
			double *chol_matrix;
			chol_matrix = (double *) calloc(num_rows*num_cols,sizeof(double));
			
			fp = fopen("cholesky_output-mkl.txt","a");
			//set matrix input here			
			for(i = 0; i < num_cols; i++)
			{
				for(j = 0; j < num_rows; j++)
				{			
						if(i == j)
						{
							chol_matrix[j + i*num_rows] = i+1;
						}
						
						else 
							chol_matrix[j + i*num_rows] = fmin((float)i,(float)j) - 1; 
						//	fprintf (fp,"%f\t",chol_matrix[j + i*num_rows] );
				}
				//	fprintf (fp,"\n");
			}
			fclose(fp);
							
			s_time = timerval();
			
			errorcheck = LAPACKE_dpotrf( matrix_order, 
										uplo,
										order_matrix, 
										chol_matrix, 
										lda);
	
			e_time = timerval();			
			avg_time += (e_time - s_time);

			if(errorcheck != 0)
			{
				fp = fopen("cholesky_output-mkl.txt","a");	
				fprintf(fp,"\nError check from LAPACKE_dpotrf function is not zero. \n");
				
				if(errorcheck)
				{
					fprintf(fp, "The leading minor of order %d is not positive-definite", errorcheck);
				}
				
				else
				{
					fprintf(fp, "The parameter %d is an illegal value", errorcheck);
				}
				
				fclose(fp);
				
				return -1;
			}
			/*
			fp = fopen("cholesky_output-mkl.txt","a");	
			//fprintf(fp,"\nFactorized matrix\n");
			for( i = 0; i < num_rows; i++)
				for( j = 0; j < num_cols; j++)
					if (i <= j) //fprintf(fp,"L[%i, %i] = %lf\n", i, j, chol_matrix[j + i*num_rows]);	
		*/
			free(chol_matrix);		
		}	
		
		fp = fopen("cholesky_output-mkl.txt","a");
		fprintf(fp,"Execution time for matrix size %d, iterations= %d is %g\n", num_rows ,iterations, avg_time/iterations );
		fclose(fp);
		
		num_cols *= 2;
		num_rows *= 2;	
	}		
	
	return 0;
}
