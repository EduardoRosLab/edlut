/***************************************************************************
 *                           openmp.h                                      *
 *                           -------------------                           *
 * copyright            : (C) 2014 by Francisco Naveros                    *
 * email                : fnaveros@ugr.es                                  *
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 3 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

#include <iostream>

using namespace std;

#include "../../include/openmp/openmp.h"
#include "../../include/cudaError.h"

int NumberOfOpenMPThreads;
int NumberOfOpenMPQueues;
int NumberOfGPUs;
int * GPUsIndex;

void Set_Number_of_openmp_threads(int NumberOfQueues, int NumberOfThreads){
	if(NumberOfThreads>omp_get_max_threads()){
		NumberOfThreads=omp_get_max_threads();
		cout<<"WARNING: thread number higher than CPU cores number. Thread number fixed to "<< NumberOfThreads<<endl;
	}
	if(NumberOfQueues>omp_get_max_threads()){
		NumberOfQueues=omp_get_max_threads();
		cout<<"WARNING: queue number higher than CPU cores number. Queue number fixed to "<< NumberOfQueues<<endl;
	}
	if(NumberOfQueues>NumberOfThreads){
		NumberOfThreads=NumberOfQueues;
		cout<<"WARNING: queue number higher than thread number. Thread number fixed to "<< NumberOfThreads<<endl;
	}

	#ifdef _OPENMP 
		#if	_OPENMP >= OPENMPVERSION30
			cout<<"OPNEMP3.0 or higher: tasks available"<<endl;
		#else
			cout<<"OPNEMP2.5 or lower: tasks not available"<<endl;
			if(NumberOfThreads>NumberOfQueues){
				NumberOfThreads=NumberOfQueues;
				cout<<"WARNING: thread number fixed equal that queue number. Thread number fixed to "<< NumberOfThreads<<endl;
			}
		#endif
	#else
		cout<<"OpenMP support is disabled"<<endl;
	#endif



	(void) omp_set_num_threads(NumberOfThreads);
	NumberOfOpenMPThreads=NumberOfThreads;
	NumberOfOpenMPQueues=NumberOfQueues;


	

	int nDevices=0;
	int nValidDevices=0;
	int mayor_architecture=0;
	cudaGetDeviceCount(&nDevices);
	if(nDevices==0){
		printf("WARNING: No CUDA capable GPU available\n");
		NumberOfGPUs=0;
	}else{
		GPUsIndex=new int[nDevices]();
		cudaDeviceProp prop;
		for(int i=0; i<nDevices; i++){
			cudaGetDeviceProperties(&prop, i);
			if(prop.major>=2){
				GPUsIndex[nValidDevices]=i;
				nValidDevices++;
			}
		}
		if(nValidDevices==0){
			printf("WARNING: No CUDA capable GPU available with CUDA architecture 2.0 or higher\n");
		}else{
			printf("%d CUDA capable GPU available with CUDA architecture 2.0 or higher\n", nValidDevices);
		}
	}


	if(nValidDevices>NumberOfOpenMPQueues){
		NumberOfGPUs=NumberOfOpenMPQueues;
	}else{
		NumberOfGPUs=nValidDevices;
	}

	printf("%d GPUs will be used\n",NumberOfGPUs);

}
