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

int NumberOfOpenMPThreads;
int NumberOfOpenMPQueues;

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

}
