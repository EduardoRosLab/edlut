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

#ifndef OPENMP_H_
#define OPENMP_H_

/*!
 * \file openmp.h
 *
 * \author Francisco Naveros
 * \date January 2014
 *
 * This file declares functions for OpenMP.
 */
extern int NumberOfOpenMPThreads;
extern int NumberOfOpenMPQueues;

#ifdef _OPENMP
	#include <omp.h>
	//IN ORDER TO USE OPENMP TASK, IT IS NEEDED TO USE OPENMP 3.0 OR HIGHER.
	#define OPENMPVERSION30 200805 
#else
	#define omp_get_thread_num() 0
	#define omp_get_num_threads() 1
	#define omp_get_max_threads() 1
	#define omp_set_num_threads() 1
	#define omp_set_nested(true)
#endif




/*!
 * \brief It checks the number of queues and threads and store its values.
 *
 * It checks the number of queues and threads and store its values.
 *
 * \param NumberOfQueues number of OpenMP queues.
 * \param NumberOfThreads number of OpenMP threads.
 */
void Set_Number_of_openmp_threads(int NumberOfQueues, int NumberOfThreads);




#endif /*OPENMP_H_*/
