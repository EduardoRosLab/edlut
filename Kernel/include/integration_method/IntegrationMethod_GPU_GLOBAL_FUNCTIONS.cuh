/***************************************************************************
 *                           IntegrationMethod_GPU_GLOBAL_FUNCTIONS.cuh    *
 *                           -------------------                           *
 * copyright            : (C) 2019 by Francisco Naveros                    *
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

#ifndef INTEGRATIONMETHOD_GPU_GLOBAL_FUNCTIONS_H_
#define INTEGRATIONMETHOD_GPU_GLOBAL_FUNCTIONS_H_

/*!
 * \file IntegrationMethod_GPU_GLOBAL_FUNCTIONS.cuh
 *
 * \author Francisco Naveros
 * \date September 2019
 *
 * This file declares GLOBAL GPU FUNCIONTS for GPU integration methods. 
 */


#include "../../include/cudaError.h"
//Library for CUDA
#include "cuda_runtime.h"
#include "device_launch_parameters.h"


void Call_Euler_GPU_C_INTERFACE_memory(void ** vector, float * integration_method_parameters_GPU, float * element1);


void Call_RK2_GPU_C_INTERFACE_memory(void ** vector, float * integration_method_parameters_GPU, float * element1, float * element2);


void Call_RK4_GPU_C_INTERFACE_memory(void ** vector, float * integration_method_parameters_GPU, float * element1, float * element2, float * element3, float * element4, float * element5);


void Call_BDFn_GPU_C_INTERFACE_memory(void ** vector, float * integration_method_parameters_GPU, float * element1, float * element2, float * element3, float * element4, float * element5, float * element6, float * element7, float * element8, float * element9, float * element10, int * element11, float * element12, float * element13, float * element14);


#endif /* INTEGRATIONMETHOD_GPU_GLOBAL_FUNCTIONS_H_ */
