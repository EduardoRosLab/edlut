/***************************************************************************
 *                           IntegrationMethod_GPU_GLOBAL_FUNCTIONS.cu     *
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


#include "../../include/integration_method/IntegrationMethod_GPU_GLOBAL_FUNCTIONS.cuh"

#include "../../include/cudaError.h"
//Library for CUDA
#include "cuda_runtime.h"
#include "device_launch_parameters.h"



__global__ void Euler_GPU_C_INTERFACE_memory(void ** vector, float * integration_method_parameters_GPU, float * element1){
	vector[0]=integration_method_parameters_GPU;
	vector[1]=element1;
}


void Call_Euler_GPU_C_INTERFACE_memory(void ** vector, float * integration_method_parameters_GPU, float * element1){
	Euler_GPU_C_INTERFACE_memory <<<1, 1 >>>(vector, integration_method_parameters_GPU, element1);
}


__global__ void RK2_GPU_C_INTERFACE_memory(void ** vector, float * integration_method_parameters_GPU, float * element1, float * element2){
	vector[0]=integration_method_parameters_GPU;
	vector[1]=element1;
	vector[2]=element2;
}


void Call_RK2_GPU_C_INTERFACE_memory(void ** vector, float * integration_method_parameters_GPU, float * element1, float * element2){
	RK2_GPU_C_INTERFACE_memory <<<1, 1 >>>(vector, integration_method_parameters_GPU, element1, element2);
}

__global__ void RK4_GPU_C_INTERFACE_memory(void ** vector, float * integration_method_parameters_GPU, float * element1, float * element2, float * element3, float * element4, float * element5){
	vector[0]=integration_method_parameters_GPU;
	vector[1]=element1;
	vector[2]=element2;
	vector[3]=element3;
	vector[4]=element4;
	vector[5]=element5;
}


void Call_RK4_GPU_C_INTERFACE_memory(void ** vector, float * integration_method_parameters_GPU, float * element1, float * element2, float * element3, float * element4, float * element5){
	RK4_GPU_C_INTERFACE_memory <<<1, 1 >>>(vector, integration_method_parameters_GPU, element1, element2, element3, element4, element5);
}



__global__ void BDFn_GPU_C_INTERFACE_memory(void ** vector, float * integration_method_parameters_GPU, float * element1, float * element2, float * element3, float * element4, float * element5, float * element6, float * element7, float * element8, float * element9, float * element10, int * element11, float * element12, float * element13, float * element14){
	vector[0] = integration_method_parameters_GPU;
	vector[1] = element1;
	vector[2] = element2;
	vector[3] = element3;
	vector[4] = element4;
	vector[5] = element5;
	vector[6] = element6;
	vector[7] = element7;
	vector[8] = element8;
	vector[9] = element9;
	vector[10] = element10;
	vector[11] = element11;
	vector[12] = element12;
	vector[13] = element13;
	vector[14] = element14;
}


void Call_BDFn_GPU_C_INTERFACE_memory(void ** vector, float * integration_method_parameters_GPU, float * element1, float * element2, float * element3, float * element4, float * element5, float * element6, float * element7, float * element8, float * element9, float * element10, int * element11, float * element12, float * element13, float * element14){
	BDFn_GPU_C_INTERFACE_memory << <1, 1 >> >(vector, integration_method_parameters_GPU, element1, element2, element3, element4, element5, element6, element7, element8, element9, element10, element11, element12, element13, element14);
}




