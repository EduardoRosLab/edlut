/***************************************************************************
 *                           Euler_GPU.cu                                  *
 *                           -------------------                           *
 * copyright            : (C) 2013 by Francisco Naveros                    *
 * email                : fnaveros@atc.ugr.es                              *
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 3 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

#include "../../include/integration_method/Euler_GPU.h"
#include "../../include/integration_method/Euler_GPU2.h"
#include "../../include/neuron_model/TimeDrivenNeuronModel_GPU2.h"

//Library for CUDA
#include <cutil_inline.h>


Euler_GPU::Euler_GPU(int N_neuronStateVariables, int N_differentialNeuronState, int N_timeDependentNeuronState):IntegrationMethod_GPU("Euler", N_neuronStateVariables, N_differentialNeuronState, N_timeDependentNeuronState){
}

Euler_GPU::~Euler_GPU(){
	cudaFree(AuxNeuronState);
}

__global__ void Euler_GPU_position(void ** vector, float * element0){
	vector[0]=element0;
}

void Euler_GPU::InitializeMemoryGPU(int N_neurons, int Total_N_thread){
	int size=1*sizeof(float *);

	cudaMalloc((void **)&Buffer_GPU, size);

	cudaMalloc((void**)&AuxNeuronState, N_NeuronStateVariables*Total_N_thread*sizeof(float));


	Euler_GPU_position<<<1,1>>>(Buffer_GPU, AuxNeuronState);
}
		





