/***************************************************************************
 *                           RK2_GPU.cu                                    *
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

#include "../../include/integration_method/RK2_GPU.h"
#include "../../include/integration_method/RK2_GPU2.h"
#include "../../include/neuron_model/TimeDrivenNeuronModel_GPU2.h"

//Library for CUDA
#include <helper_cuda.h>




RK2_GPU::RK2_GPU(int N_neuronStateVariables, int N_differentialNeuronState, int N_timeDependentNeuronState):IntegrationMethod_GPU("RK2", N_neuronStateVariables, N_differentialNeuronState, N_timeDependentNeuronState){
}

RK2_GPU::~RK2_GPU(){
	cudaFree(AuxNeuronState);
	cudaFree(AuxNeuronState1);
	cudaFree(AuxNeuronState2);
}

__global__ void RK2_GPU_position(void ** vector, float * element0, float * element1, float * element2){
	vector[0]=element0;
	vector[1]=element1;
	vector[2]=element2;
}
	
void RK2_GPU::InitializeMemoryGPU(int N_neurons, int Total_N_thread){
	int size=3*sizeof(float *);

	cudaMalloc((void **)&Buffer_GPU, size);

	cudaMalloc((void**)&AuxNeuronState, N_NeuronStateVariables*Total_N_thread*sizeof(float));
	cudaMalloc((void**)&AuxNeuronState1, N_NeuronStateVariables*Total_N_thread*sizeof(float));
	cudaMalloc((void**)&AuxNeuronState2, N_NeuronStateVariables*Total_N_thread*sizeof(float));

	RK2_GPU_position<<<1,1>>>(Buffer_GPU, AuxNeuronState, AuxNeuronState1, AuxNeuronState2);
}





