/***************************************************************************
 *                           VectorNeuronState_GPU.cpp                     *
 *                           -------------------                           *
 * copyright            : (C) 2012 by Jesus Garrido and Francisco Naveros  *
 * email                : jgarrido@atc.ugr.es, fnaveros@atc.ugr.es         *
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 3 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

#include "../../include/neuron_model/VectorNeuronState_GPU.h"
#include "../../include/cudaError.h"

#include <iostream>
using namespace std;

		//Library for CUDA
		#include <cutil_inline.h>


	VectorNeuronState_GPU::VectorNeuronState_GPU(unsigned int NumVariables):VectorNeuronState(NumVariables, true){
	};

	VectorNeuronState_GPU::~VectorNeuronState_GPU(){
		//HANDLE_ERROR(cudaFree(AuxStateGPU));
		HANDLE_ERROR(cudaFree(VectorNeuronStates_GPU));
		HANDLE_ERROR(cudaFree(LastUpdateGPU));
		HANDLE_ERROR(cudaFree(LastSpikeTimeGPU));
		//HANDLE_ERROR(cudaFree(InternalSpikeGPU));
		HANDLE_ERROR(cudaFreeHost(AuxStateCPU));
		HANDLE_ERROR(cudaFreeHost(InternalSpikeCPU));
	}


void VectorNeuronState_GPU::InitializeStatesGPU(int size, float * initialization){

	//Initilize State in CPU
	InitializeStates(size, initialization);
	

	
	//Memory in CPU uses as buffer.
	//cudaHostAlloc allocate CPU memory with specific properties.
	HANDLE_ERROR(cudaHostAlloc((void**)&AuxStateCPU, 2*GetSizeState()*sizeof(float),cudaHostAllocMapped));
	HANDLE_ERROR(cudaHostAlloc((void**)&InternalSpikeCPU, GetSizeState()*sizeof(bool),cudaHostAllocMapped));
	
	cudaDeviceProp prop;
	HANDLE_ERROR(cudaGetDeviceProperties( &prop, 0 ));	
	
	//Memory for GPU

	//GPU can use MapHostMemory
	if(prop.canMapHostMemory){
		HANDLE_ERROR ( cudaHostGetDevicePointer( (void**)&AuxStateGPU,AuxStateCPU, 0 ) );
		HANDLE_ERROR ( cudaHostGetDevicePointer( (void**)&InternalSpikeGPU,InternalSpikeCPU, 0 ) );
	}
	//GPU can not use MapHostMemory.
	else{
		HANDLE_ERROR(cudaMalloc((void**)&AuxStateGPU, 2*GetSizeState()*sizeof(float)));
		HANDLE_ERROR(cudaMalloc((void**)&InternalSpikeGPU, GetSizeState()*sizeof(bool)));
	}
	HANDLE_ERROR(cudaMalloc((void**)&VectorNeuronStates_GPU, GetNumberOfVariables()*GetSizeState()*sizeof(float)));
	HANDLE_ERROR(cudaMalloc((void**)&LastUpdateGPU, GetSizeState()*sizeof(double)));
	HANDLE_ERROR(cudaMalloc((void**)&LastSpikeTimeGPU, GetSizeState()*sizeof(double)));

	
	//Copy initial state from CPU to GPU
	HANDLE_ERROR(cudaMemcpy(VectorNeuronStates_GPU,VectorNeuronStates,GetNumberOfVariables()*GetSizeState()*sizeof(float),cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(LastUpdateGPU,LastUpdate,GetSizeState()*sizeof(double),cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(LastSpikeTimeGPU,LastSpikeTime,GetSizeState()*sizeof(double),cudaMemcpyHostToDevice));


}


bool * VectorNeuronState_GPU::getInternalSpike(){
	return InternalSpikeCPU;
}

