/***************************************************************************
 *                           LIFTimeDrivenModelRK_GPU.cpp                      *
 *                           -------------------                           *
 * copyright            : (C) 2012 by Francisco Naveros                    *
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

#include "../../include/neuron_model/LIFTimeDrivenModelRK_GPU.h"
#include "../../include/neuron_model/VectorNeuronState.h"
#include "../../include/neuron_model/VectorNeuronState_GPU.h"

#include <iostream>
#include <cmath>
#include <string>

#include "../../include/neuron_model/LIFTimeDrivenModel_CUDA.h"
#include "../../include/cudaError.h"
//Library for CUDA
#include <cutil_inline.h>


LIFTimeDrivenModelRK_GPU::LIFTimeDrivenModelRK_GPU(string NeuronTypeID, string NeuronModelID): LIFTimeDrivenModel_GPU(NeuronTypeID, NeuronModelID) {
}

LIFTimeDrivenModelRK_GPU::~LIFTimeDrivenModelRK_GPU(){

}
		
bool LIFTimeDrivenModelRK_GPU::UpdateState(int index, VectorNeuronState * State, double CurrentTime){

	counter++;

	VectorNeuronState_GPU *state = (VectorNeuronState_GPU *) State;
	if((counter%size)==0){
		float elapsed_time;
		UpdateStateRKGPU(&elapsed_time,parameter, state->AuxStateGPU, state->AuxStateCPU, state->VectorNeuronStates_GPU, state->LastUpdateGPU, state->LastSpikeTimeGPU, state->InternalSpikeGPU, state->InternalSpikeCPU, state->SizeStates, CurrentTime);
		time+=elapsed_time;
	}else{
		UpdateStateRKGPU(parameter, state->AuxStateGPU, state->AuxStateCPU, state->VectorNeuronStates_GPU, state->LastUpdateGPU, state->LastSpikeTimeGPU, state->InternalSpikeGPU, state->InternalSpikeCPU, state->SizeStates, CurrentTime);
	}
	

	memset(state->AuxStateCPU,0,4*state->SizeStates*sizeof(float));

	if(this->GetVectorNeuronState()->Get_Is_Monitored()){
		HANDLE_ERROR(cudaMemcpy(state->VectorNeuronStates,state->VectorNeuronStates_GPU,state->GetNumberOfVariables()*state->SizeStates*sizeof(float),cudaMemcpyDeviceToHost));
		HANDLE_ERROR(cudaMemcpy(state->LastUpdate,state->LastUpdateGPU,state->SizeStates*sizeof(double),cudaMemcpyDeviceToHost));
		HANDLE_ERROR(cudaMemcpy(state->LastSpikeTime,state->LastSpikeTimeGPU,state->SizeStates*sizeof(double),cudaMemcpyDeviceToHost));
		synchronizeGPU_CPU();
	}

	return false;
}
