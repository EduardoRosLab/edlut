/***************************************************************************
 *                           TimeDrivenNeuronModel_GPU.cpp                 *
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

#include "../../include/neuron_model/TimeDrivenNeuronModel_GPU.h"
#include "../../include/neuron_model/TimeDrivenNeuronModel_GPU2.h"
#include "../../include/neuron_model/NeuronModel.h"
#include "../../include/neuron_model/VectorNeuronState.h"
#include "../../include/neuron_model/VectorNeuronState_GPU.h"


//Library for CUDA
#include "../../include/cudaError.h"
#include <helper_cuda.h>

#include <string>

TimeDrivenNeuronModel_GPU::TimeDrivenNeuronModel_GPU(string NeuronTypeID, string NeuronModelID): NeuronModel(NeuronTypeID, NeuronModelID), TimeDrivenStep_GPU(0.001) {
	// TODO Auto-generated constructor stub
}

TimeDrivenNeuronModel_GPU::~TimeDrivenNeuronModel_GPU() {
	delete integrationMethod_GPU;
	HANDLE_ERROR(cudaEventDestroy(stop));
}

double TimeDrivenNeuronModel_GPU::GetTimeDrivenStep_GPU(){
	return TimeDrivenStep_GPU;
}


enum NeuronModelType TimeDrivenNeuronModel_GPU::GetModelType(){
	return TIME_DRIVEN_MODEL_GPU;
}

