/***************************************************************************
 *                           TimeDrivenNeuronModel_GPU_C_INTERFACE.cpp     *
 *                           -------------------                           *
 * copyright            : (C) 2012 by Francisco Naveros                    *
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

#include "../../include/neuron_model/TimeDrivenNeuronModel_GPU_C_INTERFACE.cuh"
#include "../../include/neuron_model/TimeDrivenNeuronModel_GPU2.cuh"
#include "../../include/neuron_model/TimeDrivenModel.h"
#include "../../include/neuron_model/VectorNeuronState.h"
#include "../../include/neuron_model/VectorNeuronState_GPU_C_INTERFACE.cuh"


//Library for CUDA
#include "../../include/cudaError.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <string>

TimeDrivenNeuronModel_GPU_C_INTERFACE::TimeDrivenNeuronModel_GPU_C_INTERFACE(TimeScale time_scale) : TimeDrivenModel(time_scale),
N_thread(1), N_block(1), integration_method_GPU(0), State_GPU(0){
}

TimeDrivenNeuronModel_GPU_C_INTERFACE::~TimeDrivenNeuronModel_GPU_C_INTERFACE() {
	delete integration_method_GPU;
}

enum NeuronModelSimulationMethod TimeDrivenNeuronModel_GPU_C_INTERFACE::GetModelSimulationMethod(){
	return TIME_DRIVEN_MODEL_GPU;
}

enum NeuronModelType TimeDrivenNeuronModel_GPU_C_INTERFACE::GetModelType(){
	return NEURAL_LAYER;
}



void TimeDrivenNeuronModel_GPU_C_INTERFACE::InitializeInputCurrentSynapseStructure(){
	if (this->CurrentSynapsis != 0){
		this->CurrentSynapsis->InitializeInputCurrentPerSynapseStructure();
	}
}

std::map<std::string, boost::any> TimeDrivenNeuronModel_GPU_C_INTERFACE::GetParameters() const {
	// Return a dictionary with the parameters
	std::map<std::string, boost::any> newMap = TimeDrivenModel::GetParameters();
	ModelDescription imethod;
	imethod.param_map = this->integration_method_GPU->GetParameters();
	imethod.model_name = boost::any_cast<std::string>(imethod.param_map["name"]);
	newMap["int_meth"] = imethod;
	return newMap;
}

void TimeDrivenNeuronModel_GPU_C_INTERFACE::SetParameters(std::map<std::string, boost::any> param_map) noexcept(false){

	// Search for the parameters in the dictionary
	std::map<std::string, boost::any>::iterator it = param_map.find("int_meth");
	if (it != param_map.end()){
		if (this->integration_method_GPU != 0){
			delete this->integration_method_GPU;
		}
		ModelDescription newIntegration = boost::any_cast<ModelDescription>(it->second);
		this->integration_method_GPU = this->CreateIntegrationMethod(newIntegration);
		param_map.erase(it);

		//SET TIME-DRIVEN STEP SIZE
		this->SetTimeDrivenStepSize(this->integration_method_GPU->GetIntegrationTimeStep());
	}

	// Search for the parameters in the dictionary
	TimeDrivenModel::SetParameters(param_map);
	return;
}

