/***************************************************************************
 *                           TimeDrivenInputDevice.cpp                     *
 *                           -------------------                           *
 * copyright            : (C) 2020 by Francisco Naveros                    *
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

#include "neuron_model/TimeDrivenInputDevice.h"
//#include "neuron_model/TimeDrivenModel.h"
#include "neuron_model/VectorNeuronState.h"
#include "simulation/NetworkDescription.h"
#include "integration_method/IntegrationMethodFactory.h"
//#include "integration_method/Euler.h"
//#include "openmp/openmp.h"
#include "spike/Interconnection.h"

TimeDrivenInputDevice::TimeDrivenInputDevice(TimeScale new_time_scale):
		TimeDrivenModel(new_time_scale){
}

TimeDrivenInputDevice::~TimeDrivenInputDevice() {
}


enum NeuronModelSimulationMethod TimeDrivenInputDevice::GetModelSimulationMethod(){
	return TIME_DRIVEN_MODEL_CPU;
}

enum NeuronModelType TimeDrivenInputDevice::GetModelType(){
	return INPUT_DEVICE;
}

enum NeuronModelInputActivityType TimeDrivenInputDevice::GetModelInputActivityType(){
	return NONE_INPUT;
}

//VERIFICAR EN TODOS LOS MODELOS
bool TimeDrivenInputDevice::CheckSynapseType(Interconnection * connection){
	cout << "ERROR: Synapse " << connection->GetIndex() << " connect with input device " << connection->GetTargetNeuronModel()->GetNeuronModelName();
	cout << ". Input devices do not support input synapses." << endl;
	return false;
}


std::map<std::string,boost::any> TimeDrivenInputDevice::GetParameters() const {
	// Return a dictionary with the parameters
	std::map<std::string,boost::any> newMap = TimeDrivenModel::GetParameters();
	return newMap;
}

void TimeDrivenInputDevice::SetParameters(std::map<std::string, boost::any> param_map) noexcept(false){
	// Search for the parameters in the dictionary
	TimeDrivenModel::SetParameters(param_map);
}


