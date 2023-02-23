/***************************************************************************
 *                           EventDrivenInputDevice.cpp                    *
 *                           -------------------                           *
 * copyright            : (C) 2018 by Francisco Naveros                    *
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

#include "../../include/neuron_model/EventDrivenInputDevice.h"
//#include "../../include/neuron_model/NeuronModel.h"
//#include "../../include/neuron_model/VectorNeuronState.h"

//#include "../../include/openmp/openmp.h"
#include "spike/Interconnection.h"

EventDrivenInputDevice::EventDrivenInputDevice(TimeScale new_time_scale): NeuronModel(new_time_scale){
}

EventDrivenInputDevice::~EventDrivenInputDevice() {
}


enum NeuronModelSimulationMethod EventDrivenInputDevice::GetModelSimulationMethod(){
	return EVENT_DRIVEN_MODEL;
}


enum NeuronModelType EventDrivenInputDevice::GetModelType(){
	return INPUT_DEVICE;
}


enum NeuronModelInputActivityType EventDrivenInputDevice::GetModelInputActivityType(){
	return NONE_INPUT;
}

bool EventDrivenInputDevice::CheckSynapseType(Interconnection * connection){
	cout << "ERROR: Synapse " << connection->GetIndex() << " connect with input device " << connection->GetTargetNeuronModel()->GetNeuronModelName();
	cout << ". Input devices do not support input synapses." << endl;
	return false;
}

std::map<std::string,boost::any> EventDrivenInputDevice::GetParameters() const {
	// Return a dictionary with the parameters
	std::map<std::string,boost::any> newMap = NeuronModel::GetParameters();
	return newMap;
}

void EventDrivenInputDevice::SetParameters(std::map<std::string, boost::any> param_map) noexcept(false){

	// Search for the parameters in the dictionary
	NeuronModel::SetParameters(param_map);
	return;
}

std::map<std::string,boost::any> EventDrivenInputDevice::GetDefaultParameters() {
	// Return a dictionary with the parameters
	std::map<std::string,boost::any> newMap = NeuronModel::GetDefaultParameters();
	return newMap;
}






