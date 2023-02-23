/***************************************************************************
 *                           InputSpikeNeuronModel.cpp                     *
 *                           -------------------                           *
 * copyright            : (C) 2015 by Francisco Naveros                    *
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

#include "neuron_model/InputSpikeNeuronModel.h"
#include "simulation/NetworkDescription.h"

InputSpikeNeuronModel::InputSpikeNeuronModel() : EventDrivenInputDevice(MilisecondScale) {
	std::map<std::string, boost::any> param_map = InputSpikeNeuronModel::GetDefaultParameters();
	param_map["name"] = InputSpikeNeuronModel::GetName();
	this->SetParameters(param_map);
}

InputSpikeNeuronModel::~InputSpikeNeuronModel() {
}


enum NeuronModelOutputActivityType InputSpikeNeuronModel::GetModelOutputActivityType(){
	return OUTPUT_SPIKE;
}

ostream & InputSpikeNeuronModel::PrintInfo(ostream & out){
	out << "- Input Neuron Model: " << InputSpikeNeuronModel::GetName() << endl;

	return out;
}	
std::map<std::string,boost::any> InputSpikeNeuronModel::GetParameters() const {
	// Return a dictionary with the parameters
	std::map<std::string,boost::any> newMap = EventDrivenInputDevice::GetParameters();
	return newMap;
}

std::map<std::string, boost::any> InputSpikeNeuronModel::GetSpecificNeuronParameters(int index) const noexcept(false){
	return GetParameters();
}

void InputSpikeNeuronModel::SetParameters(std::map<std::string, boost::any> param_map) noexcept(false){
	// Search for the parameters in the dictionary
	EventDrivenInputDevice::SetParameters(param_map);
	return;
}


std::map<std::string,boost::any> InputSpikeNeuronModel::GetDefaultParameters() {
	// Return a dictionary with the parameters
	std::map<std::string, boost::any> newMap = EventDrivenInputDevice::GetDefaultParameters();

	return newMap;
}

NeuronModel* InputSpikeNeuronModel::CreateNeuronModel(ModelDescription nmDescription){
	InputSpikeNeuronModel * nmodel = new InputSpikeNeuronModel();
	nmodel->SetParameters(nmDescription.param_map);
	return nmodel;
}

ModelDescription InputSpikeNeuronModel::ParseNeuronModel(std::string FileName) noexcept(false){
	ModelDescription nmodel;
	nmodel.model_name = InputSpikeNeuronModel::GetName();

    nmodel.param_map["name"] = boost::any(InputSpikeNeuronModel::GetName());
	return nmodel;
}

std::string InputSpikeNeuronModel::GetName(){
	return "InputSpikeNeuronModel";
}

std::map<std::string, std::string> InputSpikeNeuronModel::GetNeuronModelInfo() {
	// Return a dictionary with the parameters
	std::map<std::string, std::string> newMap;
	newMap["info"] = std::string("CPU Event-driven input device able to propagate input spikes events to other neurons");

	return newMap;
}


