/***************************************************************************
 *                           InputCurrentNeuronModel.cpp                   *
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

#include "neuron_model/InputCurrentNeuronModel.h"
#include "simulation/NetworkDescription.h"

InputCurrentNeuronModel::InputCurrentNeuronModel() : EventDrivenInputDevice(MilisecondScale) {
	std::map<std::string, boost::any> param_map = InputCurrentNeuronModel::GetDefaultParameters();
	param_map["name"] = InputCurrentNeuronModel::GetName();
	this->SetParameters(param_map);
}

InputCurrentNeuronModel::~InputCurrentNeuronModel() {
}


enum NeuronModelOutputActivityType InputCurrentNeuronModel::GetModelOutputActivityType(){
	return OUTPUT_CURRENT;
}

ostream & InputCurrentNeuronModel::PrintInfo(ostream & out){
	out << "- Input Neuron Model: " << InputCurrentNeuronModel::GetName() << endl;

	return out;
}	
std::map<std::string,boost::any> InputCurrentNeuronModel::GetParameters() const {
	// Return a dictionary with the parameters
	std::map<std::string,boost::any> newMap = EventDrivenInputDevice::GetParameters();
	return newMap;
}

std::map<std::string, boost::any> InputCurrentNeuronModel::GetSpecificNeuronParameters(int index) const noexcept(false){
	return GetParameters();
}

void InputCurrentNeuronModel::SetParameters(std::map<std::string, boost::any> param_map) noexcept(false){
	// Search for the parameters in the dictionary
	EventDrivenInputDevice::SetParameters(param_map);
	return;
}


std::map<std::string,boost::any> InputCurrentNeuronModel::GetDefaultParameters() {
	// Return a dictionary with the parameters
	std::map<std::string, boost::any> newMap = EventDrivenInputDevice::GetDefaultParameters();

	return newMap;
}

NeuronModel* InputCurrentNeuronModel::CreateNeuronModel(ModelDescription nmDescription){
	InputCurrentNeuronModel * nmodel = new InputCurrentNeuronModel();
	nmodel->SetParameters(nmDescription.param_map);
	return nmodel;
}

ModelDescription InputCurrentNeuronModel::ParseNeuronModel(std::string FileName) noexcept(false){
	ModelDescription nmodel;
	nmodel.model_name = InputCurrentNeuronModel::GetName();

    nmodel.param_map["name"] = boost::any(InputCurrentNeuronModel::GetName());
	return nmodel;
}

std::string InputCurrentNeuronModel::GetName(){
	return "InputCurrentNeuronModel";
}

std::map<std::string, std::string> InputCurrentNeuronModel::GetNeuronModelInfo() {
	// Return a dictionary with the parameters
	std::map<std::string, std::string> newMap;
	newMap["info"] = std::string("CPU Event-driven input device able to propagate input current events (defined in pA) to other neurons");

	return newMap;
}


