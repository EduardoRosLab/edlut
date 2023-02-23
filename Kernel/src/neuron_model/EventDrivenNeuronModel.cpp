/***************************************************************************
 *                           EventDrivenNeuronModel.cpp                    *
 *                           -------------------                           *
 * copyright            : (C) 2011 by Jesus Garrido                        *
 * email                : jgarrido@atc.ugr.es                              *
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 3 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

#include "../../include/neuron_model/EventDrivenNeuronModel.h"

EventDrivenNeuronModel::EventDrivenNeuronModel(): NeuronModel() {

}

EventDrivenNeuronModel::~EventDrivenNeuronModel() {
}

enum NeuronModelSimulationMethod EventDrivenNeuronModel::GetModelSimulationMethod(){
	return EVENT_DRIVEN_MODEL;
}


enum NeuronModelType EventDrivenNeuronModel::GetModelType(){
	return NEURAL_LAYER;
}

std::map<std::string,boost::any> EventDrivenNeuronModel::GetParameters() const {
	// Return a dictionary with the parameters
	std::map<std::string,boost::any> newMap = NeuronModel::GetParameters();
	return newMap;
}

void EventDrivenNeuronModel::SetParameters(std::map<std::string, boost::any> param_map) noexcept(false){

	// Search for the parameters in the dictionary
	NeuronModel::SetParameters(param_map);
	return;
}

std::map<std::string,boost::any> EventDrivenNeuronModel::GetDefaultParameters() {
	// Return a dictionary with the parameters
	std::map<std::string,boost::any> newMap = NeuronModel::GetDefaultParameters();
	return newMap;
}