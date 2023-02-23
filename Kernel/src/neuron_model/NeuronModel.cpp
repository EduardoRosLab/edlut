/***************************************************************************
 *                           NeuronModel.cpp                               *
 *                           -------------------                           *
 * copyright            : (C) 2012 by Jesus Garrido and Francisco Naveros  *
 * email                : jgarrido@atc.ugr.es, fnaveros@ugr.es             *
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 3 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

#include "neuron_model/NeuronModel.h"

#include "neuron_model/VectorNeuronState.h"
#include "spike/Interconnection.h"
#include "spike/Neuron.h"
#include "spike/NeuronModelPropagationDelayStructure.h"



NeuronModel::NeuronModel() : State(0), time_scale(0.0), inv_time_scale(0.0) {
	PropagationStructure = new NeuronModelPropagationDelayStructure();
}

NeuronModel::NeuronModel(TimeScale new_time_scale) : State(0), time_scale(new_time_scale), inv_time_scale(1.0f/new_time_scale) {
	PropagationStructure=new NeuronModelPropagationDelayStructure();
}


NeuronModel::~NeuronModel() {
	if (this->State!=0){
		delete this->State;
	}

	if(this->PropagationStructure){
		delete this->PropagationStructure;
	}
}

NeuronModelPropagationDelayStructure * NeuronModel::GetNeuronModelPropagationDelayStructure(){
	return PropagationStructure;
}

void NeuronModel::SetTimeScale(float new_time_scale){
	this->time_scale=new_time_scale;
	this->inv_time_scale=1.0f/new_time_scale;
}

float NeuronModel::GetTimeScale() const{
    return this->time_scale;
}



void NeuronModel::SetParameters(std::map<std::string, boost::any> param_map) noexcept(false){
    // Search for the parameters in the dictionary
    std::map<std::string,boost::any>::iterator it=param_map.find("name");
    if (it!=param_map.end()){
        std::string new_param = boost::any_cast<std::string>(it->second);
        this->name = new_param;
        param_map.erase(it);
    }

    if (!param_map.empty()){
		for (auto& param : param_map) {
			cout << param.first << " ";
		}
		throw EDLUTException(TASK_NEURON_MODEL_SET,ERROR_NEURON_MODEL_UNKNOWN_PARAMETER, REPAIR_NEURON_MODEL_PARAMETER_NAME);
	}
}

void NeuronModel::SetSpecificNeuronParameters(int index, std::map<std::string, boost::any> param_map) noexcept(false){
	if (!param_map.empty()){
		cout << "Parameters ";
		for (std::map<std::string, boost::any>::iterator it = param_map.begin(); it != param_map.end(); ++it){
			cout << "\"" << it->first << "\", ";
		}
		cout << "are not valid or can not be fixed for a specific neuron." << endl;
		throw EDLUTException(TASK_NEURON_MODEL_SET, ERROR_NEURON_MODEL_UNKNOWN_PARAMETER, REPAIR_NEURON_MODEL_PARAMETER_NAME);
	}
}


std::map<std::string,boost::any> NeuronModel::GetParameters() const{
	// Return a dictionary with the parameters
    std::map<std::string,boost::any> newMap;
    newMap["name"] = boost::any(this->name);
    return newMap;
}

std::map<std::string,boost::any> NeuronModel::GetDefaultParameters(){
    std::map<std::string,boost::any> newMap;
    return newMap;
}

std::string NeuronModel::GetNeuronModelName() const {
    return this->name;
}