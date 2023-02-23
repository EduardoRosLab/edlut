/***************************************************************************
 *                           TimeDrivenModel.cpp                           *
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

#include "neuron_model/TimeDrivenModel.h"
//#include "../../include/neuron_model/NeuronModel.h"


TimeDrivenModel::TimeDrivenModel(TimeScale new_time_scale) : NeuronModel(new_time_scale), time_driven_step_size(0.0) {
}


TimeDrivenModel::~TimeDrivenModel() {
}

void TimeDrivenModel::SetTimeDrivenStepSize(double step){
	this->time_driven_step_size = step;
}


double TimeDrivenModel::GetTimeDrivenStepSize(){
	return this->time_driven_step_size;
}


std::map<std::string,boost::any> TimeDrivenModel::GetParameters() const {
	// Return a dictionary with the parameters
	std::map<std::string,boost::any> newMap = NeuronModel::GetParameters();
	return newMap;
}

void TimeDrivenModel::SetParameters(std::map<std::string, boost::any> param_map) noexcept(false){

	// Search for the parameters in the dictionary
	NeuronModel::SetParameters(param_map);
	return;
}

std::map<std::string,boost::any> TimeDrivenModel::GetDefaultParameters() {
	// Return a dictionary with the parameters
	std::map<std::string,boost::any> newMap = NeuronModel::GetDefaultParameters();
	return newMap;
}