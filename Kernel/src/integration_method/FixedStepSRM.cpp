/***************************************************************************
 *                           FixedStepSRM.cpp                              *
 *                           -------------------                           *
 * copyright            : (C) 2013 by Francisco Naveros                    *
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

#include "../../include/integration_method/FixedStepSRM.h"
#include "../../include/neuron_model/TimeDrivenNeuronModel.h"


FixedStepSRM::FixedStepSRM(TimeDrivenNeuronModel * NewModel) : FixedStep(NewModel){
	this->SetParameters(FixedStepSRM::GetDefaultParameters());
}

FixedStepSRM::~FixedStepSRM(){

}


std::ostream & FixedStepSRM::PrintInfo(std::ostream & out){
	out << "Integration Method Type: " << FixedStepSRM::GetName() << endl;

	return out;
}

void FixedStepSRM::SetParameters(std::map<std::string, boost::any> param_map) noexcept(false){
	// Search for the parameters in the dictionary
	FixedStep::SetParameters(param_map);
}

std::map<std::string,boost::any> FixedStepSRM::GetParameters() const{
	// Return a dictionary with the parameters
	std::map<std::string,boost::any> newMap = FixedStep::GetParameters();
	newMap["Name"] = FixedStepSRM::GetName();
	return newMap;
}

std::map<std::string,boost::any> FixedStepSRM::GetDefaultParameters(){
	std::map<std::string,boost::any> newMap = FixedStep::GetDefaultParameters();
	newMap["Name"] = FixedStepSRM::GetName();
	return newMap;
}

ModelDescription FixedStepSRM::ParseIntegrationMethod(FILE * fh) noexcept(false){
	ModelDescription nmodel = FixedStep::ParseIntegrationMethod(fh);
	nmodel.model_name = FixedStepSRM::GetName();
	return nmodel;
}

std::string FixedStepSRM::GetName() {
	return "FixedStepSRM";
}

IntegrationMethod* FixedStepSRM::CreateIntegrationMethod(ModelDescription nmDescription, TimeDrivenNeuronModel *nmodel){
	FixedStepSRM * newmodel = new FixedStepSRM();
	newmodel->model = nmodel;
	newmodel->SetParameters(nmDescription.param_map);
	return newmodel;
}