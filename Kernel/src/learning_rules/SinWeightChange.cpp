/***************************************************************************
 *                           SinWeightChange.cpp                           *
 *                           -------------------                           *
 * copyright            : (C) 2009 by Jesus Garrido and Richard Carrillo   *
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


#include "../../include/learning_rules/SinWeightChange.h"

#include "../../include/learning_rules/SinState.h"

#include "../../include/spike/Interconnection.h"

SinWeightChange::SinWeightChange():AdditiveKernelChange(){
	// Set the default values for the learning rule parameters
	this->SetParameters(SinWeightChange::GetDefaultParameters());
}

SinWeightChange::~SinWeightChange(){
}


void SinWeightChange::InitializeConnectionState(unsigned int NumberOfSynapses, unsigned int NumberOfNeurons){
	this->State=(ConnectionState *) new SinState(NumberOfSynapses, this->exponent,this->kernelpeak);
}

int SinWeightChange::GetNumberOfVar() const{
	return this->exponent+2;
}

int SinWeightChange::GetExponent() const{
	return this->exponent;
}

ModelDescription SinWeightChange::ParseLearningRule(FILE * fh) noexcept(false) {
	ModelDescription lrule = AdditiveKernelChange::ParseLearningRule(fh);

	int exponent;
	if(fscanf(fh,"%i",&exponent)!=1){
		throw EDLUTException(TASK_LEARNING_RULE_LOAD, ERROR_LEARNING_RULE_LOAD, REPAIR_SIN_WEIGHT_CHANGE_LOAD);
	}
	if (exponent <= 0 || exponent % 2 == 1 || exponent > 20){
		throw EDLUTException(TASK_LEARNING_RULE_LOAD, ERROR_SIN_WEIGHT_CHANGE_EXPONENT, REPAIR_LEARNING_RULE_VALUES);
	}

	lrule.model_name = SinWeightChange::GetName();
	lrule.param_map["exp"] = boost::any(exponent);
	return lrule;
}

ostream & SinWeightChange::PrintInfo(ostream & out){
	out << "- SinAdditiveKernel Learning Rule: " << endl;
	out << "\t kernel_peak:" << this->kernelpeak << endl;
	out << "\t fixed_change: " << this->fixwchange << endl;
	out << "\t kernel_change: " << this->kernelwchange << endl;
	out << "\t exp: " << this->exponent << endl;
	return out;
}

void SinWeightChange::SetParameters(std::map<std::string, boost::any> param_map) noexcept(false){

	// Search for the parameters in the dictionary
	std::map<std::string,boost::any>::iterator it=param_map.find("exp");
	if (it!=param_map.end()){
		int newexponent = boost::any_cast<int>(it->second);
		if (newexponent<=0) {
			throw EDLUTException(TASK_LEARNING_RULE_LOAD, ERROR_SIN_WEIGHT_CHANGE_EXPONENT,
								 REPAIR_LEARNING_RULE_VALUES);
		}
		this->exponent = newexponent;
		param_map.erase(it);
	}

	AdditiveKernelChange::SetParameters(param_map);
}

LearningRule* SinWeightChange::CreateLearningRule(ModelDescription lrDescription){
	SinWeightChange * lrule = new SinWeightChange();
	lrule->SetParameters(lrDescription.param_map);
	return lrule;
}

std::map<std::string,boost::any> SinWeightChange::GetParameters(){
	std::map<std::string,boost::any> newMap = AdditiveKernelChange::GetParameters();
	newMap["exp"] = boost::any(this->exponent);
	return newMap;
}


std::map<std::string,boost::any> SinWeightChange::GetDefaultParameters(){
	std::map<std::string,boost::any> newMap = AdditiveKernelChange::GetDefaultParameters();
	newMap["exp"] = boost::any(2);
	return newMap;
}

