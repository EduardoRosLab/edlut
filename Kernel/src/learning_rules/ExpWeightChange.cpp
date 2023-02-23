/***************************************************************************
 *                           ExpWeightChange.cpp                           *
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


#include "../../include/learning_rules/ExpWeightChange.h"

#include "../../include/learning_rules/ExpState.h"

#include "../../include/spike/Interconnection.h"

ExpWeightChange::ExpWeightChange():AdditiveKernelChange(){
}

ExpWeightChange::~ExpWeightChange(){
}


void ExpWeightChange::InitializeConnectionState(unsigned int NumberOfSynapses, unsigned int NumberOfNeurons){
	this->State=(ConnectionState *) new ExpState(NumberOfSynapses, this->kernelpeak);
}


ModelDescription ExpWeightChange::ParseLearningRule(FILE * fh) noexcept(false) {
	ModelDescription lrule = AdditiveKernelChange::ParseLearningRule(fh);
	lrule.model_name = ExpWeightChange::GetName();
	return lrule;
}

ostream & ExpWeightChange::PrintInfo(ostream & out){
	out << "- ExpAdditiveKernel Learning Rule: " << endl;
	out << "\t kernel_peak:" << this->kernelpeak << endl;
	out << "\t fixed_change: " << this->fixwchange << endl;
	out << "\t kernel_change: " << this->kernelwchange << endl;
	return out;
}

LearningRule* ExpWeightChange::CreateLearningRule(ModelDescription lrDescription){
	ExpWeightChange * lrule = new ExpWeightChange();
    lrule->SetParameters(lrDescription.param_map);
	return lrule;
}

std::map<std::string,boost::any> ExpWeightChange::GetDefaultParameters(){
	std::map<std::string,boost::any> newMap = AdditiveKernelChange::GetDefaultParameters();
	return newMap;
}