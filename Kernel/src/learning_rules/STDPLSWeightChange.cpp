/***************************************************************************
 *                           STDPLSWeightChange.cpp                        *
 *                           -------------------                           *
 * copyright            : (C) 2013 by Jesus Garrido                        *
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

#include "../../include/learning_rules/STDPLSWeightChange.h"

#include "../../include/learning_rules/STDPLSState.h"

#include "../../include/spike/Interconnection.h"

#include "../../include/simulation/Utils.h"

#include "../../include/neuron_model/NeuronState.h"


STDPLSWeightChange::STDPLSWeightChange():STDPWeightChange(){
	this->SetParameters(STDPLSWeightChange::GetDefaultParameters());
}

STDPLSWeightChange::~STDPLSWeightChange(){
}


void STDPLSWeightChange::InitializeConnectionState(unsigned int NumberOfSynapses, unsigned int NumberOfNeurons){
	this->State=(ConnectionState *) new STDPLSState(NumberOfSynapses, this->tauLTP, this->tauLTD);
}

ostream & STDPLSWeightChange::PrintInfo(ostream & out){
	out << "- STDPLS Learning Rule: " << endl;
	out << "\t max_LTP:" << this->MaxChangeLTP << endl;
	out << "\t tau_LTP:" << this->tauLTP << endl;
	out << "\t max_LTD:" << this->MaxChangeLTD << endl;
	out << "\t tau_LTD:" << this->tauLTD << endl;
	return out;
}

ModelDescription STDPLSWeightChange::ParseLearningRule(FILE * fh) noexcept(false) {
	ModelDescription lrule = STDPWeightChange::ParseLearningRule(fh);
	lrule.model_name = STDPLSWeightChange::GetName();
	return lrule;
}

LearningRule* STDPLSWeightChange::CreateLearningRule(ModelDescription lrDescription){
	STDPLSWeightChange * lrule = new STDPLSWeightChange();
	lrule->SetParameters(lrDescription.param_map);
	return lrule;
}

std::map<std::string,boost::any> STDPLSWeightChange::GetDefaultParameters(){
	std::map<std::string,boost::any> newMap = STDPWeightChange::GetDefaultParameters();
	return newMap;
}


