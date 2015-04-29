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


STDPLSWeightChange::STDPLSWeightChange(int NewLearningRuleIndex):STDPWeightChange(NewLearningRuleIndex){
}

STDPLSWeightChange::~STDPLSWeightChange(){
}


void STDPLSWeightChange::InitializeConnectionState(unsigned int NumberOfSynapses){
	this->State=(ConnectionState *) new STDPLSState(NumberOfSynapses, this->tauLTP, this->tauLTD);
}

ostream & STDPLSWeightChange::PrintInfo(ostream & out){

	out << "- STDP Last Spike Learning Rule: LTD " << this->MaxChangeLTD << "\t" << this->tauLTD << "\tLTP " << this->MaxChangeLTP << "\t" << this->tauLTP << endl;

	return out;
}
