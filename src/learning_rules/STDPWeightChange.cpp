/***************************************************************************
 *                           STDPWeightChange.cpp                          *
 *                           -------------------                           *
 * copyright            : (C) 2010 by Jesus Garrido                        *
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

#include "../../include/learning_rules/STDPWeightChange.h"

#include "../../include/learning_rules/STDPState.h"

#include "../../include/spike/Interconnection.h"

#include "../../include/simulation/Utils.h"

#include "../../include/neuron_model/NeuronState.h"


STDPWeightChange::STDPWeightChange():WithPostSynaptic(){
}

STDPWeightChange::~STDPWeightChange(){

}


void STDPWeightChange::InitializeConnectionState(unsigned int NumberOfSynapses){
	this->State=(ConnectionState *) new STDPState(NumberOfSynapses, this->tauLTP, this->tauLTD);
}

void STDPWeightChange::ApplyPreSynapticSpike(Interconnection * Connection,double SpikeTime){
	unsigned int LearningRuleIndex = Connection->GetLearningRuleIndex_withPost();

	// Apply synaptic activity decaying rule
	State->SetNewUpdateTime(LearningRuleIndex, SpikeTime, false);

	// Apply presynaptic spike
	State->ApplyPresynapticSpike(LearningRuleIndex);

	// Apply weight change
	Connection->IncrementWeight(-this->MaxChangeLTD*State->GetPostsynapticActivity(LearningRuleIndex));

	return;
}

void STDPWeightChange::ApplyPostSynapticSpike(Interconnection * Connection,double SpikeTime){
	int LearningRuleIndex = Connection->GetLearningRuleIndex_withPost();
	
	// Apply synaptic activity decaying rule
	State->SetNewUpdateTime(LearningRuleIndex, SpikeTime, true);

	// Apply postsynaptic spike
	State->ApplyPostsynapticSpike(LearningRuleIndex);

	// Apply weight change
	Connection->IncrementWeight(this->MaxChangeLTP*State->GetPresynapticActivity(LearningRuleIndex));

	return;
}


void STDPWeightChange::LoadLearningRule(FILE * fh, long & Currentline) throw (EDLUTFileException){
	skip_comments(fh,Currentline);

	if(!(fscanf(fh,"%f",&this->MaxChangeLTP)==1 && fscanf(fh,"%f",&this->tauLTP)==1 && fscanf(fh,"%f",&this->MaxChangeLTD)==1 && fscanf(fh,"%f",&this->tauLTD)==1)){
		throw EDLUTFileException(4,28,23,1,Currentline);
	}

}

ostream & STDPWeightChange::PrintInfo(ostream & out){

	out << "- STDP Learning Rule: LTD " << this->MaxChangeLTD << "\t" << this->tauLTD << "\tLTP " << this->MaxChangeLTP << "\t" << this->tauLTP << endl;

	return out;
}

float STDPWeightChange::GetMaxWeightChangeLTP() const{
	return this->MaxChangeLTP;
}

void STDPWeightChange::SetMaxWeightChangeLTP(float NewMaxChange){
	this->MaxChangeLTP = NewMaxChange;
}

float STDPWeightChange::GetMaxWeightChangeLTD() const{
	return this->MaxChangeLTD;
}

void STDPWeightChange::SetMaxWeightChangeLTD(float NewMaxChange){
	this->MaxChangeLTD = NewMaxChange;
}


