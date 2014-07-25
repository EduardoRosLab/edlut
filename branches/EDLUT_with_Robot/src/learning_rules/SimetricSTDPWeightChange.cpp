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

#include "../../include/learning_rules/SimetricSTDPWeightChange.h"

#include "../../include/learning_rules/SimetricSTDPState.h"

#include "../../include/spike/Interconnection.h"

#include "../../include/simulation/Utils.h"

#include "../../include/neuron_model/NeuronState.h"
#include "../../include/spike/Neuron.h"


SimetricSTDPWeightChange::SimetricSTDPWeightChange():WithPostSynaptic(){
}

SimetricSTDPWeightChange::~SimetricSTDPWeightChange(){

}


void SimetricSTDPWeightChange::InitializeConnectionState(unsigned int NumberOfSynapsesAndNeurons){
	this->State=(ConnectionState *) new SimetricSTDPState(NumberOfSynapsesAndNeurons, this->tau);
}

void SimetricSTDPWeightChange::ApplyPreSynapticSpike(Interconnection * Connection,double SpikeTime){

	//Increment propagate spike kernel previous to the "future internal spike"
	int LearningRuleIndex = Connection->GetLearningRuleIndex_withPost();

	// Update the presynaptic activity
	State->SetNewUpdateTime(LearningRuleIndex, SpikeTime, false);

	// Add the presynaptic spike influence
	State->ApplyPresynapticSpike(LearningRuleIndex);



	//aplicate internal spike kernel to "future propagate spike"
	int SecondLearningRuleIndex = Connection->GetTarget()->GetIndex();

	State->SetNewUpdateTime(SecondLearningRuleIndex, SpikeTime, false);

	Connection->IncrementWeight((this->MaxChangeLTP+this->MaxChangeLTD)*State->GetPresynapticActivity(SecondLearningRuleIndex) - this->MaxChangeLTD);
}

void SimetricSTDPWeightChange::ApplyPostSynapticSpike(Interconnection * Connection,double SpikeTime){
	//increment internal spike kernel previous to "future propagate spike"
	int SecondLearningRuleIndex = Connection->GetTarget()->GetIndex();
	if(SpikeTime < State->GetLastUpdateTime(SecondLearningRuleIndex)){
		State->SetNewUpdateTime(SecondLearningRuleIndex, SpikeTime, false);

		State->ApplyPresynapticSpike(SecondLearningRuleIndex);
	}



	//Aplicate propagate spike kernel to "future internal spike"
	int LearningRuleIndex = Connection->GetLearningRuleIndex_withPost();

	// Update the presynaptic activity
	State->SetNewUpdateTime(LearningRuleIndex, SpikeTime, false);

	// Update synaptic weight
	Connection->IncrementWeight((this->MaxChangeLTP+this->MaxChangeLTD)*State->GetPresynapticActivity(LearningRuleIndex) - this->MaxChangeLTD);



	return;
}


void SimetricSTDPWeightChange::LoadLearningRule(FILE * fh, long & Currentline) throw (EDLUTFileException){
	skip_comments(fh,Currentline);

	if(!(fscanf(fh,"%f",&this->tau)==1 && fscanf(fh,"%f",&this->MaxChangeLTP)==1 && fscanf(fh,"%f",&this->MaxChangeLTD)==1)){
		throw EDLUTFileException(4,28,23,1,Currentline);
	}

}

ostream & SimetricSTDPWeightChange::PrintInfo(ostream & out){

	out << "- SimetricSTDP Learning Rule: LTD " << this->MaxChangeLTD << "\tLTP " << this->MaxChangeLTP << "\tTau" << this->tau << endl;

	return out;
}

float SimetricSTDPWeightChange::GetMaxWeightChangeLTP() const{
	return this->MaxChangeLTP;
}

void SimetricSTDPWeightChange::SetMaxWeightChangeLTP(float NewMaxChange){
	this->MaxChangeLTP = NewMaxChange;
}

float SimetricSTDPWeightChange::GetMaxWeightChangeLTD() const{
	return this->MaxChangeLTD;
}

void SimetricSTDPWeightChange::SetMaxWeightChangeLTD(float NewMaxChange){
	this->MaxChangeLTD = NewMaxChange;
}


