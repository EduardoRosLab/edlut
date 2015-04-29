/***************************************************************************
 *                           SimetricCosSTDPWeightChange.cpp               *
 *                           -------------------                           *
 * copyright            : (C) 2015 by Francisco Naveros                    *
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

#include "../../include/learning_rules/SimetricCosSTDPWeightChange.h"

#include "../../include/learning_rules/SimetricCosSTDPState.h"

#include "../../include/spike/Interconnection.h"

#include "../../include/simulation/Utils.h"

#include "../../include/neuron_model/NeuronState.h"
#include "../../include/spike/Neuron.h"


SimetricCosSTDPWeightChange::SimetricCosSTDPWeightChange(int NewLearningRuleIndex):WithPostSynaptic(NewLearningRuleIndex){
}

SimetricCosSTDPWeightChange::~SimetricCosSTDPWeightChange(){

}


void SimetricCosSTDPWeightChange::InitializeConnectionState(unsigned int NumberOfSynapsesAndNeurons){
	this->State=(ConnectionState *) new SimetricCosSTDPState(NumberOfSynapsesAndNeurons, this->tau, this->exponent);
}

void SimetricCosSTDPWeightChange::ApplyPreSynapticSpike(Interconnection * Connection,double SpikeTime){

	int LearningRuleIndex = Connection->GetLearningRuleIndex_withPost();

	//LTP
	// Second case: the weight change is linked to this connection
	Connection->IncrementWeight(this->a1pre);

	// Update the presynaptic activity
	State->SetNewUpdateTime(LearningRuleIndex, SpikeTime, false);

	// Add the presynaptic spike influence
	State->ApplyPresynapticSpike(LearningRuleIndex);


	//LTD
	//aplicate internal spike kernel to "future propagate spike"
	int SecondLearningRuleIndex = Connection->GetTarget()->GetIndex();

	// Update the presynaptic activity
	State->SetNewUpdateTime(SecondLearningRuleIndex, SpikeTime, false);

	// Update synaptic weight
	Connection->IncrementWeight(this->a2prepre*State->GetPresynapticActivity(SecondLearningRuleIndex));
}

void SimetricCosSTDPWeightChange::ApplyPostSynapticSpike(Interconnection * Connection,double SpikeTime){
	//increment internal spike kernel previous to "future propagate spike"
	int SecondLearningRuleIndex = Connection->GetTarget()->GetIndex();
	if(SpikeTime > State->GetLastUpdateTime(SecondLearningRuleIndex)){
		State->SetNewUpdateTime(SecondLearningRuleIndex, SpikeTime, false);

		State->ApplyPresynapticSpike(SecondLearningRuleIndex);
	}

	//Aplicate propagate spike kernel to "future internal spike"
	int LearningRuleIndex = Connection->GetLearningRuleIndex_withPost();

	// Update the presynaptic activity
	State->SetNewUpdateTime(LearningRuleIndex, SpikeTime, false);


	// Update synaptic weight
	Connection->IncrementWeight(this->a2prepre*State->GetPresynapticActivity(LearningRuleIndex));

	return;
}


void SimetricCosSTDPWeightChange::LoadLearningRule(FILE * fh, long & Currentline) throw (EDLUTFileException){
	skip_comments(fh,Currentline);

	if(fscanf(fh,"%f",&this->tau)==1 && fscanf(fh,"%f",&this->exponent)==1 && fscanf(fh,"%f",&this->a1pre)==1 && fscanf(fh,"%f",&this->a2prepre)==1){
		if(this->a1pre < -1.0 || this->a1pre > 1.0){
			throw EDLUTFileException(4,27,22,1,Currentline);
		}
	}else{
		throw EDLUTFileException(4,28,23,1,Currentline);
	}
}

ostream & SimetricCosSTDPWeightChange::PrintInfo(ostream & out){

	out << "- Simetric Cos Kernel Learning Rule: " << this->tau << "\t" << this->exponent<< "\t" << this->a1pre << "\t" << this->a2prepre << endl;


	return out;
}
