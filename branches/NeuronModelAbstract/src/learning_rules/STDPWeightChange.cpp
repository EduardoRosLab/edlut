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

#include "../../include/spike/Interconnection.h"
#include "../../include/spike/Neuron.h"

#include "../../include/simulation/Utils.h"

#include "../../include/neuron_model/NeuronState.h"

double STDPWeightChange::GetWeightChange(Interconnection * Connection, double CurrentTime){
	Neuron * target = Connection->GetTarget();

	double tpre = Connection->GetLastSpikeTime();
	double tpost = CurrentTime - target->GetNeuronState()->GetLastSpikeTime();

	if (tpre<tpost){
		return this->MaxChange*exp((tpre-tpost)/this->tau);
	} else {
		return -this->MaxChange*exp((tpost-tpre)/this->tau);
	}
}

void STDPWeightChange::ApplyPreSynapticSpike(Interconnection * Connection,double SpikeTime){

	float newWeight = Connection->GetWeight();

	newWeight += this->GetWeightChange(Connection,SpikeTime);

	if (newWeight > Connection->GetMaxWeight()) {
		newWeight = Connection->GetMaxWeight();
	} else if (newWeight < 0.0){
		newWeight = 0;
	}

	Connection->SetWeight(newWeight);

	return;
}

void STDPWeightChange::ApplyPostSynapticSpike(Interconnection * Connection,double SpikeTime){
	float newWeight = Connection->GetWeight();

	newWeight += this->GetWeightChange(Connection,SpikeTime);

	if (newWeight > Connection->GetMaxWeight()) {
		newWeight = Connection->GetMaxWeight();
	} else if (newWeight < 0.0){
		newWeight = 0;
	}

	Connection->SetWeight(newWeight);

	return;
}


void STDPWeightChange::LoadLearningRule(FILE * fh, long & Currentline) throw (EDLUTFileException){
	skip_comments(fh,Currentline);

	if(!(fscanf(fh,"%f",&this->MaxChange)==1 && fscanf(fh,"%f",&this->tau)==1)){
		throw EDLUTFileException(4,28,23,1,Currentline);
	}

}

ostream & STDPWeightChange::PrintInfo(ostream & out){

	out << "- STDP Learning Rule: " << this->MaxChange << "\t" << this->tau << endl;

	return out;
}

int STDPWeightChange::GetNumberOfVar() const{
	return 0;
}

