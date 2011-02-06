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
		return this->MaxChangeLTP*exp((tpre-tpost)/this->tauLTP);
	} else {
		return -this->MaxChangeLTD*exp((tpost-tpre)/this->tauLTD);
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

	if(!(fscanf(fh,"%f",&this->MaxChangeLTP)==1 && fscanf(fh,"%f",&this->tauLTP)==1 && fscanf(fh,"%f",&this->MaxChangeLTD)==1 && fscanf(fh,"%f",&this->tauLTD)==1)){
		throw EDLUTFileException(4,28,23,1,Currentline);
	}

}

ostream & STDPWeightChange::PrintInfo(ostream & out){

	out << "- STDP Learning Rule: LTD " << this->MaxChangeLTD << "\t" << this->tauLTD << "\tLTP " << this->MaxChangeLTP << "\t" << this->tauLTP << endl;

	return out;
}

int STDPWeightChange::GetNumberOfVar() const{
	return 0;
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

