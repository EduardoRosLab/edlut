/***************************************************************************
 *                           ExpState.cpp                                  *
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

#include "../../include/learning_rules/ExpState.h"

#include "../../include/simulation/ExponentialTable.h"

#include <cmath>
#include <stdio.h>

ExpState::ExpState(unsigned int NumSynapses, float NewTau): ConnectionState(NumSynapses, 2), tau(NewTau){
	if (this->tau==0){
		this->tau = 1e-6;
	}
	inv_tau=1.0f/tau;
}

ExpState::~ExpState() {
}

unsigned int ExpState::GetNumberOfPrintableValues(){
	return ConnectionState::GetNumberOfPrintableValues()+1;
}

double ExpState::GetPrintableValuesAt(unsigned int index, unsigned int position){
	if (position<ConnectionState::GetNumberOfPrintableValues()){
		return ConnectionState::GetStateVariableAt(index, position);
	} else if (position==ConnectionState::GetNumberOfPrintableValues()) {
		return this->tau;
	} else return -1;
}

float ExpState::GetPresynapticActivity(unsigned int index){
	return this->GetStateVariableAt(index, 0);
}

float ExpState::GetPostsynapticActivity(unsigned int index){
	return 0.0f;
}



void ExpState::SetNewUpdateTime(unsigned int index, double NewTime, bool pre_post){
	float ElapsedTime=float(NewTime -  this->GetLastUpdateTime(index));
	float factor = ElapsedTime*this->inv_tau;
	float expon = ExponentialTable::GetResult(-factor);

	// Update the activity value
	float OldExpon = this->GetStateVariableAt(index, 0);
	float OldExpon1 = this->GetStateVariableAt(index, 1);

	float NewExpon = 2.7183*(OldExpon+factor*OldExpon1)*expon;
	float NewExpon1 = OldExpon1*expon;

	this->SetStateVariableAt(index, 0, NewExpon);
	this->SetStateVariableAt(index, 1, NewExpon1);

	this->SetLastUpdateTime(index, NewTime);
}



void ExpState::ApplyPresynapticSpike(unsigned int index){
	this->incrementStateVariableAt(index, 1, 1.0f);
}

void ExpState::ApplyPostsynapticSpike(unsigned int index){
	return;
}
