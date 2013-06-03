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

#include <cmath>

ExpState::ExpState(float NewTau): ConnectionState(2), tau(NewTau){
	for (int i=0; i<2; ++i){
		ConnectionState::SetStateVariableAt(i,0); // Initialize presynaptic activity
	}

	if (this->tau==0){
		this->tau = 1e-6;
	}
}

ExpState::~ExpState() {
}

unsigned int ExpState::GetNumberOfPrintableValues(){
	return ConnectionState::GetNumberOfPrintableValues()+1;
}

double ExpState::GetPrintableValuesAt(unsigned int position){
	if (position<ConnectionState::GetNumberOfPrintableValues()){
		return ConnectionState::GetStateVariableAt(position);
	} else if (position==ConnectionState::GetNumberOfPrintableValues()) {
		return this->tau;
	} else return -1;
}

float ExpState::GetPresynapticActivity(){
	return this->GetStateVariableAt(0);
}

float ExpState::GetPostsynapticActivity(){
	return 0;
}


void ExpState::AddElapsedTime(float ElapsedTime){
	float factor = ElapsedTime/this->tau;
	float expon = exp(-factor);

	// Update the activity value
	float OldExpon1 = this->GetStateVariableAt(1);
	float OldExpon = this->GetStateVariableAt(0);

	float NewExpon1 = OldExpon1*expon;
	float NewExpon = (OldExpon+factor*OldExpon1)*expon;

	this->SetStateVariableAt(1,NewExpon1);
	this->SetStateVariableAt(0,NewExpon);

	this->SetLastUpdateTime(this->GetLastUpdateTime()+ElapsedTime);
}

void ExpState::ApplyPresynapticSpike(){
	float OldExpon = this->GetStateVariableAt(1);

	this->SetStateVariableAt(1,OldExpon+1);
}

void ExpState::ApplyPostsynapticSpike(){
	return;
}


