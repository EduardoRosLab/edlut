/***************************************************************************
 *                           SimetricSTDPState.cpp                         *
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

#include "../../include/learning_rules/SimetricSTDPState.h"

#include "../../include/simulation/ExponentialTable.h"


#include <cmath>
#include <stdio.h>
#include <float.h>

SimetricSTDPState::SimetricSTDPState(int NumSynapsesAndNeurons, float NewTau): ConnectionState(NumSynapsesAndNeurons, 3), tau(NewTau){
	if (this->tau==0){
		this->tau = 0.5f;
	}
	inv_tau=1.0f/tau;
}

SimetricSTDPState::~SimetricSTDPState() {
}

unsigned int SimetricSTDPState::GetNumberOfPrintableValues(){
	return ConnectionState::GetNumberOfPrintableValues()+1;
}

double SimetricSTDPState::GetPrintableValuesAt(unsigned int position){
	if (position<ConnectionState::GetNumberOfPrintableValues()){
		return ConnectionState::GetStateVariableAt(0, position);
	} else if (position==ConnectionState::GetNumberOfPrintableValues()) {
		return this->tau;
	} else return -1;
}




void SimetricSTDPState::SetNewUpdateTime(unsigned int index, double NewTime, bool pre_post){
	if(NewTime>this->GetLastUpdateTime(index)){
		float OldCos2= this->GetStateVariableAt(index, 0);
		float OldSin2= this->GetStateVariableAt(index, 1);
		float OldCosSin= this->GetStateVariableAt(index, 2);

		float ElapsedTime=float(NewTime -  this->GetLastUpdateTime(index));
		float ElapsedRelative = ElapsedTime*this->inv_tau;
		float expon = ExponentialTable::GetResult(-ElapsedRelative);


		float auxCos2=cos(ElapsedRelative);
		float auxSin2=sin(ElapsedRelative);
		float auxCosSin=auxCos2*auxSin2;
		auxCos2*=auxCos2;
		auxSin2*=auxSin2;

		
		float NewCos2 = expon*(OldCos2 * auxCos2 + OldSin2*auxSin2-2*OldCosSin*auxCosSin);
		float NewSin2 = expon*(OldSin2 * auxCos2 + OldCos2*auxSin2+2*OldCosSin*auxCosSin);
		float NewCosSin = expon*(OldCosSin *(auxCos2-auxSin2) + (OldCos2-OldSin2)*auxCosSin);

		this->SetStateVariableAt(index, 0, NewCos2, NewSin2, NewCosSin);


		this->SetLastUpdateTime(index, NewTime);
	}
}



void SimetricSTDPState::ApplyPresynapticSpike(unsigned int index){
	// Increment the activity in the state variable
	this->incrementStateVaraibleAt(index, 0, 1.0f);
}

void SimetricSTDPState::ApplyPostsynapticSpike(unsigned int index){
	// Increment the activity in the state variable
	return; 
}

