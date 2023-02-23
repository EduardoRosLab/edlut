/***************************************************************************
 *                           SimetricCosSTDPState.cpp                      *
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

#include "../../include/learning_rules/SimetricCosSTDPState.h"

#include "../../include/simulation/ExponentialTable.h"
#include "../../include/simulation/TrigonometricTable.h"

#include <cmath>
#include <stdio.h>
#include <float.h>

SimetricCosSTDPState::SimetricCosSTDPState(int NumSynapsesAndNeurons, float NewTau, float NewExponent): ConnectionState(NumSynapsesAndNeurons, 3), tau(NewTau), exponent(NewExponent){
	inv_tau=1.0f/tau;
}

SimetricCosSTDPState::~SimetricCosSTDPState() {
}

float SimetricCosSTDPState::GetPresynapticActivity(unsigned int index){
	return this->GetStateVariableAt(index, 0);
}

float SimetricCosSTDPState::GetPostsynapticActivity(unsigned int index){
	return this->GetStateVariableAt(index, 0);
}

unsigned int SimetricCosSTDPState::GetNumberOfPrintableValues(){
	return ConnectionState::GetNumberOfPrintableValues()+1;
}

double SimetricCosSTDPState::GetPrintableValuesAt(unsigned int index, unsigned int position){
	if (position<ConnectionState::GetNumberOfPrintableValues()){
		return ConnectionState::GetStateVariableAt(index, position);
	} else if (position==ConnectionState::GetNumberOfPrintableValues()) {
		return this->tau;
	} else return -1;
}




void SimetricCosSTDPState::SetNewUpdateTime(unsigned int index, double NewTime, bool pre_post){
	if(NewTime>this->GetLastUpdateTime(index)){
		float OldCos2= this->GetStateVariableAt(index, 0);
		float OldSin2= this->GetStateVariableAt(index, 1);
		float OldCosSin= this->GetStateVariableAt(index, 2);

		float ElapsedTime=float(NewTime -  this->GetLastUpdateTime(index));
		float ElapsedRelative = exponent*ElapsedTime*this->inv_tau;
		float expon = ExponentialTable::GetResult(-ElapsedRelative);

		float ElapsedRelativeTrigonometric=ElapsedTime*this->inv_tau*1.5708f;


		int LUTindex=TrigonometricTable::CalculateOffsetPosition(ElapsedRelativeTrigonometric);
		LUTindex = TrigonometricTable::CalculateValidPosition(0,LUTindex);

		float SinVar = TrigonometricTable::GetElement(LUTindex);
		float CosVar = TrigonometricTable::GetElement(LUTindex+1);

		float auxCos2=CosVar*CosVar;
		float auxSin2=SinVar*SinVar;
		float auxCosSin=CosVar*SinVar;


		float NewCos2 = expon*(OldCos2 * auxCos2 + OldSin2*auxSin2-2*OldCosSin*auxCosSin);
		float NewSin2 = expon*(OldSin2 * auxCos2 + OldCos2*auxSin2+2*OldCosSin*auxCosSin);
		float NewCosSin = expon*(OldCosSin *(auxCos2-auxSin2) + (OldCos2-OldSin2)*auxCosSin);

		this->SetStateVariableAt(index, 0, NewCos2, NewSin2, NewCosSin);


		this->SetLastUpdateTime(index, NewTime);
	}
}



void SimetricCosSTDPState::ApplyPresynapticSpike(unsigned int index){
	// Increment the activity in the state variable
	this->incrementStateVariableAt(index, 0, 1.0f);
}

void SimetricCosSTDPState::ApplyPostsynapticSpike(unsigned int index){
	// Increment the activity in the state variable
	this->incrementStateVariableAt(index, 0, 1.0f);
}
