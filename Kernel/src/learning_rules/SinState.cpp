/***************************************************************************
 *                           SinState.cpp                                  *
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

#include "../../include/learning_rules/SinState.h"

#include "../../include/simulation/ExponentialTable.h"
#include "../../include/simulation/TrigonometricTable.h"


#include <string.h>

#include <iostream>
#include <cstdio>
using namespace std;

#define A 1.0f/2.0f



const float SinState::terms[11][11]  = {{1,0,0,0,0,0,0,0,0,0,0},
	{A,-A,0,0,0,0,0,0,0,0,0},
	{3.0f/2.0f*pow(A,2),-4.0f/2.0f*pow(A,2),1.0f/2.0f*pow(A,2),0,0,0,0,0,0,0,0},
	{10.0f/4.0f*pow(A,3),-15.0f/4.0f*pow(A,3),6.0f/4.0f*pow(A,3),-1.0f/4.0f*pow(A,3),0,0,0,0,0,0,0},
	{35.0f/8.0f*pow(A,4),-56.0f/8.0f*pow(A,4),28.0f/8.0f*pow(A,4),-8.0f/8.0f*pow(A,4),1.0f/8.0f*pow(A,4),0,0,0,0,0,0},
	{126.0f/16.0f*pow(A,5),-210.0f/16.0f*pow(A,5),120.0f/16.0f*pow(A,5),-45.0f/16.0f*pow(A,5),10.0f/16.0f*pow(A,5),-1.0f/16.0f*pow(A,5),0,0,0,0,0},
	{231.0f/16.0f*pow(A,6),-99.0f/4.0f*pow(A,6),495.0f/32.0f*pow(A,6),-55.0f/8.0f*pow(A,6),66.0f/32.0f*pow(A,6),-3.0f/8.0f*pow(A,6),1.0f/32.0f*pow(A,6),0,0,0,0},
	{429.0f/16.0f*pow(A,7),-3003.0f/64.0f*pow(A,7),1001.0f/32.0f*pow(A,7),-1001.0f/64.0f*pow(A,7),91.0f/16.0f*pow(A,7),-91.0f/64.0f*pow(A,7),7.0f/32.0f*pow(A,7),-1.0f/64.0f*pow(A,7),0,0,0},
	{6435.0f/128.0f*pow(A,8),-715.0f/8.0f*pow(A,8),1001.0f/16.0f*pow(A,8),-273.0f/8.0f*pow(A,8),455.0f/32.0f*pow(A,8),-35.0f/8.0f*pow(A,8),15.0f/16.0f*pow(A,8),-1.0f/8.0f*pow(A,8),1.0f/128.0f*pow(A,8),0,0},
	{12155.0f/128.0f*pow(A,9),-21879.0f/128.0f*pow(A,9),1989.0f/16.0f*pow(A,9),-4641.0f/64.0f*pow(A,9),1071.0f/32.0f*pow(A,9),-765.0f/64.0f*pow(A,9),51.0f/16.0f*pow(A,9),-153.0f/256.0f*pow(A,9),9.0f/128.0f*pow(A,9),-1.0f/256.0f*pow(A,9),0},
	{46189.0f/256.0f*pow(A,10),-20995.0f/64.0f*pow(A,10),62985.0f/256.0f*pow(A,10),-4845.0f/32.0f*pow(A,10),4845.0f/64.0f*pow(A,10),-969.0f/32.0f*pow(A,10),4845.0f/512.0f*pow(A,10),-285.0f/128.0f*pow(A,10),95.0f/256.0f*pow(A,10),-5.0f/128.0f*pow(A,10),1.0f/512.0f*pow(A,10)}};


SinState::SinState(unsigned int NumSynapses, unsigned int NewExponent, float NewMaxpos): ConnectionState(NumSynapses, NewExponent+2), exponent(NewExponent), maxpos(NewMaxpos){

	this->tau = this->maxpos/atan((float)exponent);
	this->factor = 1.0f/(exp(-atan((float)this->exponent))*pow(sin(atan((float)this->exponent)),(int) this->exponent));

	if (this->tau==0){
		this->tau = 1e-6;
	}
	inv_tau=1.0f/tau;

	unsigned int ExponenLine = NewExponent/2;

	TermPointer = this->terms[ExponenLine];
}

SinState::~SinState() {
}

float SinState::GetPresynapticActivity(unsigned int index){
	return this->GetStateVariableAt(index, 0);
}

float SinState::GetPostsynapticActivity(unsigned int index){
	return 0.0f;
}

unsigned int SinState::GetNumberOfPrintableValues(){
	return ConnectionState::GetNumberOfPrintableValues()+2;
}

double SinState::GetPrintableValuesAt(unsigned int index, unsigned int position){
	if (position<ConnectionState::GetNumberOfPrintableValues()){
		return ConnectionState::GetStateVariableAt(index,position);
	} else if (position==ConnectionState::GetNumberOfPrintableValues()) {
		return this->exponent;
	} else if (position==ConnectionState::GetNumberOfPrintableValues()+1) {
		return this->maxpos;
	} else return -1;
}


void SinState::SetNewUpdateTime (unsigned int index, double NewTime, bool pre_post){
	// Update the activity value
	float OldExpon = this->GetStateVariableAt(index, 1);

	float ElapsedTime = float(NewTime - this->GetLastUpdateTime(index));
	float ElapsedRelative = ElapsedTime*this->inv_tau;

	float expon = ExponentialTable::GetResult(-ElapsedRelative);

	this->SetLastUpdateTime(index, NewTime);

	float NewExpon = OldExpon * expon;
	float NewActivity =NewExpon*TermPointer[0];

	int aux=TrigonometricTable::CalculateOffsetPosition(2*ElapsedRelative);
	int LUTindex=0;

	float SinVar, CosVar, OldVarCos, OldVarSin, NewVarCos, NewVarSin;
	int grade, offset;
	for (grade=2, offset=1; grade<=this->exponent; grade+=2, offset++){

		LUTindex =TrigonometricTable::CalculateValidPosition(LUTindex,aux);

		OldVarCos = this->GetStateVariableAt(index, grade);
		OldVarSin = this->GetStateVariableAt(index, grade + 1);

		SinVar = TrigonometricTable::GetElement(LUTindex);
		CosVar = TrigonometricTable::GetElement(LUTindex+1);

		NewVarCos = (OldVarCos*CosVar-OldVarSin*SinVar)*expon;
		NewVarSin = (OldVarSin*CosVar+OldVarCos*SinVar)*expon;

		NewActivity+= NewVarCos*TermPointer[offset];

		this->SetStateVariableAt(index, grade , NewVarCos, NewVarSin);

	}
	NewActivity*=this->factor;
	this->SetStateVariableAt(index, 0, NewActivity, NewExpon);
}





void SinState::ApplyPresynapticSpike(unsigned int index){
	this->incrementStateVariableAt(index, 1, 1.0f);
	for (unsigned int grade=2; grade<=this->exponent; grade+=2){
		this->incrementStateVariableAt(index, grade, 1.0f);
	}
}

void SinState::ApplyPostsynapticSpike(unsigned int index){
	return;
}
