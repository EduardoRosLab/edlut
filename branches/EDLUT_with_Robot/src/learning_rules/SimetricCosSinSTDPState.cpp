/***************************************************************************
 *                           SimetricCosSinSTDPState.cpp                   *
 *                           -------------------                           *
 * copyright            : (C) 2014 by Francisco Naveros                    *
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

#include "../../include/learning_rules/SimetricCosSinSTDPState.h"

#include "../../include/simulation/ExponentialTable.h"
#include "../../include/simulation/TrigonometricTable.h"


#include <cmath>
#include <stdio.h>
#include <float.h>

SimetricCosSinSTDPState::SimetricCosSinSTDPState(unsigned int NumSynapses, float NewMaxMinDistance, float NewCentralAmplitudeFactor, float NewLateralAmplitudeFactor): ConnectionState(NumSynapses, 6), 
		MaxMinDistance(NewMaxMinDistance), inv_MaxMinDistance(1.0f/NewMaxMinDistance), CentralExpFactor(1), LateralExpFactor(2){


	inv_AuxMaxMinDistanceForSin=(atan(3.141592/LateralExpFactor)*2/3.141592)/NewMaxMinDistance;
	CentralAmplitudeFactor=NewCentralAmplitudeFactor;
	LateralAmplitudeFactor=NewLateralAmplitudeFactor/(exp(-atan(3.141592/LateralExpFactor)*2*LateralExpFactor/3.141592)*pow(sin(atan(3.141592/LateralExpFactor)),2));
	
}

SimetricCosSinSTDPState::~SimetricCosSinSTDPState() {
}

float SimetricCosSinSTDPState::GetPresynapticActivity(unsigned int index){
	return CentralAmplitudeFactor*this->GetStateVariableAt(index, 0) + LateralAmplitudeFactor*this->GetStateVariableAt(index, 4);
}

float SimetricCosSinSTDPState::GetPostsynapticActivity(unsigned int index){
	return 0.0f;
}

unsigned int SimetricCosSinSTDPState::GetNumberOfPrintableValues(){
	return ConnectionState::GetNumberOfPrintableValues()+3;
}

double SimetricCosSinSTDPState::GetPrintableValuesAt(unsigned int position){
	if (position<ConnectionState::GetNumberOfPrintableValues()){
		return ConnectionState::GetStateVariableAt(0, position);
	} else if (position==ConnectionState::GetNumberOfPrintableValues()) {
		return this->MaxMinDistance;
	} else if (position==ConnectionState::GetNumberOfPrintableValues()+1) {
		return this->CentralAmplitudeFactor;
	} else if (position==ConnectionState::GetNumberOfPrintableValues()+2) {
		return this->LateralAmplitudeFactor;
	} else return -1;
}



void SimetricCosSinSTDPState::SetNewUpdateTime (unsigned int index, double NewTime, bool pre_post){
	float COSOldCos2= this->GetStateVariableAt(index, 0);
	float COSOldSin2= this->GetStateVariableAt(index, 1);
	float COSOldCosSin= this->GetStateVariableAt(index, 2);
	float SINOldCos2= this->GetStateVariableAt(index, 3);
	float SINOldSin2= this->GetStateVariableAt(index, 4);
	float SINOldCosSin= this->GetStateVariableAt(index, 5);

	float ElapsedTime=float(NewTime -  this->GetLastUpdateTime(index));
	float COSElapsedRelative = ElapsedTime*this->inv_MaxMinDistance;
	float COSexpon = ExponentialTable::GetResult(-COSElapsedRelative*CentralExpFactor);

	float SINElapsedRelative = ElapsedTime*this->inv_AuxMaxMinDistanceForSin;
	float SINexpon = ExponentialTable::GetResult(-SINElapsedRelative*LateralExpFactor);

COSElapsedRelative=ElapsedTime*this->inv_MaxMinDistance*1.5708f;
SINElapsedRelative=ElapsedTime*this->inv_AuxMaxMinDistanceForSin*1.5708f;



	int LUTindexCos=TrigonometricTable::CalculateOffsetPosition(COSElapsedRelative);
	LUTindexCos = TrigonometricTable::CalculateValidPosition(0,LUTindexCos);

	float COSauxSin2 = TrigonometricTable::GetElement(LUTindexCos);
	float COSauxCos2 = TrigonometricTable::GetElement(LUTindexCos+1);
	float COSauxCosSin=COSauxCos2*COSauxSin2;
	COSauxCos2*=COSauxCos2;
	COSauxSin2*=COSauxSin2;



	int LUTindexSin=TrigonometricTable::CalculateOffsetPosition(SINElapsedRelative);
	LUTindexSin = TrigonometricTable::CalculateValidPosition(0,LUTindexSin);

	float SINauxSin2 = TrigonometricTable::GetElement(LUTindexSin);
	float SINauxCos2 = TrigonometricTable::GetElement(LUTindexSin+1);
	float SINauxCosSin=SINauxCos2*SINauxSin2;
	SINauxCos2*=SINauxCos2;
	SINauxSin2*=SINauxSin2;

	
	float COSNewCos2 = COSexpon*(COSOldCos2 * COSauxCos2 + COSOldSin2*COSauxSin2 - 2*COSOldCosSin*COSauxCosSin);
	float COSNewSin2 = COSexpon*(COSOldSin2 * COSauxCos2 + COSOldCos2*COSauxSin2 + 2*COSOldCosSin*COSauxCosSin);
	float COSNewCosSin = COSexpon*(COSOldCosSin *(COSauxCos2-COSauxSin2) + (COSOldCos2-COSOldSin2)*COSauxCosSin);

	//REVISAR
	float SINNewCos2 = SINexpon*(SINOldCos2 * SINauxCos2 + SINOldSin2*SINauxSin2 - 2*SINOldCosSin*SINauxCosSin);
	float SINNewSin2 = SINexpon*(SINOldSin2 * SINauxCos2 + SINOldCos2*SINauxSin2 + 2*SINOldCosSin*SINauxCosSin);
	float SINNewCosSin = SINexpon*(SINOldCosSin *(SINauxCos2-SINauxSin2) + (SINOldCos2-SINOldSin2)*SINauxCosSin);

	this->SetStateVariableAt(index, 0, COSNewCos2, COSNewSin2, COSNewCosSin);
	this->SetStateVariableAt(index, 3, SINNewCos2, SINNewSin2, SINNewCosSin);


	this->SetLastUpdateTime(index, NewTime);
}


void SimetricCosSinSTDPState::ApplyPresynapticSpike(unsigned int index){
	this->incrementStateVaraibleAt(index, 0, 1.0f);
	this->incrementStateVaraibleAt(index, 3, 1.0f);
}

void SimetricCosSinSTDPState::ApplyPostsynapticSpike(unsigned int index){
	return;
}

