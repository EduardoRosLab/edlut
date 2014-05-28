/***************************************************************************
 *                           SinWeightChange.cpp                           *
 *                           -------------------                           *
 * copyright            : (C) 2009 by Jesus Garrido and Richard Carrillo   *
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


#include "../../include/learning_rules/SinWeightChange.h"

#include "../../include/learning_rules/SinState.h"

#include "../../include/spike/Interconnection.h"

SinWeightChange::SinWeightChange():exponent(0){
}

SinWeightChange::~SinWeightChange(){
}


void SinWeightChange::InitializeConnectionState(unsigned int NumberOfSynapses){
	this->State=(ConnectionState *) new SinState(NumberOfSynapses, this->exponent,this->maxpos);
}

int SinWeightChange::GetNumberOfVar() const{
	return this->exponent+2;
}

int SinWeightChange::GetExponent() const{
	return this->exponent;
}

void SinWeightChange::LoadLearningRule(FILE * fh, long & Currentline) throw (EDLUTFileException){
	AdditiveKernelChange::LoadLearningRule(fh,Currentline);

	if(!(fscanf(fh,"%i",&this->exponent)==1)){
		throw EDLUTFileException(4,28,23,1,Currentline);
	}
	
	//exponent must be multiple of 2.
	if(exponent%2 == 1){
		exponent=(exponent/2)*2;
		cerr << "Warning: exponent in SinAdditiveKernel must be multiple of 2. It has been rounded to "<< exponent << endl;


	}


}

