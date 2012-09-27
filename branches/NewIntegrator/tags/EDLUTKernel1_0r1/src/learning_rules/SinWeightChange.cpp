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

ConnectionState * SinWeightChange::GetInitialState(){
	return (ConnectionState *) new SinState(this->exponent,this->maxpos);
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
}

