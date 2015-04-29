/***************************************************************************
 *                           ExpWeightChange.cpp                           *
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


#include "../../include/learning_rules/ExpWeightChange.h"

#include "../../include/learning_rules/ExpState.h"

#include "../../include/spike/Interconnection.h"

ExpWeightChange::ExpWeightChange(int NewLearningRuleIndex):AdditiveKernelChange(NewLearningRuleIndex){
}

ExpWeightChange::~ExpWeightChange(){
}


void ExpWeightChange::InitializeConnectionState(unsigned int NumberOfSynapses){
	this->State=(ConnectionState *) new ExpState(NumberOfSynapses, this->maxpos);
}
