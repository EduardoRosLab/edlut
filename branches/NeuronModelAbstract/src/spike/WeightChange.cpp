/***************************************************************************
 *                           WeightChange.cpp                              *
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
 
#include "../../include/spike/WeightChange.h"

#include "../../include/spike/Interconnection.h"
#include "../../include/spike/Neuron.h"

#include <math.h>

float WeightChange::GetMaxPos() const{
	return this->maxpos;
}
   		
void WeightChange::SetMaxPos(float NewMaxPos){
	this->maxpos = NewMaxPos;
}
   		
int WeightChange::GetNumExps() const{
	return this->numexps;
}
   		
void WeightChange::SetNumExps(int NewNumExps){
	this->numexps = NewNumExps;
}
   		
int WeightChange::GetTrigger() const{
	return this->trigger;
}
   		
void WeightChange::SetTrigger(int NewTrigger){
	this->trigger = NewTrigger;
}
   		
float WeightChange::GetA1Pre() const{
	return this->a1pre;
}
   		
void WeightChange::SetA1Pre(float NewA1Pre){
	this->a1pre = NewA1Pre;
}
   		
float WeightChange::GetA2PrePre() const{
	return this->a2prepre;
}
   		
void WeightChange::SetA2PrePre(float NewA2PrePre){
	this->a2prepre = NewA2PrePre;
}

ostream & WeightChange::PrintInfo(ostream & out) {
	out << "- Weight Change: " << endl;

	out << "\tMaximum Value in " << this->GetMaxPos() << "s" << endl;

	out << "\tNumber of state variables: " << this->GetNumberOfVar() << endl;

   	out << "\tNumber of registers: " << this->GetNumExps() << endl;

   	out << "\tA1 parameter value: " << this->GetA1Pre() << endl;

   	out << "\tA2 parameter value: " << this->GetA2PrePre() << endl;

   	if (this->GetTrigger()) out << "\tTrigger" << endl;
   	else out << "\tNon-trigger" << endl;
}

