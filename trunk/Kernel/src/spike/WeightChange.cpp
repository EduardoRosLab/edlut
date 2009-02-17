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
 
#include "./include/WeightChange.h"

#include "./include/Interconnection.h"
#include "./include/Neuron.h"

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
   		
float WeightChange::GetLparAt(int index) const{
	return this->lpar[index];
}
   		
void WeightChange::SetLparAt(int index, float NewLpar){
	this->lpar[index] = NewLpar;
}
   		
float WeightChange::GetCparAt(int index) const{
	return this->cpar[index];
}
   		
void WeightChange::SetCparAt(int index, float NewCpar){
	this->cpar[index] = NewCpar;
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