/***************************************************************************
 *                           Interconnection.cpp                           *
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

#include "./include/Interconnection.h"

#include <math.h>
#include "./include/WeightChange.h"
#include "./include/Neuron.h"

Interconnection::Interconnection(): source(0), target(0), index(0), delay(0), type(0), weight(0), maxweight(0), wchange(0), lastspiketime(0){
	
}

Interconnection::Interconnection(int NewIndex, Neuron * NewSource, Neuron * NewTarget, float NewDelay, int NewType, float NewWeight, float NewMaxWeight, WeightChange* NewWeightChange, float NewActivity [3], float NewLastSpikeTime):
	source(NewSource), target(NewTarget), index(NewIndex), delay(NewDelay), type(NewType), weight(NewWeight), maxweight(NewMaxWeight), wchange(NewWeightChange), lastspiketime(NewLastSpikeTime) {
	this->activity[0] = NewActivity[0];
	this->activity[1] = NewActivity[1];
	this->activity[2] = NewActivity[2];		
}

long int Interconnection::GetIndex() const{
	return this->index;
}
		
void Interconnection::SetIndex(long int NewIndex){
	this->index = NewIndex;
}
		
Neuron * Interconnection::GetSource() const{
	return this->source;	
}
		
void Interconnection::SetSource(Neuron * NewSource){
	this->source = NewSource;	
}
		
Neuron * Interconnection::GetTarget() const{
	return this->target;
}
		
void Interconnection::SetTarget(Neuron * NewTarget){
	this->target = NewTarget;
}
		
double Interconnection::GetDelay() const{
	return this->delay;
}
		
void Interconnection::SetDelay(float NewDelay){
	this->delay = NewDelay;
}
		
int Interconnection::GetType() const{
	return this->type;
}
		
void Interconnection::SetType(int NewType){
	this->type = NewType;
}
		
float Interconnection::GetWeight() const{
	return this->weight;
}
		
void Interconnection::SetWeight(float NewWeight){
	this->weight = NewWeight;
}
		
float Interconnection::GetMaxWeight() const{
	return this->maxweight;
}
		
void Interconnection::SetMaxWeight(float NewMaxWeight){
	this->maxweight = NewMaxWeight;
}
		
WeightChange * Interconnection::GetWeightChange() const{
	return this->wchange;
}
		
void Interconnection::SetWeightChange(WeightChange * NewWeightChange){
	this->wchange = NewWeightChange;
}

void Interconnection::ClearActivity(){
	for (int i=0; i<3; ++i){
		this->activity[i] = 0.0F;
	}
}
		
float Interconnection::GetActivityAt(int index) const{
	return this->activity[index];
}
		
void Interconnection::SetActivityAt(int index, float NewActivity){
	this->activity[index] = NewActivity;
}
		
double Interconnection::GetLastSpikeTime() const{
	return this->lastspiketime;
}	
		
void Interconnection::SetLastSpikeTime(double NewTime){
	this->lastspiketime = NewTime;
}

void Interconnection::ChangeWeights(double stime){
	this->wchange->ApplyWeightChange(this,stime);
}
