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

#include "../../include/spike/Interconnection.h"

#include <cmath>

#include "../../include/learning_rules/ActivityRegister.h"
#include "../../include/spike/Neuron.h"

#include "../../include/learning_rules/LearningRule.h"
#include "../../include/learning_rules/ConnectionState.h"

Interconnection::Interconnection(): source(0), target(0), index(0), delay(0), type(0), weight(0), maxweight(0), wchange(0), LearningRuleIndex(0){
	
}

Interconnection::Interconnection(int NewIndex, Neuron * NewSource, Neuron * NewTarget, float NewDelay, int NewType, float NewWeight, float NewMaxWeight, LearningRule* NewWeightChange, unsigned int NewLearningRuleIndex):
	source(NewSource), target(NewTarget), index(NewIndex), delay(NewDelay), type(NewType), weight(NewWeight), maxweight(NewMaxWeight), wchange(NewWeightChange),LearningRuleIndex(NewLearningRuleIndex) {
}

Interconnection::~Interconnection(){

}

long int Interconnection::GetIndex() const{
	return this->index;
}
		
void Interconnection::SetIndex(long int NewIndex){
	this->index = NewIndex;
}
		
//Neuron * Interconnection::GetSource() const{
//	return this->source;	
//}
		
void Interconnection::SetSource(Neuron * NewSource){
	this->source = NewSource;	
}
		
//Neuron * Interconnection::GetTarget() const{
//	return this->target;
//}
		
void Interconnection::SetTarget(Neuron * NewTarget){
	this->target = NewTarget;
}
		
//double Interconnection::GetDelay() const{
//	return this->delay;
//}
		
void Interconnection::SetDelay(float NewDelay){
	this->delay = NewDelay;
}
		
//int Interconnection::GetType() const{
//	return this->type;
//}
		
void Interconnection::SetType(int NewType){
	this->type = NewType;
}
		
//float Interconnection::GetWeight() const{
//	return this->weight;
//}
		
//void Interconnection::SetWeight(float NewWeight){
//	this->weight = NewWeight;
//}
		
//float Interconnection::GetMaxWeight() const{
//	return this->maxweight;
//}
		
void Interconnection::SetMaxWeight(float NewMaxWeight){
	this->maxweight = NewMaxWeight;
}
		
//LearningRule * Interconnection::GetWeightChange() const{
//	return this->wchange;
//}
		
void Interconnection::SetWeightChange(LearningRule * NewWeightChange){
	this->wchange=NewWeightChange;
}


void Interconnection::SetLearningRuleIndex(int NewIndex){
	this->LearningRuleIndex=NewIndex;
}

//int Interconnection::GetLearningRuleIndex() const{
//	return this->LearningRuleIndex;
//}

//void Interconnection::IncrementWeight(float Increment){
//	this->weight += Increment;
//	if(this->weight > this->GetMaxWeight()){
//		this->weight = this->GetMaxWeight();
//	}else if(this->weight < 0.0f){
//		this->weight = 0.0f;
//	}
//}


ostream & Interconnection::PrintInfo(ostream & out) {
	out << "- Interconnection: " << this->index << endl;

	out << "\tSource: " << this->source->GetIndex() << endl;

	out << "\tTarget: " << this->target->GetIndex() << endl;

   	out << "\tDelay: " << this->GetDelay() << "s" << endl;

   	out << "\tConnection Type: " << this->GetType() << endl;

   	out << "\tCurrent Weight: " << this->GetWeight() << endl;

   	out << "\tMaximum Weight: " << this->GetMaxWeight() << endl;

   	out << "\tWeight Change: ";

   	if (this->GetWeightChange()!=0) this->GetWeightChange()->PrintInfo(out);
   	else out << "None" << endl;

   	return out;
}
