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

#include "../../include/spike/Neuron.h"
#include "../../include/neuron_model/NeuronModel.h"

#include "../../include/learning_rules/LearningRule.h"
#include "../../include/learning_rules/ConnectionState.h"

Interconnection::Interconnection() : source(0), target(0), index(0), delay(0), type(0), subindex_type(0), weight(0), maxweight(0), wchange_withPost(0), LearningRuleIndex_withPost(-1), wchange_withTrigger(0), LearningRuleIndex_withTrigger(-1), wchange_withPostAndTrigger(0), LearningRuleIndex_withPostAndTrigger(-1), TriggerConnection(false), LearningRuleIndex_withPost_insideTargetNeuron(-1), LearningRuleIndex_withTrigger_insideTargetNeuron(-1), LearningRuleIndex_withPostAndTrigger_insideTargetNeuron(-1), targetNeuronModel(0), targetNeuronModelIndex(1){

}

Interconnection::Interconnection(int NewIndex, Neuron * NewSource, Neuron * NewTarget, float NewDelay, int NewType, float NewWeight, float NewMaxWeight, LearningRule* NewWeightChange_withPost, unsigned int NewLearningRuleIndex_withPost, LearningRule* NewWeightChange_withTrigger, unsigned int NewLearningRuleIndex_withTrigger, LearningRule* NewWeightChange_withPostAndTrigger, unsigned int NewLearningRuleIndex_withPostAndTrigger):
source(NewSource), target(NewTarget), index(NewIndex), delay(NewDelay), type(NewType), subindex_type(0), weight(NewWeight), maxweight(NewMaxWeight), wchange_withPost(NewWeightChange_withPost), LearningRuleIndex_withPost(NewLearningRuleIndex_withPost), wchange_withTrigger(NewWeightChange_withTrigger), LearningRuleIndex_withTrigger(NewLearningRuleIndex_withTrigger), wchange_withPostAndTrigger(NewWeightChange_withPostAndTrigger), LearningRuleIndex_withPostAndTrigger(NewLearningRuleIndex_withPostAndTrigger), LearningRuleIndex_withPost_insideTargetNeuron(-1), LearningRuleIndex_withTrigger_insideTargetNeuron(-1), LearningRuleIndex_withPostAndTrigger_insideTargetNeuron(-1){
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

void Interconnection::SetDelay(double NewDelay){
	this->delay = NewDelay;
}

//int Interconnection::GetType() const{
//	return this->type;
//}

void Interconnection::SetType(int NewType){
	this->type = NewType;
}

//int Interconnection::GetSubindexType() const{
//	return this->subindex_type;
//}

void Interconnection::SetSubindexType(int index){
	this->subindex_type = index;
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

//LearningRule * Interconnection::GetWeightChange_withPost() const{
//	return this->wchange_withPost;
//}

void Interconnection::SetWeightChange_withPost(LearningRule * NewWeightChange_withPost){
	this->wchange_withPost=NewWeightChange_withPost;
}

//LearningRule * Interconnection::GetWeightChange_withTrigger() const{
//	return this->wchange_withTrigger;
//}

void Interconnection::SetWeightChange_withTrigger(LearningRule * NewWeightChange_withTrigger){
	this->wchange_withTrigger=NewWeightChange_withTrigger;
}

//LearningRule * Interconnection::GetWeightChange_withPostAndTrigger() const{
//	return this->wchange_withPostAndTrigger;
//}

void Interconnection::SetWeightChange_withPostAndTrigger(LearningRule * NewWeightChange_withPostAndTrigger){
	this->wchange_withPostAndTrigger=NewWeightChange_withPostAndTrigger;
}

void Interconnection::SetLearningRuleIndex_withPost(unsigned int NewIndex){
	this->LearningRuleIndex_withPost=NewIndex;
}

//int Interconnection::GetLearningRuleIndex_withPost() const{
//	return this->LearningRuleIndex_withPost;
//}

void Interconnection::SetLearningRuleIndex_withTrigger(unsigned int NewIndex){
	this->LearningRuleIndex_withTrigger=NewIndex;
}

//int Interconnection::GetLearningRuleIndex_withTrigger() const{
//	return this->LearningRuleIndex_withTrigger;
//}

void Interconnection::SetLearningRuleIndex_withPostAndTrigger(unsigned int NewIndex){
	this->LearningRuleIndex_withPostAndTrigger=NewIndex;
}

//int Interconnection::GetLearningRuleIndex_withPostAndTrigger() const{
//	return this->LearningRuleIndex_withPostAndTrigger;
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

   	if (this->GetWeightChange_withPost()!=0) this->GetWeightChange_withPost()->PrintInfo(out);
   	else out << "None learning rule with postsynaptic learning" << endl;

		if (this->GetWeightChange_withTrigger()!=0) this->GetWeightChange_withTrigger()->PrintInfo(out);
   	else out << "None learning rule with trigger learning" << endl;

		if (this->GetWeightChange_withPostAndTrigger()!=0) this->GetWeightChange_withPostAndTrigger()->PrintInfo(out);
		else out << "None learning rule without postsynaptic and trigger learning" << endl;

   	return out;
}

void Interconnection::SetTriggerConnection(){
	this->TriggerConnection=true;
}

bool Interconnection::GetTriggerConnection(){
	return this->TriggerConnection;
}
