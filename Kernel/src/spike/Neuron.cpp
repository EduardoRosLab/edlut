/***************************************************************************
 *                           Neuron.cpp                                    *
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

#include "../../include/spike/Neuron.h"

#include "../../include/spike/InternalSpike.h"
#include "../../include/spike/PropagatedSpike.h"
#include "../../include/spike/Interconnection.h"

#include "../../include/neuron_model/NeuronModel.h"
#include "../../include/neuron_model/VectorNeuronState.h"

#include "../../include/simulation/EventQueue.h"

#include <string>

using namespace std;

Neuron::Neuron(){
}

Neuron::Neuron(int NewIndex, NeuronModel * Type, bool Monitored, bool IsOutput){
	InitNeuron(NewIndex,-1,Type,Monitored,IsOutput);
}

Neuron::~Neuron(){
	//state is deleted en Neuron Model.
	if (this->InputLearningConnections!=0){
		delete [] this->InputLearningConnections;
	}

	if (this->OutputConnections!=0){
		delete [] this->OutputConnections;
	}
}

void Neuron::InitNeuron(int NewIndex, int index_VectorNeuronState, NeuronModel * Type, bool Monitored, bool IsOutput){

	this->type = Type;

	this->state = Type->InitializeState();

	this->OutputConnections = 0;

	this->OutputConNumber = 0;

	this->InputLearningConnections = 0;

	this->InputConLearningNumber = 0;

	this->index = NewIndex;

	this->index_VectorNeuronState=index_VectorNeuronState;

	this->monitored=Monitored;

	this->spikeCounter = 0; // For LSAM

	this->isOutput = IsOutput;
}

long int Neuron::GetIndex() const{
	return this->index;	
}

VectorNeuronState * Neuron::GetVectorNeuronState() const{
	return this->state;
}
   		
unsigned int Neuron::GetInputNumberWithLearning() const{
	return this->InputConLearningNumber;
}
   		
unsigned int Neuron::GetOutputNumber() const{
	return this->OutputConNumber;
}

Interconnection * Neuron::GetInputConnectionWithLearningAt(unsigned int index) const{
	return *(this->InputLearningConnections+index);
}
   		
void Neuron::SetInputConnectionsWithLearning(Interconnection ** Connections, unsigned int NumberOfConnections){
	if (this->InputLearningConnections!=0){
		delete [] this->InputLearningConnections;
	}
	
	this->InputLearningConnections = Connections;

	this->InputConLearningNumber = NumberOfConnections;
}

bool Neuron::IsInputConnected() const{
	return this->InputConLearningNumber!=0;
}
   		
Interconnection * Neuron::GetOutputConnectionAt(unsigned int index) const{
	return *(this->OutputConnections+index);
}
   		
void Neuron::SetOutputConnections(Interconnection ** Connections, unsigned int NumberOfConnections){
	if (this->OutputConnections!=0){
		delete [] this->OutputConnections;
	}
	
	this->OutputConnections = Connections;

	this->OutputConNumber = NumberOfConnections;
}

bool Neuron::IsOutputConnected() const{
	return this->OutputConNumber!=0;
}
   		
bool Neuron::IsMonitored() const{
	return this->monitored;	
}

bool Neuron::IsOutput() const{
	return this->isOutput;	
}

NeuronModel * Neuron::GetNeuronModel() const{
	return this->type;
}

ostream & Neuron::PrintInfo(ostream & out) {
	out << "- Neuron: " << this->index << endl;

	out << "\tType: " << this->type->GetModelID() << endl;

	out << "\tInput Connections With Learning: " << this->InputConLearningNumber << endl;

	out << "\tOutput Connections: " << this->OutputConNumber << endl;

   	if (this->monitored) out << "\tMonitored" << endl;
   	else out << "\tNon-monitored" << endl;

   	if (this->isOutput) out << "\tOutput" << endl;
   	else out << "\tNon-output" << endl;

	return out;
}

/* For LSAM */
void Neuron::SetSpikeCounter(long n) {
  this->spikeCounter = n;
}

/* For LSAM */
long Neuron::GetSpikeCounter() {
  return this->spikeCounter;
}

void Neuron::SetIndex_VectorNeuronState(long int index){
	index_VectorNeuronState=index;
}

long int Neuron::GetIndex_VectorNeuronState(){
	return index_VectorNeuronState;
}