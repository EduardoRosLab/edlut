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
	if (this->OutputConnections!=0){
		delete [] this->OutputConnections;
	}

	if (this->InputLearningConnectionsWithPostSynapticLearning!=0){
		delete [] this->InputLearningConnectionsWithPostSynapticLearning;
	}

	if (this->InputLearningConnectionsWithoutPostSynapticLearning!=0){
		delete [] this->InputLearningConnectionsWithoutPostSynapticLearning;
	}
}

void Neuron::InitNeuron(int NewIndex, int index_VectorNeuronState, NeuronModel * Type, bool Monitored, bool IsOutput){

	this->type = Type;

	this->state = Type->InitializeState();

	this->OutputConnections = 0;

	this->OutputConNumber = 0;

	this->InputLearningConnectionsWithPostSynapticLearning = 0;

	this->InputLearningConnectionsWithoutPostSynapticLearning = 0;

	this->InputConLearningNumberWithPostSynaptic = 0;

	this->InputConLearningNumberWithoutPostSynaptic = 0;

	this->index = NewIndex;

	this->index_VectorNeuronState=index_VectorNeuronState;

	this->monitored=Monitored;

	this->spikeCounter = 0; // For LSAM

	this->isOutput = IsOutput;
}

long int Neuron::GetIndex() const{
	return this->index;	
}

//VectorNeuronState * Neuron::GetVectorNeuronState() const{
//	return this->state;
//}
   		
unsigned int Neuron::GetInputNumberWithPostSynapticLearning() const{
	return this->InputConLearningNumberWithPostSynaptic;
}

unsigned int Neuron::GetInputNumberWithoutPostSynapticLearning() const{
	return this->InputConLearningNumberWithoutPostSynaptic;
}

   		
//unsigned int Neuron::GetOutputNumber() const{
//	return this->OutputConNumber;
//}

Interconnection * Neuron::GetInputConnectionWithPostSynapticLearningAt(unsigned int index) const{
	return *(this->InputLearningConnectionsWithPostSynapticLearning+index);
}

Interconnection * Neuron::GetInputConnectionWithoutPostSynapticLearningAt(unsigned int index) const{
	return *(this->InputLearningConnectionsWithoutPostSynapticLearning+index);
}
   		
void Neuron::SetInputConnectionsWithPostSynapticLearning(Interconnection ** Connections, unsigned int NumberOfConnections){
	if (this->InputLearningConnectionsWithPostSynapticLearning!=0){
		delete [] this->InputLearningConnectionsWithPostSynapticLearning;
	}
	
	this->InputLearningConnectionsWithPostSynapticLearning = Connections;

	this->InputConLearningNumberWithPostSynaptic = NumberOfConnections;
}

void Neuron::SetInputConnectionsWithoutPostSynapticLearning(Interconnection ** Connections, unsigned int NumberOfConnections){
	if (this->InputLearningConnectionsWithoutPostSynapticLearning!=0){
		delete [] this->InputLearningConnectionsWithoutPostSynapticLearning;
	}
	
	this->InputLearningConnectionsWithoutPostSynapticLearning = Connections;

	this->InputConLearningNumberWithoutPostSynaptic = NumberOfConnections;
}



   		
//Interconnection * Neuron::GetOutputConnectionAt(unsigned int index) const{
//	return *(this->OutputConnections+index);
//}
   		
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
   		
//bool Neuron::IsMonitored() const{
//	return this->monitored;	
//}

bool Neuron::IsOutput() const{
	return this->isOutput;	
}

//NeuronModel * Neuron::GetNeuronModel() const{
//	return this->type;
//}

ostream & Neuron::PrintInfo(ostream & out) {
	out << "- Neuron: " << this->index << endl;

	out << "\tType: " << this->type->GetModelID() << endl;

	out << "\tInput Connections With Learning: " << this->InputConLearningNumberWithPostSynaptic + this->InputConLearningNumberWithoutPostSynaptic << endl;

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

//long Neuron::GetIndex_VectorNeuronState(){
//	return index_VectorNeuronState;
//}