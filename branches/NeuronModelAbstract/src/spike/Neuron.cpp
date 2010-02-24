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
#include "../../include/neuron_model/NeuronState.h"

#include "../../include/simulation/EventQueue.h"

Neuron::Neuron(){
}

Neuron::Neuron(int NewIndex, NeuronModel * Type, bool Monitored, bool IsOutput){
	InitNeuron(NewIndex,Type,Monitored,IsOutput);
}

void Neuron::InitNeuron(int NewIndex, NeuronModel * Type, bool Monitored, bool IsOutput){

	this->type = Type;

	this->state = Type->InitializeState();

	this->index = NewIndex;

	this->monitored=Monitored;

	this->isOutput = IsOutput;
}

long int Neuron::GetIndex() const{
	return this->index;	
}

NeuronState * Neuron::GetNeuronState() const{
	return this->state;
}
   		
int Neuron::GetInputNumber() const{
	return InputConnections.size();	
}
   		
int Neuron::GetOutputNumber() const{
	return OutputConnections.size();
}

Interconnection * Neuron::GetInputConnectionAt(int index) const{
	return InputConnections[index];
}
   		
void Neuron::AddInputConnection(Interconnection * Connection){
	InputConnections.push_back(Connection);	
}

bool Neuron::IsInputConnected() const{
	return !InputConnections.empty();
}
   		
Interconnection * Neuron::GetOutputConnectionAt(int index) const{
	return OutputConnections[index];
}
   		
void Neuron::AddOutputConnection(Interconnection * Connection){
	OutputConnections.push_back(Connection);
}

bool Neuron::IsOutputConnected() const{
	return !OutputConnections.empty();
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

	out << "\tInput Connections: " << this->InputConnections.size() << endl;

   	out << "\tOutput Connections: " << this->OutputConnections.size() << endl;

   	if (this->monitored) out << "\tMonitored" << endl;
   	else out << "\tNon-monitored" << endl;

   	if (this->isOutput) out << "\tOutput" << endl;
   	else out << "\tNon-output" << endl;
}


