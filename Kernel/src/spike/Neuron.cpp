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

#include "../../include/learning_rules/WithTriggerSynaptic.h"

#include "../../include/openmp/openmp.h"

#include "../../include/spike/NeuronPropagationDelayStructure.h"
#include "../../include/spike/NeuronModelPropagationDelayStructure.h"

#include <string>

using namespace std;

Neuron::Neuron():N_TriggerConnectionPerRule(0), TriggerConnectionPerRule(0){
}

Neuron::~Neuron(){
	//state is deleted en Neuron Model.
	if (this->OutputConnections!=0){
		for(int i=0; i<NumberOfOpenMPQueues; i++){
			delete [] OutputConnections[i];
		}
		delete [] this->OutputConnections;
	}

	if (this->InputLearningConnectionsWithPostSynapticLearning!=0){
		for (unsigned int wcindex=0; wcindex<this->NumberOfLearningRules; ++wcindex){
			if(this->InputConLearningNumberWithPostSynaptic[wcindex]>0){
				delete [] this->InputLearningConnectionsWithPostSynapticLearning[wcindex];
			}
		}
		delete [] this->InputLearningConnectionsWithPostSynapticLearning;
	}

	if (this->InputConLearningNumberWithPostSynaptic != 0){
		delete [] this->InputConLearningNumberWithPostSynaptic;
	}

	if (this->InputLearningConnectionsWithTriggerSynapticLearning!=0){
		for (unsigned int wcindex=0; wcindex<this->NumberOfLearningRules; ++wcindex){
			if(this->InputConLearningNumberWithTriggerSynaptic[wcindex]>0){
				delete [] this->InputLearningConnectionsWithTriggerSynapticLearning[wcindex];
			}
		}
		delete [] this->InputLearningConnectionsWithTriggerSynapticLearning;
	}

	if (this->InputConLearningNumberWithTriggerSynaptic != 0){
		delete [] this->InputConLearningNumberWithTriggerSynaptic;
	}

	if (this->InputLearningConnectionsWithPostAndTriggerSynapticLearning!=0){
		for (unsigned int wcindex=0; wcindex<this->NumberOfLearningRules; ++wcindex){
			if(this->InputConLearningNumberWithPostAndTriggerSynaptic[wcindex]>0){
				delete [] this->InputLearningConnectionsWithPostAndTriggerSynapticLearning[wcindex];
			}
		}
		delete [] this->InputLearningConnectionsWithPostAndTriggerSynapticLearning;
	}

	if (this->InputConLearningNumberWithPostAndTriggerSynaptic != 0){
		delete [] this->InputConLearningNumberWithPostAndTriggerSynaptic;
	}

	if(this->N_TriggerConnectionPerRule != 0){
		for (unsigned int wcindex=0; wcindex<this->NumberOfLearningRules; ++wcindex){
			if(this->N_TriggerConnectionPerRule[wcindex] > 0){
				delete [] this->TriggerConnectionPerRule[wcindex];
			}
		}
		delete [] this->N_TriggerConnectionPerRule;
		if (this->TriggerConnectionPerRule!=0){
			delete [] this->TriggerConnectionPerRule;
		}
	}

	delete PropagationStructure;

	if (IndexInputLearningConnections[0]){
		delete IndexInputLearningConnections[0];
	}
	if (IndexInputLearningConnections[1]){
		delete IndexInputLearningConnections[1];
	}
	if (IndexInputLearningConnections[2]){
		delete IndexInputLearningConnections[2];
	}
	delete IndexInputLearningConnections;
}

void Neuron::InitNeuron(int NewIndex, int index_VectorNeuronState, NeuronModel * Type, bool Monitored, bool IsOutput, int blockIndex){

	this->type = Type;

	this->state = type->InitializeState();

	this->OutputConnections = 0;

	this->OutputConNumber = 0;

	this->InputLearningConnectionsWithPostSynapticLearning = 0;

	this->InputLearningConnectionsWithTriggerSynapticLearning = 0;

	this->InputLearningConnectionsWithPostAndTriggerSynapticLearning = 0;

	this->InputConLearningNumberWithPostSynaptic = 0;

	this->InputConLearningNumberWithTriggerSynaptic = 0;

	this->InputConLearningNumberWithPostAndTriggerSynaptic = 0;

	this->NumberOfLearningRules = 0;

	this->index = NewIndex;

	this->index_VectorNeuronState = index_VectorNeuronState;

	this->monitored=Monitored;

	this->spikeCounter = 0; // For LSAM

	this->isOutput = IsOutput;

	this->N_TriggerConnectionPerRule = 0;

	this->TriggerConnectionPerRule = 0;

	this->OpenMP_queue_index=blockIndex;

	this->IndexInputLearningConnections = (int ***) new int**[3]();
}

//long int Neuron::GetIndex() const{
//	return this->index;
//}

//VectorNeuronState * Neuron::GetVectorNeuronState() const{
//	return this->state;
//}

unsigned int Neuron::GetInputNumberWithPostSynapticLearning(unsigned int weight_change_index) {
	return this->InputConLearningNumberWithPostSynaptic[weight_change_index];
}

unsigned int Neuron::GetInputNumberWithTriggerSynapticLearning(unsigned int weight_change_index) {
	return this->InputConLearningNumberWithTriggerSynaptic[weight_change_index];
}

unsigned int Neuron::GetInputNumberWithPostAndTriggerSynapticLearning(unsigned int weight_change_index) {
	return this->InputConLearningNumberWithPostAndTriggerSynaptic[weight_change_index];
}

//unsigned int Neuron::GetOutputNumber(int index) const{
//	return this->OutputConNumber[index];
//}

Interconnection * Neuron::GetInputConnectionWithPostSynapticLearningAt(unsigned int learning_rule_id, unsigned int index) const{
	return this->InputLearningConnectionsWithPostSynapticLearning[learning_rule_id][index];
}

Interconnection * Neuron::GetInputConnectionWithTriggerSynapticLearningAt(unsigned int learning_rule_id, unsigned int index) const{
	return this->InputLearningConnectionsWithTriggerSynapticLearning[learning_rule_id][index];
}

Interconnection * Neuron::GetInputConnectionWithPostAndTriggerSynapticLearningAt(unsigned int learning_rule_id, unsigned int index) const{
	return this->InputLearningConnectionsWithPostAndTriggerSynapticLearning[learning_rule_id][index];
}

void Neuron::SetInputConnectionsWithPostSynapticLearning(Interconnection *** ConnectionsPerRule, unsigned int * NumberOfConnectionsPerRule, unsigned int NumberOfLearningRules){
	if (this->InputLearningConnectionsWithPostSynapticLearning!=0){
		for (unsigned int wcindex=0; wcindex<this->NumberOfLearningRules; ++wcindex){
			if(this->InputConLearningNumberWithPostSynaptic[wcindex]>0){
				delete [] this->InputLearningConnectionsWithPostSynapticLearning[wcindex];
			}
		}
		delete [] this->InputLearningConnectionsWithPostSynapticLearning;
	}

	if (this->InputConLearningNumberWithPostSynaptic != 0){
		delete [] this->InputConLearningNumberWithPostSynaptic;
	}

	this->InputLearningConnectionsWithPostSynapticLearning = ConnectionsPerRule;

	this->InputConLearningNumberWithPostSynaptic = NumberOfConnectionsPerRule;

	this->NumberOfLearningRules = NumberOfLearningRules;
}

void Neuron::SetInputConnectionsWithTriggerSynapticLearning(Interconnection *** ConnectionsPerRule, unsigned int * NumberOfConnectionsPerRule, unsigned int NumberOfLearningRules){
	if (this->InputLearningConnectionsWithTriggerSynapticLearning!=0){
		for (unsigned int wcindex=0; wcindex<this->NumberOfLearningRules; ++wcindex){
			if(this->InputConLearningNumberWithTriggerSynaptic[wcindex]>0){
				delete [] this->InputLearningConnectionsWithTriggerSynapticLearning[wcindex];
			}
		}
		delete [] this->InputLearningConnectionsWithTriggerSynapticLearning;
	}

	if (this->InputConLearningNumberWithTriggerSynaptic != 0){
		delete [] this->InputConLearningNumberWithTriggerSynaptic;
	}

	this->InputLearningConnectionsWithTriggerSynapticLearning = ConnectionsPerRule;

	this->InputConLearningNumberWithTriggerSynaptic = NumberOfConnectionsPerRule;

	this->NumberOfLearningRules = NumberOfLearningRules;


	this->N_TriggerConnectionPerRule = new int[NumberOfLearningRules];
	this->TriggerConnectionPerRule = (Interconnection***)new Interconnection**[NumberOfLearningRules];

	for (unsigned int wcindex=0; wcindex<this->NumberOfLearningRules; ++wcindex){
		this->N_TriggerConnectionPerRule[wcindex]=0;
		this->TriggerConnectionPerRule[wcindex]=0;
		if(NumberOfConnectionsPerRule[wcindex]>0){
			Interconnection ** aux= (Interconnection **) new Interconnection * [NumberOfConnectionsPerRule[wcindex]];
			for(int i=0; i<NumberOfConnectionsPerRule[wcindex]; i++){
				if(ConnectionsPerRule[wcindex][i]->GetTriggerConnection()){
					aux[this->N_TriggerConnectionPerRule[wcindex]]=ConnectionsPerRule[wcindex][i];
					this->N_TriggerConnectionPerRule[wcindex]++;
				}
			}

			if(this->N_TriggerConnectionPerRule[wcindex]>0){
				this->TriggerConnectionPerRule[wcindex]=(Interconnection**)new Interconnection*[this->N_TriggerConnectionPerRule[wcindex]];
				for(int i=0; i<this->N_TriggerConnectionPerRule[wcindex]; i++){
					this->TriggerConnectionPerRule[wcindex][i]=aux[i];
				}
			}
			delete [] aux;
		}
	}
}

void Neuron::SetInputConnectionsWithPostAndTriggerSynapticLearning(Interconnection *** ConnectionsPerRule, unsigned int * NumberOfConnectionsPerRule, unsigned int NumberOfLearningRules){
	if (this->InputLearningConnectionsWithPostAndTriggerSynapticLearning!=0){
		for (unsigned int wcindex=0; wcindex<this->NumberOfLearningRules; ++wcindex){
			if(this->InputConLearningNumberWithPostAndTriggerSynaptic[wcindex]>0){
				delete [] this->InputLearningConnectionsWithPostAndTriggerSynapticLearning[wcindex];
			}
		}
		delete [] this->InputLearningConnectionsWithPostAndTriggerSynapticLearning;
	}

	if (this->InputConLearningNumberWithPostAndTriggerSynaptic != 0){
		delete [] this->InputConLearningNumberWithPostAndTriggerSynaptic;
	}

	this->InputLearningConnectionsWithPostAndTriggerSynapticLearning = ConnectionsPerRule;

	this->InputConLearningNumberWithPostAndTriggerSynaptic = NumberOfConnectionsPerRule;

	this->NumberOfLearningRules = NumberOfLearningRules;

//THIS TWO VECTORS HAVE BEEN INITIALIZED IN THE PREVIOUS FUNCTION.
//	this->N_TriggerConnectionPerRule = new int[NumberOfLearningRules];
//	this->TriggerConnectionPerRule = (Interconnection***)new Interconnection**[NumberOfLearningRules];

	for (unsigned int wcindex=0; wcindex<this->NumberOfLearningRules; ++wcindex){
//		this->N_TriggerConnectionPerRule[wcindex]=0;
//		this->TriggerConnectionPerRule[wcindex]=0;
		if(NumberOfConnectionsPerRule[wcindex]>0){
			Interconnection ** aux= (Interconnection **) new Interconnection * [NumberOfConnectionsPerRule[wcindex]];
			for(int i=0; i<NumberOfConnectionsPerRule[wcindex]; i++){
				if(ConnectionsPerRule[wcindex][i]->GetTriggerConnection()){
					aux[this->N_TriggerConnectionPerRule[wcindex]]=ConnectionsPerRule[wcindex][i];
					this->N_TriggerConnectionPerRule[wcindex]++;
				}
			}

			if(this->N_TriggerConnectionPerRule[wcindex]>0){
				this->TriggerConnectionPerRule[wcindex]=(Interconnection**)new Interconnection*[this->N_TriggerConnectionPerRule[wcindex]];
				for(int i=0; i<this->N_TriggerConnectionPerRule[wcindex]; i++){
					this->TriggerConnectionPerRule[wcindex][i]=aux[i];
				}
			}
			delete [] aux;
		}
	}
}


//Interconnection * Neuron::GetOutputConnectionAt(unsigned int index1, unsigned int index2) const{
//	return OutputConnections[index1][index2];
//}

void Neuron::SetOutputConnections(Interconnection *** Connections, unsigned long * NumberOfConnections){
	if (this->OutputConnections!=0){
		delete [] this->OutputConnections;
	}

	this->OutputConnections = Connections;

	this->OutputConNumber = NumberOfConnections;
}

bool Neuron::IsOutputConnected(int index) const{
	return this->OutputConNumber[index]!=0;
}

//int Neuron::IsMonitored() const{
//	return this->monitored;
//}

//int Neuron::IsOutput() const{
//	return this->isOutput;
//}

//NeuronModel * Neuron::GetNeuronModel() const{
//	return this->type;
//}

ostream & Neuron::PrintInfo(ostream & out) {
	out << "- Neuron: " << this->index << endl;

	out << "\tType: " << boost::any_cast<std::string>(this->type->GetParameters()["name"]) << endl;

	int TotalInputConLearningNumber = 0;
	for (int i = 0; i < this->NumberOfLearningRules; i++) {
		TotalInputConLearningNumber += this->InputConLearningNumberWithPostSynaptic[i] + this->InputConLearningNumberWithTriggerSynaptic[i] + this->InputConLearningNumberWithPostAndTriggerSynaptic[i];
	}

	out << "\tInput Connections With Learning: " << TotalInputConLearningNumber << endl;

	int Total_OutputConNumber = this->OutputConNumber[0];
	for (int i = 1; i < NumberOfOpenMPQueues; i++){
		Total_OutputConNumber += this->OutputConNumber[i];
	}

	out << "\tOutput Connections: " << Total_OutputConNumber << endl;

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

void Neuron::set_OpenMP_queue_index(int index){
	this->OpenMP_queue_index=index;
}

void Neuron::CalculateOutputDelayStructure(){
	PropagationStructure=new NeuronPropagationDelayStructure(this);

	for(int i=0; i<NumberOfOpenMPQueues; i++){
		for(int j=0;j<PropagationStructure->NDifferentDelays[i];j++){
			this->GetNeuronModel()->GetNeuronModelPropagationDelayStructure()->IncludeNewDelay(i, PropagationStructure->SynapseDelay[i][j]);
		}
	}
}


void Neuron::CalculateOutputDelayIndex(){
	PropagationStructure->CalculateOutputDelayIndex(this->GetNeuronModel()->GetNeuronModelPropagationDelayStructure());
}


void Neuron::initializeLearningRuleIndex(){
	//Calculate learning rule index for learning rules with postsynaptic learning
	if (this->NumberOfLearningRules){
		this->IndexInputLearningConnections[0] = (int **) new int * [this->NumberOfLearningRules];
		this->IndexInputLearningConnections[1] = (int **) new int * [this->NumberOfLearningRules];
		this->IndexInputLearningConnections[2] = (int **) new int * [this->NumberOfLearningRules];
	}
	for (unsigned int wcindex=0; wcindex<this->NumberOfLearningRules; ++wcindex){
		if (this->GetInputNumberWithPostSynapticLearning(wcindex)){
			this->IndexInputLearningConnections[0][wcindex] = new int[this->GetInputNumberWithPostSynapticLearning(wcindex)]();
			for (int i = 0; i < this->GetInputNumberWithPostSynapticLearning(wcindex); i++){
				if (this->GetInputConnectionWithPostSynapticLearningAt(wcindex,i)->GetTriggerConnection()){
					this->IndexInputLearningConnections[0][wcindex][i] = -1;
				}
				else{
					this->IndexInputLearningConnections[0][wcindex][i] = this->GetInputConnectionWithPostSynapticLearningAt(wcindex,i)->GetLearningRuleIndex_withPost();
				}
				this->GetInputConnectionWithPostSynapticLearningAt(wcindex,i)->LearningRuleIndex_withPost_insideTargetNeuron = i;
			}
		}


		//Calculate learning rule index for learning rules without postsynaptic learning (trigger learning)
		if (this->GetInputNumberWithTriggerSynapticLearning(wcindex)){
			this->IndexInputLearningConnections[1][wcindex] = new int[this->GetInputNumberWithTriggerSynapticLearning(wcindex)]();
			for (int i = 0; i < this->GetInputNumberWithTriggerSynapticLearning(wcindex); i++){
				if (this->GetInputConnectionWithTriggerSynapticLearningAt(wcindex, i)->GetTriggerConnection()){
					IndexInputLearningConnections[1][wcindex][i] = -1;
				}
				else{
					IndexInputLearningConnections[1][wcindex][i] = this->GetInputConnectionWithTriggerSynapticLearningAt(wcindex,i)->GetLearningRuleIndex_withTrigger();
				}
				this->GetInputConnectionWithTriggerSynapticLearningAt(wcindex,i)->LearningRuleIndex_withTrigger_insideTargetNeuron = i;
			}
		}


		if (this->GetInputNumberWithPostAndTriggerSynapticLearning(wcindex)){
			this->IndexInputLearningConnections[2][wcindex] = new int[this->GetInputNumberWithPostAndTriggerSynapticLearning(wcindex)]();
			for (int i = 0; i < this->GetInputNumberWithPostAndTriggerSynapticLearning(wcindex); i++){
				if (this->GetInputConnectionWithPostAndTriggerSynapticLearningAt(wcindex,i)->GetTriggerConnection()){
					this->IndexInputLearningConnections[2][wcindex][i] = -1;
				}
				else{
					this->IndexInputLearningConnections[2][wcindex][i] = this->GetInputConnectionWithPostAndTriggerSynapticLearningAt(wcindex,i)->GetLearningRuleIndex_withPostAndTrigger();
				}
				this->GetInputConnectionWithPostAndTriggerSynapticLearningAt(wcindex,i)->LearningRuleIndex_withPostAndTrigger_insideTargetNeuron = i;
			}
		}
	}
}
