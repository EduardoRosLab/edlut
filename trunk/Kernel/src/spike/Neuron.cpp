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

#include "../../include/spike/NeuronType.h"
#include "../../include/spike/InternalSpike.h"
#include "../../include/spike/PropagatedSpike.h"
#include "../../include/spike/Interconnection.h"

#include "../../include/simulation/EventQueue.h"

Neuron::Neuron(){
}

Neuron::Neuron(int NewIndex, NeuronType * Type, bool Monitored, bool IsOutput){
	InitNeuron(NewIndex,Type,Monitored,IsOutput);
}

void Neuron::InitNeuron(int NewIndex, NeuronType * Type, bool Monitored, bool IsOutput){
	int nv;
	//neutype=&Net.neutypes[type];
	for(nv=0;nv<Type->GetStateVarsNumber();nv++)
		statevars[nv+1]=Type->GetInitValueAt(nv);
   	statevars[0]=0.0F;
   	type=Type;
   	lastupdate=0.0F; // must be 0.0
   	predictionend=NOPREDICTION;
   	predictedspike=NOPREDICTION;
   	this->index = NewIndex;
   	//outconind=0;
  	//noutputs=0;
  	//firstincon=NULL; // initial value needed by find_in_connections
   	monitored=Monitored;
   	this->isOutput = IsOutput;
}

void Neuron::InitNeuronPrediction(EventQueue * eventQueue){
	predictedspike=FiringPrediction();
	if(predictedspike != NOPREDICTION){
		predictedspike+=lastupdate;
		InternalSpike * spike = new InternalSpike();
		spike->SetTime(predictedspike);
		spike->SetSource(this);
		eventQueue->InsertEvent(spike);
    }
}

void Neuron::NeuronUpdate(double ElapsedTime){
	int ivar,orderedvar;
	NeuronType *type;
	float vars[MAXSTATEVARS];
	type=this->type;
	this->statevars[0]=ElapsedTime;
	for(ivar=0;ivar<type->GetTimeDependentStateVarsNumber();ivar++){
		orderedvar=type->GetStateVarAt(ivar);
		vars[orderedvar]=type->TableAccess(type->GetStateVarTableAt(orderedvar), this->statevars);
	}
	for(ivar=0;ivar<type->GetTimeDependentStateVarsNumber();ivar++){
		orderedvar=type->GetStateVarAt(ivar);
		this->statevars[orderedvar+1]=vars[orderedvar];
	}
	for(ivar=type->GetTimeDependentStateVarsNumber();ivar<type->GetStateVarsNumber();ivar++){
		orderedvar=type->GetStateVarAt(ivar);
		this->statevars[orderedvar+1]=type->TableAccess(type->GetStateVarAt(orderedvar), this->statevars);
	}
}

void Neuron::SynapsisEffect(Interconnection *inter){
	NeuronType *neutype;
	neutype=this->type;
	this->statevars[neutype->GetSynapticVarsAt(inter->GetType())+1]+=inter->GetWeight()*WEIGHTSCALE;
}

double Neuron::FiringEndPrediction(){
	NeuronType *type;
	double pred_time;
	type=this->type;
	pred_time=type->TableAccess(type->GetFiringEndTable(), this->statevars);
	return(pred_time);
}

double Neuron::FiringPrediction(){
	NeuronType *type;
	float pred_time;
	type=this->type;
	pred_time=type->TableAccess(type->GetFiringTable(), this->statevars);
	return(pred_time);
}

void Neuron::GenerateAutoActivity(EventQueue * eventQueue){
	Neuron postfiring=*this;
	postfiring.NeuronUpdate(this->GetPredictionEnd()-this->lastupdate);
	this->predictedspike = postfiring.FiringPrediction();
	if(this->predictedspike != NOPREDICTION){
		this->predictedspike += this->predictionend;
		InternalSpike * nextspike = new InternalSpike(this->predictedspike,this);
		eventQueue->InsertEvent(nextspike);
	}	
}
   		
void Neuron::ProcessInputActivity(InternalSpike * InputSpike){
	this->NeuronUpdate(InputSpike->GetTime()-this->lastupdate); // could be removed
	this->lastupdate = InputSpike->GetTime();
	this->predictionend = this->FiringEndPrediction();
	
	if(this->predictionend != NOPREDICTION){
		this->predictionend += InputSpike->GetTime();
	}else{
		this->predictionend = InputSpike->GetTime()+DEF_REF_PERIOD;
		cerr << "Warning: firing table and firing-end table discrepance (using default ref period)" << endl;
	}
}
   		
void Neuron::ProcessInputSynapticActivity(PropagatedSpike * InputSpike){
	Interconnection * inter = InputSpike->GetSource()->GetOutputConnectionAt(InputSpike->GetTarget());
	this->NeuronUpdate(InputSpike->GetTime()-this->lastupdate);
	this->SynapsisEffect(inter);
	this->lastupdate=InputSpike->GetTime();
}
   		
void Neuron::GenerateOutputActivity(Spike * InputSpike, EventQueue * eventQueue){
	if (this->IsOutputConnected()){
		PropagatedSpike * spike = new PropagatedSpike(InputSpike->GetTime() + this->OutputConnections[0]->GetDelay(), this, 0);
		eventQueue->InsertEvent(spike);
	}
}

void Neuron::PropagateOutputSpike(PropagatedSpike * LastSpike, EventQueue * eventQueue){
	if(this->GetOutputNumber() > LastSpike->GetTarget()+1){
		Interconnection * NewConnection = this->OutputConnections[LastSpike->GetTarget()+1]; 
		PropagatedSpike * nextspike = new PropagatedSpike(LastSpike->GetTime()-this->OutputConnections[LastSpike->GetTarget()]->GetDelay()+NewConnection->GetDelay(),this,LastSpike->GetTarget()+1); 
		eventQueue->InsertEvent(nextspike);
	}
}
   		
void Neuron::GenerateInputActivity(EventQueue * eventQueue){
	this->predictedspike = this->FiringPrediction();
	if(this->predictedspike != NOPREDICTION){
		this->predictedspike += this->lastupdate;
		if(this->predictedspike > this->predictionend){
    		InternalSpike * nextspike = new InternalSpike(this->predictedspike,this);
    		eventQueue->InsertEvent(nextspike);
 		}else{ // needed only for neurons which never stop firing
 			this->GenerateAutoActivity(eventQueue);
 		}
    }
}
   		
long int Neuron::GetIndex() const{
	return this->index;	
}
   		
double Neuron::GetLastUpdate() const{
	return this->lastupdate;
}

/*void Neuron::SetLastUpdate(float LastUpdate){
	this->lastupdate = LastUpdate;
}*/
   		
double Neuron::GetPredictedSpike() const{
	return this->predictedspike;
}

/*void Neuron::SetPredictedSpike(float PredictedSpike){
	this->predictedspike = PredictedSpike;
}*/
   		
double Neuron::GetPredictionEnd() const{
	return this->predictionend;
}

/*void Neuron::SetPredictionEnd(float PredictionEnd){
	this->predictionend = PredictionEnd;
}*/
   		
int Neuron::GetInputNumber() const{
	return InputConnections.size();	
}
   		
int Neuron::GetOutputNumber() const{
	return OutputConnections.size();
}

float Neuron::GetStateVarAt(int index) const{
	return this->statevars[index];
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

NeuronType * Neuron::GetNeuronType() const{
	return this->type;
}


