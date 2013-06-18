/***************************************************************************
 *                           SRMDetTimeDrivenModel.cpp                     *
 *                           -------------------------                     *
 * copyright            : (C) 2013 by Jesus Garrido and Francisco Naveros  *
 * email                : jesusgarrido@ugr.es, fnaveros@atc.ugr.es         *
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 3 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

#include <iostream>
#include <cmath>
#include <string>

#include "../../include/neuron_model/SRMDetTimeDrivenModel.h"
#include "../../include/neuron_model/VectorSRMState.h"
#include "../../include/neuron_model/VectorNeuronState.h"

#include "../../include/spike/EDLUTFileException.h"
#include "../../include/spike/Neuron.h"
#include "../../include/spike/InternalSpike.h"
#include "../../include/spike/PropagatedSpike.h"

#include "../../include/simulation/Utils.h"

using namespace std;

void SRMDetTimeDrivenModel::LoadNeuronModel(string ConfigFile) throw (EDLUTFileException){
	FILE *fh;
	long Currentline = 0L;

	fh=fopen(ConfigFile.c_str(),"rt");
	if(!fh){
		// Error: Neuron model file doesn't exist
		throw EDLUTFileException(13,25,31,1,Currentline);
	}

	Currentline=1L;
	skip_comments(fh,Currentline);
	if (!fscanf(fh, "%f", &this->tau_m)==1){
		throw EDLUTFileException(13,81,3,1,Currentline);
	}

	skip_comments(fh,Currentline);
	if (!fscanf(fh, "%f", &this->tau_s)==1){
		throw EDLUTFileException(13,82,3,1,Currentline);
	}
	
	skip_comments(fh,Currentline);
	if (!fscanf(fh, "%f", &this->threshold)==1){
		throw EDLUTFileException(13,83,3,1,Currentline);
	}

	skip_comments(fh,Currentline);
	if (!fscanf(fh, "%f", &this->k1)==1){
		throw EDLUTFileException(13,84,3,1,Currentline);
	}
	
	skip_comments(fh,Currentline);
	if (!fscanf(fh, "%f", &this->k2)==1){
		throw EDLUTFileException(13,85,3,1,Currentline);
	}

	skip_comments(fh,Currentline);
	if (!fscanf(fh, "%f", &this->buffer_amplitude)==1){
		throw EDLUTFileException(13,86,3,1,Currentline);
	}

	skip_comments(fh,Currentline);
	if (!fscanf(fh, "%f", &this->refractory)==1){
		throw EDLUTFileException(13,87,3,1,Currentline);
	}
	
	// Initialize the neuron state
	// With debugging proposals, we store the membrane potential and the exponential values too.
	this->InitialState = (VectorSRMState *) new VectorSRMState(1,1,true);
}

void SRMDetTimeDrivenModel::SynapsisEffect(int index, VectorSRMState * State, Interconnection * InputConnection){
	State->AddActivity(index, InputConnection);
}

float SRMDetTimeDrivenModel::PotentialIncrement(int index, VectorSRMState * State){
	float Increment = 0;

	VectorBufferedState::Iterator itEnd = State->End();
		
	for (VectorBufferedState::Iterator it=State->Begin(index,0); it!=itEnd; ++it){
		float TimeDifference = it.GetSpikeTime();
		float Weight = it.GetConnection()->GetWeight();

		float ExpTauM = exp(-TimeDifference/this->tau_m);
		float ExpTauS = exp(-TimeDifference/this->tau_s);

		float EPSP = this->k*(ExpTauM-ExpTauS);

		// Inhibitory channels must define negative W values
		Increment += Weight*EPSP;
	}

	return Increment;
}

bool SRMDetTimeDrivenModel::CheckSpikeAt(int index, VectorSRMState * State, double CurrentTime){
	return(State->GetStateVariableAt(index,0)>=this->threshold);
}

SRMDetTimeDrivenModel::SRMDetTimeDrivenModel(string NeuronTypeID, string NeuronModelID): TimeDrivenNeuronModel(NeuronTypeID, NeuronModelID), 
	tau_m(0), tau_s(0), threshold(0), refractory(0), k(0), k1(0), k2(0), buffer_amplitude(0) {

}

SRMDetTimeDrivenModel::~SRMDetTimeDrivenModel(){
	
}

void SRMDetTimeDrivenModel::LoadNeuronModel() throw (EDLUTFileException) {

	this->LoadNeuronModel(this->GetModelID() + ".cfg");
}

VectorNeuronState * SRMDetTimeDrivenModel::InitializeState(){
	//return (VectorSRMState *) new VectorSRMState(*((VectorSRMState *) this->InitialState));
	return ((VectorSRMState *) InitialState);
}

InternalSpike * SRMDetTimeDrivenModel::ProcessInputSpike(PropagatedSpike *  InputSpike){
	Interconnection * inter = InputSpike->GetSource()->GetOutputConnectionAt(InputSpike->GetTarget());

	Neuron * TargetCell = inter->GetTarget();

	VectorNeuronState * CurrentState = TargetCell->GetVectorNeuronState();

	InternalSpike * ProducedSpike = 0;

	// Update Cell State
	if (this->UpdateState(inter->GetTarget()->GetIndex_VectorNeuronState(),TargetCell->GetVectorNeuronState(),InputSpike->GetTime())){
		ProducedSpike = new InternalSpike(InputSpike->GetTime(),TargetCell);
	}

	// Add the effect of the input spike
	this->SynapsisEffect(inter->GetTarget()->GetIndex_VectorNeuronState(),(VectorSRMState *)CurrentState,inter);

	return ProducedSpike;
}


bool SRMDetTimeDrivenModel::UpdateState(int index, VectorNeuronState * State, double CurrentTime){

	VectorSRMState * SRMStateAux = (VectorSRMState *) State;

	if(index!=-1){
		double ElapsedTime = CurrentTime-SRMStateAux->GetLastUpdateTime(index);
				
		SRMStateAux->AddElapsedTime(index, ElapsedTime);

		// Calculate the post-spike potential
		float PSP = 0.0f;
		float TimeSinceSpike = SRMStateAux->GetLastSpikeTime(index);

		float ExpTauM = exp(-TimeSinceSpike/this->tau_m);
		float ExpTauS = exp(-TimeSinceSpike/this->tau_s);

		
		if (TimeSinceSpike<this->buffer_amplitude){
			PSP = this->threshold*(this->k1*ExpTauM-this->k2*(ExpTauM-ExpTauS));
		}

		float Potential = PSP + this->PotentialIncrement(index,SRMStateAux);
		SRMStateAux->SetStateVariableAt(index,0,Potential);
		
		SRMStateAux->SetLastUpdateTime(index,CurrentTime);

		if (TimeSinceSpike>this->refractory && this->CheckSpikeAt(index,SRMStateAux, CurrentTime)){
			SRMStateAux->NewFiredSpike(index);
			SRMStateAux->ClearBuffer(index,0);
			return true;
		}

		return false;
	
	
	}else{

		bool * internalSpike=SRMStateAux->getInternalSpike();
		int Size=SRMStateAux->GetSizeState();
		int i;
		float ElapsedTime, PSP, TimeSinceSpike, Potential, ExpTauM, ExpTauS;

#pragma omp parallel for default(none) shared(Size, SRMStateAux, internalSpike, CurrentTime) private(i,ElapsedTime, PSP, TimeSinceSpike, Potential, ExpTauM, ExpTauS)
		for (i=0; i<Size; i++){

			ElapsedTime = (float) (CurrentTime-SRMStateAux->GetLastUpdateTime(i));

			SRMStateAux->AddElapsedTime(i, ElapsedTime);

			// Calculate the post-spike potential
			PSP = 0.0f;
			TimeSinceSpike = SRMStateAux->GetLastSpikeTime(i);

			ExpTauM = exp(-TimeSinceSpike/this->tau_m);
			ExpTauS = exp(-TimeSinceSpike/this->tau_s);

		
			if (TimeSinceSpike<this->buffer_amplitude){
				PSP = this->threshold*(this->k1*ExpTauM-this->k2*(ExpTauM-ExpTauS));
			}

			Potential = PSP + this->PotentialIncrement(i,SRMStateAux);
			SRMStateAux->SetStateVariableAt(i,0,Potential);
			
			SRMStateAux->SetLastUpdateTime(i,CurrentTime);

			if (TimeSinceSpike>this->refractory && this->CheckSpikeAt(i,SRMStateAux, CurrentTime)){
				SRMStateAux->NewFiredSpike(i);
				SRMStateAux->ClearBuffer(i,0);
				internalSpike[i]=true;
			} else {
				internalSpike[i]=false;
			}
		}
	}
	return false;
}




ostream & SRMDetTimeDrivenModel::PrintInfo(ostream & out) {
	out << "- SRM Deterministic Time-Driven Model: " << this->GetModelID() << endl;

	out << "\tNumber of channels: " << 1 << endl;

	out << "\tTau membrane: " << this->tau_m;
	
	out << "\tTau synapses: " << this->tau_s;
	
	out << "\tFiring threshold: " << this->threshold;
	
	out << "\tRefractory period: " << this->refractory;
	
	out << "\tK1: " << this->k1;
	
	out << "\tK2: " << this->k2;
	
	out << "\tBuffer amplitude: " << this->buffer_amplitude;
	
	return out;
}

enum NeuronModelType SRMDetTimeDrivenModel::GetModelType(){
	return TIME_DRIVEN_MODEL_CPU;
}


void SRMDetTimeDrivenModel::InitializeStates(int N_neurons){

	VectorSRMState * state = (VectorSRMState *) this->InitialState;

	float inicialization[] = {0.0};
	//Initialize the state variables
	state->InitializeSRMStates(N_neurons, inicialization);

	for (int j=0; j<N_neurons; j++){
		// Initialize the amplitude of each buffer
		state->SetBufferAmplitude(j,0,this->buffer_amplitude);
	}

	float MaxPoint = log(this->tau_m/this->tau_s)/(1/this->tau_s-1/this->tau_m);
	float MaxValue = exp(-MaxPoint/this->tau_m)-exp(-MaxPoint/this->tau_s);

	// Calculate the normalization factor K
	this->k = 1.f/MaxValue;
}