/***************************************************************************
 *                           SRMModel.cpp                                  *
 *                           -------------------                           *
 * copyright            : (C) 2010 by Jesus Garrido                        *
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

#include "../../include/neuron_model/SRMModel.h"

void SRMModel::LoadNeuronModel(string ConfigFile) throw (EDLUTFileException){
	FILE *fh;
	long Currentline = 0L;
	fh=fopen(ConfigFile.c_str(),"rt");
	if(fh){
		Currentline=1L;
		skip_comments(fh,Currentline);
		if(fscanf(fh,"%i",&this->tau)==1){
			skip_comments(fh,Currentline);

			if(fscanf(fh,"%i",&this->vr)==1){
				skip_comments(fh,Currentline);

				if(fscanf(fh,"%i",&this->W)==1){
					skip_comments(fh,Currentline);

					if(fscanf(fh,"%i",&this->r0)==1){
						skip_comments(fh,Currentline);

						if(fscanf(fh,"%i",&this->v0)==1){
							skip_comments(fh,Currentline);

							if(fscanf(fh,"%i",&this->tauabs)==1){
								skip_comments(fh,Currentline);

								if(fscanf(fh,"%i",&this->taurel)==1){
									skip_comments(fh,Currentline);

									if(fscanf(fh,"%i",&this->timestep)==1){
										this->InitialState = (SRMState *) new SRMState(0,8*this->tau,30);

										this->InitialState->SetLastUpdateTime(0);
										this->InitialState->SetNextPredictedSpikeTime(NO_SPIKE_PREDICTED);
									} else {

									}
								} else {

								}
							} else {

							}
						} else {

						}
					} else {

					}
				} else {

				}
			} else {

			}
		} else {

		}
	}
}



void SRMModel::UpdateState(SRMState & State, double CurrentTime){
	State.AddElapsedTime(CurrentTime-State.GetLastUpdateTime());
	State.SetLastUpdateTime(CurrentTime);
}

void SRMModel::SynapsisEffect(SRMState & State, const Interconnection * InputConnection){
	State.AddActivity(InputConnection);
}

double SRMModel::PotentialIncrement(SRMState & State, double CurrentTime){
	double Increment = 0;

	for (int i=0; i<State.GetNumberOfSpikes(); ++i){
		double TimeDifference = CurrentTime - State.GetLastUpdateTime() + State.GetSpikeTimeAt(i);
		double Weight = State.GetInterconnectionAt(i)->GetWeight();

		double EPSPMax = sqrt(tau/2)*exp(-0.5);
		double EPSP = (1/EPSPMax) * sqrt(TimeDifference)*exp(TimeDifference/tau);

		Increment += Weight*this->W*EPSP;
	}

	return Increment;
}

bool SRMModel::CheckSpikeAt(SRMState & State, double CurrentTime){
	if (CurrentTime-State.GetLastSpikeTime()<=this->tauabs){
		return false;
	} else {
		double Potential = this->vr + this->PotentialIncrement(State,Time);
		double FiringRate = this->r0 * log(1+exp((Potential-this->v0)/this->vf));

		double Aux = CurrentTime-State.GetLastSpikeTime()-this->tauabs;
		double Refractoriness = (Aux*Aux)/((this->taurel*this->taurel)+Aux*Aux);

		double Probability = 1 - exp(-FiringRate*Refractoriness);

		return rand()<Probability();
	}
}

double SRMModel::NextFiringPrediction(NeuronState & State){
	double Spike = -1;
	double time = State.GetLastUpdateTime()+this->tauabs+this->timestep;

	while(!CheckSpikeAt(State,time)){
		time += this->timestep;
	}

	return time-State.GetLastUpdateTime();
}

SRMModel::SRMModel(string NeuronModelID): tau(0), vr(0), W(0), r0(0), v0(0), vf(0),
		tauabs(0), taurel(0), timestep(0) {
	this->LoadNeuronModel(NeuronModelID + ".cfg");
}

SRMModel::~SRMModel(){

}

SRMState * SRMModel::InitializeState(){
	return (SRMState *) new SRMState(*(this->InitialState));
}

InternalSpike * SRMModel::GenerateInitialActivity(Neuron &  Cell){
	double Predicted = this->NextFiringPrediction(*(Cell.GetNeuronState()));

	InternalSpike * spike = 0;

	Predicted += Cell.GetNeuronState()->GetLastUpdateTime();

	spike = new InternalSpike(Predicted,&Cell);

	Cell.GetNeuronState()->SetNextPredictedSpikeTime(Predicted);

	return spike;
}

InternalSpike * SRMModel::ProcessInputSpike(PropagatedSpike &  InputSpike){
	Interconnection * inter = InputSpike.GetSource()->GetOutputConnectionAt(InputSpike.GetTarget());

	Neuron * TargetCell = inter->GetTarget();

	NeuronState * CurrentState = TargetCell->GetNeuronState();

	// Update the neuron state until the current time
	this->UpdateState(*CurrentState,InputSpike.GetTime());

	// Add the effect of the input spike
	this->SynapsisEffect(*CurrentState,inter);

	InternalSpike * GeneratedSpike = 0;

	// Check if an spike will be fired
	double NextSpike = this->NextFiringPrediction(*CurrentState);

	NextSpike += CurrentState->GetLastUpdateTime();

	GeneratedSpike = new InternalSpike(NextSpike,TargetCell);

	CurrentState->SetNextPredictedSpikeTime(NextSpike);

	return GeneratedSpike;
}

InternalSpike * SRMModel::GenerateNextSpike(const InternalSpike &  OutputSpike){
	Neuron * SourceCell = OutputSpike.GetSource();

	NeuronState * CurrentState = SourceCell->GetNeuronState();

	InternalSpike * NextSpike = 0;

	this->UpdateState(*CurrentState,OutputSpike.GetTime());

	double PredictedSpike = this->NextFiringPrediction(*CurrentState);

	PredictedSpike += CurrentState->LastUpdate;

	NextSpike = new InternalSpike(PredictedSpike,SourceCell);

	CurrentState->SetNextPredictedSpikeTime(PredictedSpike);

	return NextSpike;
}

bool SRMModel::DiscardSpike(InternalSpike &  OutputSpike){
	return (OutputSpike.GetSource()->GetNeuronState()->GetNextPredictedSpikeTime()!=OutputSpike.GetTime());
}

void SRMModel::GetModelInfo(){

}

