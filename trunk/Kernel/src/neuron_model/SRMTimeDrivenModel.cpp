/***************************************************************************
 *                           SRMTimeDrivenModel.cpp                        *
 *                           -------------------                           *
 * copyright            : (C) 2011 by Jesus Garrido                        *
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

#include <iostream>
#include <cmath>
#include <string>

#include "../../include/neuron_model/SRMTimeDrivenModel.h"
#include "../../include/neuron_model/SRMState.h"
#include "../../include/neuron_model/NeuronState.h"

#include "../../include/spike/EDLUTFileException.h"
#include "../../include/spike/Neuron.h"
#include "../../include/spike/InternalSpike.h"
#include "../../include/spike/PropagatedSpike.h"

#include "../../include/simulation/Utils.h"

using namespace std;

void SRMTimeDrivenModel::LoadNeuronModel(string ConfigFile) throw (EDLUTFileException){
	FILE *fh;
	long Currentline = 0L;
	fh=fopen(ConfigFile.c_str(),"rt");
	if(fh){
		Currentline=1L;
		skip_comments(fh,Currentline);
		if(fscanf(fh,"%f",&this->tau)==1){
			skip_comments(fh,Currentline);

			if(fscanf(fh,"%f",&this->vr)==1){
				skip_comments(fh,Currentline);

				if(fscanf(fh,"%f",&this->W)==1){
					skip_comments(fh,Currentline);

					if(fscanf(fh,"%f",&this->r0)==1){
						skip_comments(fh,Currentline);

						if(fscanf(fh,"%f",&this->v0)==1){
							skip_comments(fh,Currentline);

							if(fscanf(fh,"%f",&this->vf)==1){
								skip_comments(fh,Currentline);

								if(fscanf(fh,"%f",&this->tauabs)==1){
									skip_comments(fh,Currentline);

									if(fscanf(fh,"%f",&this->taurel)==1){
										skip_comments(fh,Currentline);

										this->InitialState = (SRMState *) new SRMState(4,8*this->tau,30);

										for (unsigned int i=0; i<=4; ++i){
											this->InitialState->SetStateVariableAt(i,0.0);
										}

										this->InitialState->SetLastUpdateTime(0);
										this->InitialState->SetNextPredictedSpikeTime(NO_SPIKE_PREDICTED);
									} else {
										throw EDLUTFileException(13,50,3,1,Currentline);
									}
								} else {
									throw EDLUTFileException(13,51,3,1,Currentline);
								}
							} else {
								throw EDLUTFileException(13,52,3,1,Currentline);
							}
						} else {
							throw EDLUTFileException(13,53,3,1,Currentline);
						}
					} else {
						throw EDLUTFileException(13,54,3,1,Currentline);
					}
				} else {
					throw EDLUTFileException(13,55,3,1,Currentline);
				}
			} else {
				throw EDLUTFileException(13,56,3,1,Currentline);
			}
		} else {
			throw EDLUTFileException(13,58,3,1,Currentline);
		}
	}
}

void SRMTimeDrivenModel::SynapsisEffect(SRMState * State, Interconnection * InputConnection){
	State->AddActivity(InputConnection);
}

double SRMTimeDrivenModel::PotentialIncrement(SRMState * State){
	double Increment = 0;

	for (unsigned int i=0; i<State->GetNumberOfSpikes(); ++i){
		double TimeDifference = State->GetSpikeTimeAt(i);
		double Weight = State->GetInterconnectionAt(i)->GetWeight();

		//int Position = round(TimeDifference/this->EPSPStep);
		double EPSPMax = sqrt(this->tau/2)*exp(-0.5);

		double EPSP = sqrt(TimeDifference)*exp(-(TimeDifference/this->tau))/EPSPMax;

		//Increment += Weight*this->W*EPSP[Position];
		Increment += Weight*this->W*EPSP;
	}

	return Increment;
}

bool SRMTimeDrivenModel::CheckSpikeAt(SRMState * State, double CurrentTime){
	double Probability = State->GetStateVariableAt(4);
	return (((double) rand())/RAND_MAX<Probability);
}

SRMTimeDrivenModel::SRMTimeDrivenModel(string NeuronModelID): TimeDrivenNeuronModel(NeuronModelID), tau(0), vr(0), W(0), r0(0), v0(0), vf(0),
		tauabs(0), taurel(0) {

}

SRMTimeDrivenModel::~SRMTimeDrivenModel(){

}

void SRMTimeDrivenModel::LoadNeuronModel() throw (EDLUTFileException) {

	this->LoadNeuronModel(this->GetModelID() + ".cfg");
}

NeuronState * SRMTimeDrivenModel::InitializeState(){
	return (SRMState *) new SRMState(*((SRMState *) this->InitialState));
}

InternalSpike * SRMTimeDrivenModel::ProcessInputSpike(PropagatedSpike *  InputSpike){
	Interconnection * inter = InputSpike->GetSource()->GetOutputConnectionAt(InputSpike->GetTarget());

	Neuron * TargetCell = inter->GetTarget();

	NeuronState * CurrentState = TargetCell->GetNeuronState();

	// Update Cell State
	this->UpdateState(TargetCell->GetNeuronState(),InputSpike->GetTime());

	// Add the effect of the input spike
	this->SynapsisEffect((SRMState *)CurrentState,inter);

	return 0;
}

bool SRMTimeDrivenModel::UpdateState(NeuronState * State, double CurrentTime){

	double ElapsedTime = CurrentTime-State->GetLastUpdateTime();

	State->AddElapsedTime(ElapsedTime);

	double Potential = this->vr + this->PotentialIncrement((SRMState *) State);
	State->SetStateVariableAt(1,Potential);

	double FiringRate;

	if((Potential-this->v0)*1000 > (10*this->vf)){
		FiringRate = this->r0*1000*(Potential-this->v0)/this->vf;
	} else {
		double texp=exp((Potential-this->v0)*1000/this->vf);
	    FiringRate =this->r0*log(1+texp);
	}

	//double texp = exp((Potential-this->v0)/this->vf);
	//double FiringRate = this->r0 * log(1+texp);
	State->SetStateVariableAt(2,FiringRate);

	double TimeSinceSpike = State->GetLastSpikeTime();
	double Aux = TimeSinceSpike-this->tauabs;
	double Refractoriness = 0;

	if (TimeSinceSpike>this->tauabs){
		Refractoriness = 1./(1.+(this->taurel*this->taurel)/(Aux*Aux));
	}
	State->SetStateVariableAt(3,Refractoriness);

	double Probability = (1 - exp(-FiringRate*Refractoriness))*ElapsedTime/0.001;
	State->SetStateVariableAt(4,Probability);

	State->SetLastUpdateTime(CurrentTime);

	if (this->CheckSpikeAt((SRMState *) State, CurrentTime)){
		State->NewFiredSpike();
		return true;
	}

	return false;
}

ostream & SRMTimeDrivenModel::PrintInfo(ostream & out) {
	out << "- SRM Time-Driven Model: " << this->GetModelID() << endl;

	out << "\tTau: " << this->tau << "s\tVresting: " << this->vr << endl;

	out << "\tWeight Scale: " << this->W << "\tFiring Rate: " << this->r0 << "Hz\tVthreshold: " << this->v0 << "V" << endl;

	out << "\tGain Factor: " << this->vf << "\tAbsolute Refractory Period: " << this->tauabs << "s\tRelative Refractory Period: " << this->taurel << "s" << endl;

	return out;
}
