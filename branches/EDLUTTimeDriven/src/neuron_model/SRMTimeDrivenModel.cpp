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

			if (fscanf(fh,"%f",&this->EPSPStep)==1){
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

											if(fscanf(fh,"%f",&this->steptime)==1){
												skip_comments(fh,Currentline);

												this->InitialState = (SRMState *) new SRMState(4,8*this->tau,30);

												for (unsigned int i=0; i<=4; ++i){
													this->InitialState->SetStateVariableAt(i,0.0);
												}

												this->InitialState->SetLastUpdateTime(0);
												this->InitialState->SetNextPredictedSpikeTime(NO_SPIKE_PREDICTED);
											} else {
												 throw EDLUTFileException(13,49,3,1,Currentline);
											}
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
				throw EDLUTFileException(13,57,3,1,Currentline);
			}
		} else {
			throw EDLUTFileException(13,58,3,1,Currentline);
		}
	}
}

void SRMTimeDrivenModel::PrecalculateEPSP(){
	unsigned int EPSPWindowSize = this->tau*8/this->EPSPStep;

	this->EPSP = (double *) new double [EPSPWindowSize];

	for (unsigned int i=0; i<EPSPWindowSize; ++i){
		this->EPSP[i] = sqrt(2*i*this->EPSPStep/this->tau)*exp(0.5-(i*this->EPSPStep)/tau);
	}
}

void SRMTimeDrivenModel::SynapsisEffect(SRMState * State, Interconnection * InputConnection){
	State->AddActivity(InputConnection);
}

double SRMTimeDrivenModel::PotentialIncrement(SRMState * State, double CurrentTime){
	double Increment = 0;

	for (unsigned int i=0; i<State->GetNumberOfSpikes(); ++i){
		double TimeDifference = CurrentTime - State->GetLastUpdateTime() + State->GetSpikeTimeAt(i);
		double Weight = State->GetInterconnectionAt(i)->GetWeight();

		Increment += Weight*this->W*EPSP[(int) (TimeDifference/this->EPSPStep)];
	}

	return Increment;
}

bool SRMTimeDrivenModel::CheckSpikeAt(SRMState * State, double CurrentTime){
	double Probability = State->GetStateVariableAt(4);
	return (((double) rand())/RAND_MAX<Probability);
}

SRMTimeDrivenModel::SRMTimeDrivenModel(string NeuronModelID): TimeDrivenNeuronModel(NeuronModelID), tau(0), EPSPStep(0), vr(0), W(0), r0(0), v0(0), vf(0),
		tauabs(0), taurel(0), EPSP(0) {

}

SRMTimeDrivenModel::~SRMTimeDrivenModel(){

}

void SRMTimeDrivenModel::LoadNeuronModel() throw (EDLUTFileException) {

	this->LoadNeuronModel(this->GetModelID() + ".cfg");

	this->PrecalculateEPSP();
}

NeuronState * SRMTimeDrivenModel::InitializeState(){
	return (SRMState *) new SRMState(*((SRMState *) this->InitialState));
}

InternalSpike * SRMTimeDrivenModel::ProcessInputSpike(PropagatedSpike *  InputSpike){
	Interconnection * inter = InputSpike->GetSource()->GetOutputConnectionAt(InputSpike->GetTarget());

	Neuron * TargetCell = inter->GetTarget();

	NeuronState * CurrentState = TargetCell->GetNeuronState();

	// Add the effect of the input spike
	this->SynapsisEffect((SRMState *)CurrentState,inter);

	return 0;
}

bool SRMTimeDrivenModel::UpdateState(NeuronState * State, double CurrentTime){

	double Potential = this->vr + this->PotentialIncrement((SRMState *) State,CurrentTime);
	State->SetStateVariableAt(1,Potential);

	double FiringRate = this->r0 * log(1+exp((Potential-this->v0)/this->vf));
	State->SetStateVariableAt(2,FiringRate);

	double TimeSinceSpike = CurrentTime-State->GetLastUpdateTime()+State->GetLastSpikeTime();
	double Aux = TimeSinceSpike-this->tauabs;
	double Refractoriness = 1./(1.+(this->taurel*this->taurel)/(Aux*Aux));
	State->SetStateVariableAt(3,Refractoriness);

	double Probability = 0;

	if (TimeSinceSpike>this->tauabs){
		Probability = (1 - exp(-FiringRate*Refractoriness))*this->steptime/0.001;
	}
	State->SetStateVariableAt(4,Probability);

	State->AddElapsedTime(CurrentTime-State->GetLastUpdateTime());
	State->SetLastUpdateTime(CurrentTime);

	if (this->CheckSpikeAt((SRMState *) State, CurrentTime)){
		State->NewFiredSpike();
		return true;
	}

	return false;
}

ostream & SRMTimeDrivenModel::PrintInfo(ostream & out) {
	out << "- SRM Time-Driven Model: " << this->GetModelID() << endl;

	out << "\tTau: " << this->tau << "s\tEPSP Step: " << this->EPSPStep << "s\tVresting: " << this->vr << endl;

	out << "\tWeight Scale: " << this->W << "\tFiring Rate: " << this->r0 << "Hz\tVthreshold: " << this->v0 << "V" << endl;

	out << "\tGain Factor: " << this->vf << "\tAbsolute Refractory Period: " << this->tauabs << "s\tRelative Refractory Period: " << this->taurel << "s" << endl;

	return out;
}
