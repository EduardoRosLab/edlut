/***************************************************************************
 *                           LIFTimeDrivenModel.cpp                        *
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

#include "../../include/neuron_model/LIFTimeDrivenModel.h"
#include "../../include/neuron_model/NeuronState.h"

#include <iostream>
#include <cmath>
#include <string>

#include "../../include/spike/EDLUTFileException.h"
#include "../../include/spike/Neuron.h"
#include "../../include/spike/InternalSpike.h"
#include "../../include/spike/PropagatedSpike.h"
#include "../../include/spike/Interconnection.h"

#include "../../include/simulation/Utils.h"

void LIFTimeDrivenModel::LoadNeuronModel(string ConfigFile) throw (EDLUTFileException){
	FILE *fh;
	long Currentline = 0L;
	fh=fopen(ConfigFile.c_str(),"rt");
	if(fh){
		Currentline=1L;
		skip_comments(fh,Currentline);
		if(fscanf(fh,"%f",&this->eexc)==1){
			skip_comments(fh,Currentline);

			if (fscanf(fh,"%f",&this->einh)==1){
				skip_comments(fh,Currentline);

				if(fscanf(fh,"%f",&this->erest)==1){
					skip_comments(fh,Currentline);

					if(fscanf(fh,"%f",&this->vthr)==1){
						skip_comments(fh,Currentline);

						if(fscanf(fh,"%f",&this->cm)==1){
							skip_comments(fh,Currentline);

							if(fscanf(fh,"%f",&this->texc)==1){
								skip_comments(fh,Currentline);

								if(fscanf(fh,"%f",&this->tinh)==1){
									skip_comments(fh,Currentline);

									if(fscanf(fh,"%f",&this->tref)==1){
										skip_comments(fh,Currentline);

										if(fscanf(fh,"%f",&this->grest)==1){
											skip_comments(fh,Currentline);

											this->InitialState = (NeuronState *) new NeuronState(3);

											for (unsigned int i=0; i<=3; ++i){
												this->InitialState->SetStateVariableAt(i,0.0);
											}

											this->InitialState->SetStateVariableAt(0,this->erest);

											this->InitialState->SetLastUpdateTime(0);
											this->InitialState->SetNextPredictedSpikeTime(NO_SPIKE_PREDICTED);
										} else {
											throw EDLUTFileException(13,60,3,1,Currentline);
										}
									} else {
										throw EDLUTFileException(13,61,3,1,Currentline);
									}
								} else {
									throw EDLUTFileException(13,62,3,1,Currentline);
								}
							} else {
								throw EDLUTFileException(13,63,3,1,Currentline);
							}
						} else {
							throw EDLUTFileException(13,64,3,1,Currentline);
						}
					} else {
						throw EDLUTFileException(13,65,3,1,Currentline);
					}
				} else {
					throw EDLUTFileException(13,66,3,1,Currentline);
				}
			} else {
				throw EDLUTFileException(13,67,3,1,Currentline);
			}
		} else {
			throw EDLUTFileException(13,68,3,1,Currentline);
		}
	}
}

void LIFTimeDrivenModel::SynapsisEffect(NeuronState * State, Interconnection * InputConnection){

	switch (InputConnection->GetType()){
		case 0: {
			float gexc = State->GetStateVariableAt(1);
			gexc += 1e-9*InputConnection->GetWeight();
			State->SetStateVariableAt(1,gexc);
			break;
		}case 1:{
			float ginh = State->GetStateVariableAt(2);
			ginh += 1e-9*InputConnection->GetWeight();
			State->SetStateVariableAt(2,ginh);
			break;
		}
	}
}

LIFTimeDrivenModel::LIFTimeDrivenModel(string NeuronModelID): TimeDrivenNeuronModel(NeuronModelID), eexc(0), einh(0), erest(0), vthr(0), cm(0), texc(0), tinh(0),
		tref(0), grest(0) {
}

LIFTimeDrivenModel::~LIFTimeDrivenModel(void)
{
}

void LIFTimeDrivenModel::LoadNeuronModel() throw (EDLUTFileException){
	this->LoadNeuronModel(this->GetModelID()+".cfg");
}

NeuronState * LIFTimeDrivenModel::InitializeState(){
	return (NeuronState *) new NeuronState(*((NeuronState *) this->InitialState));
}


InternalSpike * LIFTimeDrivenModel::ProcessInputSpike(PropagatedSpike *  InputSpike){
	Interconnection * inter = InputSpike->GetSource()->GetOutputConnectionAt(InputSpike->GetTarget());

	Neuron * TargetCell = inter->GetTarget();

	NeuronState * CurrentState = TargetCell->GetNeuronState();

	// Add the effect of the input spike
	this->SynapsisEffect((NeuronState *)CurrentState,inter);

	return 0;
}

		
bool LIFTimeDrivenModel::UpdateState(NeuronState * State, double CurrentTime){

	float last_update = State->GetLastUpdateTime();
	
	float elapsed_time = CurrentTime - last_update;

	State->AddElapsedTime(elapsed_time);
	
	float last_spike = State->GetLastSpikeTime();

	float vm = State->GetStateVariableAt(0);
	float gexc = State->GetStateVariableAt(1);
	float ginh = State->GetStateVariableAt(2);

	bool spike = false;

	if (last_spike > this->tref) {
		vm = vm + elapsed_time * ( gexc * (this->eexc - vm) + ginh * (this->einh - vm) + grest * (this->erest - vm))/this->cm;
		if (vm > this->vthr){
			State->NewFiredSpike();
			spike = true;
			vm = this->erest;
		}
	}
	gexc = gexc * exp(-(elapsed_time/this->texc));
	ginh = ginh * exp(-(elapsed_time/this->tinh));

	State->SetStateVariableAt(0,vm);
	State->SetStateVariableAt(1,gexc);
	State->SetStateVariableAt(2,ginh);
	State->SetLastUpdateTime(CurrentTime);

	return spike;
}

ostream & LIFTimeDrivenModel::PrintInfo(ostream & out){
	out << "- Leaky Time-Driven Model: " << this->GetModelID() << endl;

	out << "\tExc. Reversal Potential: " << this->eexc << "V\tInh. Reversal Potential: " << this->einh << "V\tResting potential: " << this->erest << "V" << endl;

	out << "\tFiring threshold: " << this->vthr << "V\tMembrane capacitance: " << this->cm << "nS\tExcitatory Time Constant: " << this->texc << "s" << endl;

	out << "\tInhibitory time constant: " << this->tinh << "s\tRefractory Period: " << this->tref << "s\tResting Conductance: " << this->grest << "nS" << endl;

	return out;
}		
