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
#include "../../include/neuron_model/VectorNeuronState.h"

#include <iostream>
#include <cmath>
#include <string>

#ifdef _OPENMP
	#include <omp.h>
#endif

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

							if(fscanf(fh,"%f",&this->tampa)==1){
								skip_comments(fh,Currentline);

								if(fscanf(fh,"%f",&this->tnmda)==1){
									skip_comments(fh,Currentline);
									
									if(fscanf(fh,"%f",&this->tinh)==1){
										skip_comments(fh,Currentline);

										if(fscanf(fh,"%f",&this->tgj)==1){
											skip_comments(fh,Currentline);
											if(fscanf(fh,"%f",&this->tref)==1){
												skip_comments(fh,Currentline);

												if(fscanf(fh,"%f",&this->grest)==1){
													skip_comments(fh,Currentline);

													if(fscanf(fh,"%f",&this->fgj)==1){
														skip_comments(fh,Currentline);

														this->InitialState = (VectorNeuronState *) new VectorNeuronState(5, true);

														//for (unsigned int i=0; i<5; ++i){
														//	this->InitialState->SetStateVariableAt(i,0.0);
														//}

														//this->InitialState->SetStateVariableAt(0,this->erest);

														//this->InitialState->SetLastUpdateTime(0);
														//this->InitialState->SetNextPredictedSpikeTime(NO_SPIKE_PREDICTED);
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
				} else {
					throw EDLUTFileException(13,69,3,1,Currentline);
				}
			} else {
				throw EDLUTFileException(13,70,3,1,Currentline);
			}
		} else {
			throw EDLUTFileException(13,71,3,1,Currentline);
		}
	}
}

void LIFTimeDrivenModel::SynapsisEffect(int index, VectorNeuronState * State, Interconnection * InputConnection){

	switch (InputConnection->GetType()){
		case 0: {
			float gampa = State->GetStateVariableAt(index,1);
			gampa += InputConnection->GetWeight();
			State->SetStateVariableAt(index,1,gampa);
			break;
		}case 1:{
			float gnmda = State->GetStateVariableAt(index,2);
			gnmda += InputConnection->GetWeight();
			State->SetStateVariableAt(index,2,gnmda);
			break;
		}case 2:{
			float ginh = State->GetStateVariableAt(index,3);
			ginh += InputConnection->GetWeight();
			State->SetStateVariableAt(index,3,ginh);
			break;
		}case 3:{
			float ggj = State->GetStateVariableAt(index,4);
			ggj += InputConnection->GetWeight();
			State->SetStateVariableAt(index,4,ggj);
			break;
		}
	}
}

LIFTimeDrivenModel::LIFTimeDrivenModel(string NeuronTypeID, string NeuronModelID): TimeDrivenNeuronModel(NeuronTypeID, NeuronModelID), eexc(0), einh(0), erest(0), vthr(0), cm(0), tampa(0), tnmda(0), tinh(0), tgj(0),
		tref(0), grest(0){
}

LIFTimeDrivenModel::~LIFTimeDrivenModel(void)
{
}

void LIFTimeDrivenModel::LoadNeuronModel() throw (EDLUTFileException){
	this->LoadNeuronModel(this->GetModelID()+".cfg");
}

VectorNeuronState * LIFTimeDrivenModel::InitializeState(){
	//return (VectorNeuronState *) new VectorNeuronState(*((VectorNeuronState *) this->InitialState));
	return this->GetVectorNeuronState();
}


InternalSpike * LIFTimeDrivenModel::ProcessInputSpike(PropagatedSpike *  InputSpike){
	Interconnection * inter = InputSpike->GetSource()->GetOutputConnectionAt(InputSpike->GetTarget());

	Neuron * TargetCell = inter->GetTarget();

	VectorNeuronState * CurrentState = TargetCell->GetVectorNeuronState();


	// Add the effect of the input spike
	this->SynapsisEffect(inter->GetTarget()->GetIndex_VectorNeuronState(),(VectorNeuronState *)CurrentState,inter);


	return 0;
}

	
bool LIFTimeDrivenModel::UpdateState(int index, VectorNeuronState * State, double CurrentTime){

	float inv_cm=1.e-9/this->cm;
	
	bool * internalSpike=State->getInternalSpike();
	int Size=State->GetSizeState();

	float last_update = State->GetLastUpdateTime(0);
	
	float elapsed_time = CurrentTime - last_update;

	float last_spike;

	float exp_gampa = exp(-(elapsed_time/this->tampa));
	float exp_gnmda = exp(-(elapsed_time/this->tnmda));
	float exp_ginh = exp(-(elapsed_time/this->tinh));
	float exp_ggj = exp(-(elapsed_time/this->tgj));

	float vm, gampa, gnmda, ginh, ggj;

	bool spike;

	float iampa, gnmdainf, inmda, iinh;

	float vm_cou;

	int i;

	#pragma omp parallel for default(none) shared(Size, State, internalSpike, CurrentTime, elapsed_time, exp_gampa, exp_gnmda, exp_ginh, exp_ggj, inv_cm) private(i,last_spike,vm, gampa, gnmda, ginh, ggj, spike, iampa, gnmdainf, inmda, iinh, vm_cou)
	for (int i=0; i< Size; i++){

		State->AddElapsedTime(i,elapsed_time);
		
		last_spike = State->GetLastSpikeTime(i);

		vm = State->GetStateVariableAt(i,0);
		gampa = State->GetStateVariableAt(i,1);
		gnmda = State->GetStateVariableAt(i,2);
		ginh = State->GetStateVariableAt(i,3);
		ggj = State->GetStateVariableAt(i,4);

		spike = false;

		if (last_spike > this->tref) {
			iampa = gampa*(this->eexc-vm);
			//gnmdainf = 1.0/(1.0 + exp(-62.0*vm)*1.2/3.57);
			gnmdainf = 1.0/(1.0 + exp(-62.0*vm)*0.336134453);
			inmda = gnmda*gnmdainf*(this->eexc-vm);
			iinh = ginh*(this->einh-vm);
			vm = vm + elapsed_time * (iampa + inmda + iinh + this->grest* (this->erest-vm))*inv_cm;

			vm_cou = vm + this->fgj * ggj;

			if (vm_cou > this->vthr){
				State->NewFiredSpike(i);
				spike = true;
				vm = this->erest;
			}
		}

		internalSpike[i]=spike;

		gampa *= exp_gampa;
		gnmda *= exp_gnmda;
		ginh *= exp_ginh;
		ggj *= exp_ggj;

		State->SetStateVariableAt(i,0,vm);
		State->SetStateVariableAt(i,1,gampa);
		State->SetStateVariableAt(i,2,gnmda);
		State->SetStateVariableAt(i,3,ginh);
		State->SetStateVariableAt(i,4,ggj);
		State->SetLastUpdateTime(i,CurrentTime);
	}

	return false;
}


//-------------------------------------------------------


ostream & LIFTimeDrivenModel::PrintInfo(ostream & out){
	out << "- Leaky Time-Driven Model: " << this->GetModelID() << endl;

	out << "\tExc. Reversal Potential: " << this->eexc << "V\tInh. Reversal Potential: " << this->einh << "V\tResting potential: " << this->erest << "V" << endl;

	out << "\tFiring threshold: " << this->vthr << "V\tMembrane capacitance: " << this->cm << "nS\tAMPA Time Constant: " << this->tampa << "sNMDA Time Constant: " << this->tnmda << "s" << endl;

	out << "\tInhibitory time constant: " << this->tinh << "s\tGap junction time constant: " << this->tgj << "s\tRefractory Period: " << this->tref << "s\tResting Conductance: " << this->grest << "nS" << endl;

	return out;
}	


enum NeuronModelType LIFTimeDrivenModel::GetModelType(){
	return TIME_DRIVEN_MODEL_CPU;
}


void LIFTimeDrivenModel::InitializeStates(int N_neurons){
	float inicialization[] = {erest,0.0,0.0,0.0,0.0};
	InitialState->InitializeStates(N_neurons, inicialization);
}

