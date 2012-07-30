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

							if(fscanf(fh,"%f",&this->texc)==1){
								skip_comments(fh,Currentline);

								if(fscanf(fh,"%f",&this->tinh)==1){
									skip_comments(fh,Currentline);

									if(fscanf(fh,"%f",&this->tref)==1){
										skip_comments(fh,Currentline);

										if(fscanf(fh,"%f",&this->grest)==1){
											skip_comments(fh,Currentline);

											this->InitialState = (VectorNeuronState *) new VectorNeuronState(3, true);

											//for (unsigned int i=0; i<3; ++i){
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
	}
}

void LIFTimeDrivenModel::SynapsisEffect(int index, VectorNeuronState * State, Interconnection * InputConnection){

	switch (InputConnection->GetType()){
		case 0: {
			float gexc = State->GetStateVariableAt(index,1);
			gexc += 1e-9*InputConnection->GetWeight();
			State->SetStateVariableAt(index,1,gexc);
			break;
		}case 1:{
			float ginh = State->GetStateVariableAt(index,2);
			ginh += 1e-9*InputConnection->GetWeight();
			State->SetStateVariableAt(index,2,ginh);
			break;
		}
	}
}

LIFTimeDrivenModel::LIFTimeDrivenModel(string NeuronTypeID, string NeuronModelID): TimeDrivenNeuronModel(NeuronTypeID, NeuronModelID), eexc(0), einh(0), erest(0), vthr(0), cm(0), texc(0), tinh(0),
		tref(0), grest(0) {
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

	bool * internalSpike=State->getInternalSpike();
	int Size=State->GetSizeState();
	int i;
	double last_update,elapsed_time,last_spike;
	float vm,gexc,ginh;
	bool spike;

#pragma omp parallel for default(none) shared(Size, State, internalSpike, CurrentTime) private(i,last_update, elapsed_time,last_spike,vm,gexc,ginh,spike)
	for (int i=0; i< Size; i++){

		last_update = State->GetLastUpdateTime(i);
		elapsed_time = CurrentTime - last_update;
	
		State->AddElapsedTime(i, elapsed_time);
	
		last_spike = State->GetLastSpikeTime(i);

		vm = State->GetStateVariableAt(i,0);
		gexc = State->GetStateVariableAt(i,1);
		ginh = State->GetStateVariableAt(i,2);

		spike = false;

		if (last_spike > this->tref) {
			vm = vm + elapsed_time * ( gexc * (this->eexc - vm) + ginh * (this->einh - vm) + grest * (this->erest - vm))/this->cm;
			if (vm > this->vthr){
				State->NewFiredSpike(i);
				spike = true;
				vm = this->erest;
			}
		}
		internalSpike[i]=spike;

		gexc = gexc * exp(-(elapsed_time/this->texc));
		ginh = ginh * exp(-(elapsed_time/this->tinh));

		State->SetStateVariableAt(i,0,vm);
		State->SetStateVariableAt(i,1,gexc);
		State->SetStateVariableAt(i,2,ginh);
		State->SetLastUpdateTime(i, CurrentTime);
	}
	return false;
}




//bool LIFTimeDrivenModel::UpdateState(int index, VectorNeuronState * State, double CurrentTime){
//	float inv_cm=1/this->cm;
//
//
//	double last_update = State->GetLastUpdateTime(0);
//	double elapsed_time = CurrentTime - last_update;
//	float elapsed_time1=elapsed_time;
//
//	float exponential1=exp(-(elapsed_time1/this->texc));
//	float exponential2=exp(-(elapsed_time1/this->tinh));
//
//	bool * internalSpike=State->getInternalSpike();
//	int i;
//	double last_spike;
//	float vm,gexc,ginh;
//	bool spike;
//	int Size=State->GetSizeState();
//
//#pragma omp parallel for default(none) shared(Size, State, internalSpike, CurrentTime,last_update, elapsed_time, exponential1,exponential2, inv_cm) private(i,last_spike,vm,gexc,ginh,spike)
//	for (i=0; i<Size; i++){
//	
//		State->AddElapsedTime(i, elapsed_time);
//	
//		last_spike = State->GetLastSpikeTime(i);
//
//		vm = State->GetStateVariableAt(i,0);
//		gexc = State->GetStateVariableAt(i,1);
//		ginh = State->GetStateVariableAt(i,2);
//
//		spike = false;
//
//		if (last_spike > this->tref) {
//			vm += elapsed_time * ( gexc * (this->eexc - vm) + ginh * (this->einh - vm) + grest * (this->erest - vm))*inv_cm;
//			if (vm > this->vthr){
//				State->NewFiredSpike(i);
//				spike = true;
//				vm = this->erest;
//			}
//		}
//		internalSpike[i]=spike;
//
//		gexc *= exponential1;
//		ginh *= exponential2;
//
//		State->SetStateVariableAt(i,0,vm);
//		State->SetStateVariableAt(i,1,gexc);
//		State->SetStateVariableAt(i,2,ginh);
//		State->SetLastUpdateTime(i, CurrentTime);
//	}
//
//	return false;
//}

ostream & LIFTimeDrivenModel::PrintInfo(ostream & out){
	out << "- Leaky Time-Driven Model: " << this->GetModelID() << endl;

	out << "\tExc. Reversal Potential: " << this->eexc << "V\tInh. Reversal Potential: " << this->einh << "V\tResting potential: " << this->erest << "V" << endl;

	out << "\tFiring threshold: " << this->vthr << "V\tMembrane capacitance: " << this->cm << "nS\tExcitatory Time Constant: " << this->texc << "s" << endl;

	out << "\tInhibitory time constant: " << this->tinh << "s\tRefractory Period: " << this->tref << "s\tResting Conductance: " << this->grest << "nS" << endl;

	return out;
}	


enum NeuronModelType LIFTimeDrivenModel::GetModelType(){
	return TIME_DRIVEN_MODEL_CPU;
}


void LIFTimeDrivenModel::InitializeStates(int N_neurons){
	float inicialization[] = {erest,0.0,0.0};
	InitialState->InitializeStates(N_neurons, inicialization);
}

