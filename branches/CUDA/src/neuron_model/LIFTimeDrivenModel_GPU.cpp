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

#include "../../include/neuron_model/LIFTimeDrivenModel_GPU.h"
#include "../../include/neuron_model/VectorNeuronState.h"
#include "../../include/neuron_model/VectorNeuronState_GPU.h"

#include <iostream>
#include <cmath>
#include <string>

#include "../../include/spike/EDLUTFileException.h"
#include "../../include/spike/Neuron.h"
#include "../../include/spike/InternalSpike.h"
#include "../../include/spike/PropagatedSpike.h"
#include "../../include/spike/Interconnection.h"

#include "../../include/simulation/Utils.h"

#include "../../include/neuron_model/LIFTimeDrivenModel_CUDA.h"
#include "../../include/cudaError.h"
//Library for CUDA
#include <cutil_inline.h>

void LIFTimeDrivenModel_GPU::LoadNeuronModel(string ConfigFile) throw (EDLUTFileException){
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

HANDLE_ERROR(cudaMalloc((void**)&parameter, 9*sizeof(float)));
float Parameter[9] ={eexc,einh,erest,vthr,cm,texc,tinh,tref,grest};
HANDLE_ERROR(cudaMemcpy(parameter,Parameter,9*sizeof(float),cudaMemcpyHostToDevice));
this->InitialState = (VectorNeuronState_GPU *) new VectorNeuronState_GPU(3);


//											this->InitialState = (NeuronState *) new NeuronState(3);

//											for (unsigned int i=0; i<3; ++i){
//												this->InitialState->SetStateVariableAt(i,0.0);
//											}

//											this->InitialState->SetStateVariableAt(0,this->erest);

//											this->InitialState->SetLastUpdateTime(0);
//											this->InitialState->SetNextPredictedSpikeTime(NO_SPIKE_PREDICTED);
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

void LIFTimeDrivenModel_GPU::SynapsisEffect(int index, VectorNeuronState_GPU * state, Interconnection * InputConnection){

	switch (InputConnection->GetType()){
		case 0: {
			state->AuxStateCPU[2*index]+=1e-9*InputConnection->GetWeight();
			break;
		}case 1:{
			state->AuxStateCPU[2*index+1]+=1e-9*InputConnection->GetWeight();
			break;
		}
	}
}

LIFTimeDrivenModel_GPU::LIFTimeDrivenModel_GPU(string NeuronTypeID, string NeuronModelID): TimeDrivenNeuronModel(NeuronTypeID, NeuronModelID), eexc(0), einh(0), erest(0), vthr(0), cm(0), texc(0), tinh(0),
		tref(0), grest(0),time(0), counter(0), size(100){
}

LIFTimeDrivenModel_GPU::~LIFTimeDrivenModel_GPU(void){
	destroySynchronize();
	HANDLE_ERROR(cudaFree(parameter));
}

void LIFTimeDrivenModel_GPU::LoadNeuronModel() throw (EDLUTFileException){
	this->LoadNeuronModel(this->GetModelID()+".cfg");
}

VectorNeuronState * LIFTimeDrivenModel_GPU::InitializeState(){
//	return (NeuronState *) new NeuronState(*((NeuronState *) this->InitialState));
	return this->GetVectorNeuronState();
}


InternalSpike * LIFTimeDrivenModel_GPU::ProcessInputSpike(PropagatedSpike *  InputSpike){
	Interconnection * inter = InputSpike->GetSource()->GetOutputConnectionAt(InputSpike->GetTarget());

	Neuron * TargetCell = inter->GetTarget();

	int indexGPU =TargetCell->GetIndex_VectorNeuronState();

	VectorNeuronState_GPU * state = (VectorNeuronState_GPU *) this->InitialState;

	// Add the effect of the input spike
	this->SynapsisEffect(inter->GetTarget()->GetIndex_VectorNeuronState(), state, inter);

	return 0;
}

		
bool LIFTimeDrivenModel_GPU::UpdateState(int index, VectorNeuronState * State, double CurrentTime){
	
	counter++;

	VectorNeuronState_GPU *state = (VectorNeuronState_GPU *) State;
	if((counter%size)==0){
		float elapsed_time;
		UpdateStateGPU(&elapsed_time,parameter, state->AuxStateGPU, state->AuxStateCPU, state->VectorNeuronStates_GPU, state->LastUpdateGPU, state->LastSpikeTimeGPU, state->InternalSpikeGPU, state->InternalSpikeCPU, state->SizeStates, CurrentTime);
		time+=elapsed_time;
	}else{
		UpdateStateGPU(parameter, state->AuxStateGPU, state->AuxStateCPU, state->VectorNeuronStates_GPU, state->LastUpdateGPU, state->LastSpikeTimeGPU, state->InternalSpikeGPU, state->InternalSpikeCPU, state->SizeStates, CurrentTime);
	}

	memset(state->AuxStateCPU,0,2*state->SizeStates*sizeof(float));

	if(this->GetVectorNeuronState()->Get_Is_Monitored()){
		HANDLE_ERROR(cudaMemcpy(state->VectorNeuronStates,state->VectorNeuronStates_GPU,3*state->SizeStates*sizeof(float),cudaMemcpyDeviceToHost));
		HANDLE_ERROR(cudaMemcpy(state->LastUpdate,state->LastUpdateGPU,state->SizeStates*sizeof(double),cudaMemcpyDeviceToHost));
		HANDLE_ERROR(cudaMemcpy(state->LastSpikeTime,state->LastSpikeTimeGPU,state->SizeStates*sizeof(double),cudaMemcpyDeviceToHost));
		synchronizeGPU_CPU();
	}

	return false;

}

ostream & LIFTimeDrivenModel_GPU::PrintInfo(ostream & out){
	out << "- Leaky Time-Driven Model: " << this->GetModelID() << endl;

	out << "\tExc. Reversal Potential: " << this->eexc << "V\tInh. Reversal Potential: " << this->einh << "V\tResting potential: " << this->erest << "V" << endl;

	out << "\tFiring threshold: " << this->vthr << "V\tMembrane capacitance: " << this->cm << "nS\tExcitatory Time Constant: " << this->texc << "s" << endl;

	out << "\tInhibitory time constant: " << this->tinh << "s\tRefractory Period: " << this->tref << "s\tResting Conductance: " << this->grest << "nS" << endl;

	return out;
}	


enum NeuronModelType LIFTimeDrivenModel_GPU::GetModelType(){
	return TIME_DRIVEN_MODEL_GPU;
}

void LIFTimeDrivenModel_GPU::InitializeStates(int N_neurons){
	createSynchronize();
	VectorNeuronState_GPU * state = (VectorNeuronState_GPU *) this->InitialState;
	
	float inicialization[] = {erest,0.0,0.0};
	state->InitializeStatesGPU(N_neurons, inicialization);
}



