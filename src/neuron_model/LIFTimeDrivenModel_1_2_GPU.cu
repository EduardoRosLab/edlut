/***************************************************************************
 *                           LIFTimeDrivenModel_1_2_GPU.cu                 *
 *                           -------------------                           *
 * copyright            : (C) 2012 by Francisco Naveros                    *
 * email                : fnaveros@atc.ugr.es                              *
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 3 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

#include "../../include/neuron_model/LIFTimeDrivenModel_1_2_GPU.h"
#include "../../include/neuron_model/LIFTimeDrivenModel_1_2_GPU2.h"
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

#include "../../include/cudaError.h"
//Library for CUDA
#include <cutil_inline.h>

void LIFTimeDrivenModel_1_2_GPU::LoadNeuronModel(string ConfigFile) throw (EDLUTFileException){
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

											this->InitialState = (VectorNeuronState_GPU *) new VectorNeuronState_GPU(3);
//NEW CODE------------------------------------------------------------------------------
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
//----------------------------------------------------------------------	

  		//INTEGRATION METHOD
		this->integrationMethod_GPU=LoadIntegrationMethod_GPU::loadIntegrationMethod_GPU(fh, &Currentline, N_NeuronStateVariables, N_DifferentialNeuronState, N_TimeDependentNeuronState);

		//TIME DRIVEN STEP
		this->TimeDrivenStep_GPU = LoadTimeEvent_GPU::loadTimeEvent_GPU(fh, &Currentline);

	}
}

void LIFTimeDrivenModel_1_2_GPU::SynapsisEffect(int index, VectorNeuronState_GPU * state, Interconnection * InputConnection){

	switch (InputConnection->GetType()){
		case 0: {
			//gampa
			state->AuxStateCPU[0*state->GetSizeState() + index]+=1e-9*InputConnection->GetWeight();
			break;
		}case 1:{
			//gnmda
			state->AuxStateCPU[1*state->GetSizeState() + index]+=1e-9*InputConnection->GetWeight();
			break;
		}
	}
}

LIFTimeDrivenModel_1_2_GPU::LIFTimeDrivenModel_1_2_GPU(string NeuronTypeID, string NeuronModelID): TimeDrivenNeuronModel_GPU(NeuronTypeID, NeuronModelID), eexc(0), einh(0), erest(0), vthr(0), cm(0), texc(0), tinh(0),
		tref(0), grest(0){

}

LIFTimeDrivenModel_1_2_GPU::~LIFTimeDrivenModel_1_2_GPU(void){
	DeleteClassGPU();
}

void LIFTimeDrivenModel_1_2_GPU::LoadNeuronModel() throw (EDLUTFileException){
	this->LoadNeuronModel(this->GetModelID()+".cfg");
}

VectorNeuronState * LIFTimeDrivenModel_1_2_GPU::InitializeState(){
	return this->GetVectorNeuronState();
}


InternalSpike * LIFTimeDrivenModel_1_2_GPU::ProcessInputSpike(PropagatedSpike *  InputSpike){
	Interconnection * inter = InputSpike->GetSource()->GetOutputConnectionAt(InputSpike->GetTarget());

	Neuron * TargetCell = inter->GetTarget();

	int indexGPU =TargetCell->GetIndex_VectorNeuronState();

	VectorNeuronState_GPU * state = (VectorNeuronState_GPU *) this->InitialState;

	// Add the effect of the input spike
	this->SynapsisEffect(inter->GetTarget()->GetIndex_VectorNeuronState(), state, inter);

	return 0;
}


__global__ void LIFTimeDrivenModel_1_2_GPU_UpdateState(TimeDrivenNeuronModel_GPU2 ** timeDrivenNeuronModel_GPU2, float * AuxStateGPU, float * StateGPU, double * LastUpdateGPU, double * LastSpikeTimeGPU, bool * InternalSpikeGPU, int SizeStates, double CurrentTime){
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	while (index<SizeStates){
		(*timeDrivenNeuronModel_GPU2)->UpdateState(index, AuxStateGPU, StateGPU, LastUpdateGPU, LastSpikeTimeGPU, InternalSpikeGPU, SizeStates, CurrentTime);
		index+=blockDim.x*gridDim.x;
	}
}
		
bool LIFTimeDrivenModel_1_2_GPU::UpdateState(int index, VectorNeuronState * State, double CurrentTime){
	
	VectorNeuronState_GPU *state = (VectorNeuronState_GPU *) State;

	//----------------------------------------------
	cudaDeviceProp prop;
	HANDLE_ERROR(cudaGetDeviceProperties( &prop, 0 ));
	if(prop.canMapHostMemory && true){
		LIFTimeDrivenModel_1_2_GPU_UpdateState<<<N_block,N_thread>>>(timeDrivenNeuronModel_GPU2, state->AuxStateGPU, state->VectorNeuronStates_GPU, state->LastUpdateGPU, state->LastSpikeTimeGPU, state->InternalSpikeGPU, state->SizeStates, CurrentTime);
	}else{
		HANDLE_ERROR(cudaMemcpy(state->AuxStateGPU,state->AuxStateCPU,4*state->SizeStates*sizeof(float),cudaMemcpyHostToDevice));
		LIFTimeDrivenModel_1_2_GPU_UpdateState<<<N_block,N_thread>>>(timeDrivenNeuronModel_GPU2, state->AuxStateGPU, state->VectorNeuronStates_GPU, state->LastUpdateGPU, state->LastSpikeTimeGPU, state->InternalSpikeGPU, state->SizeStates, CurrentTime);
		HANDLE_ERROR(cudaMemcpy(state->InternalSpikeCPU,state->InternalSpikeGPU,state->SizeStates*sizeof(bool),cudaMemcpyDeviceToHost));
	}


	if(this->GetVectorNeuronState()->Get_Is_Monitored()){
		HANDLE_ERROR(cudaMemcpy(state->VectorNeuronStates,state->VectorNeuronStates_GPU,state->GetNumberOfVariables()*state->SizeStates*sizeof(float),cudaMemcpyDeviceToHost));
		HANDLE_ERROR(cudaMemcpy(state->LastUpdate,state->LastUpdateGPU,state->SizeStates*sizeof(double),cudaMemcpyDeviceToHost));
		HANDLE_ERROR(cudaMemcpy(state->LastSpikeTime,state->LastSpikeTimeGPU,state->SizeStates*sizeof(double),cudaMemcpyDeviceToHost));
	}
 

	HANDLE_ERROR(cudaEventRecord(stop, 0)); 
	HANDLE_ERROR(cudaEventSynchronize(stop));


	memset(state->AuxStateCPU,0,N_TimeDependentNeuronState*state->SizeStates*sizeof(float));

	return false;

}

ostream & LIFTimeDrivenModel_1_2_GPU::PrintInfo(ostream & out){
	out << "- Leaky Time-Driven Model 1_2: " << this->GetModelID() << endl;

	out << "\tExc. Reversal Potential: " << this->eexc << "V\tInh. Reversal Potential: " << this->einh << "V\tResting potential: " << this->erest << "V" << endl;

	out << "\tFiring threshold: " << this->vthr << "V\tMembrane capacitance: " << this->cm << "nS\tExcitatory Time Constant: " << this->texc << "s" << endl;

	out << "\tInhibitory time constant: " << this->tinh << "s\tRefractory Period: " << this->tref << "s\tResting Conductance: " << this->grest << "nS" << endl;

	return out;
}	


void LIFTimeDrivenModel_1_2_GPU::InitializeStates(int N_neurons){

	VectorNeuronState_GPU * state = (VectorNeuronState_GPU *) this->InitialState;
	
	float initialization[] = {erest,0.0,0.0};
	state->InitializeStatesGPU(N_neurons, initialization, N_TimeDependentNeuronState);

	//INITIALIZE CLASS IN GPU
	this->InitializeClassGPU(N_neurons);
}




__global__ void LIFTimeDrivenModel_1_2_GPU_InitializeClassGPU(TimeDrivenNeuronModel_GPU2 ** timeDrivenNeuronModel_GPU2,
		float eexc, float einh, float erest, float vthr, float cm, float texc, float tinh, float tref, float grest, 
		char const* integrationName, int N_neurons, int Total_N_thread, void ** Buffer_GPU){
	if(blockIdx.x==0 && threadIdx.x==0){
		(*timeDrivenNeuronModel_GPU2) = (LIFTimeDrivenModel_1_2_GPU2 *) new LIFTimeDrivenModel_1_2_GPU2(eexc, einh, erest, 
			vthr, cm, texc, tinh, tref, grest, integrationName, N_neurons, Total_N_thread, Buffer_GPU);
	}
}

void LIFTimeDrivenModel_1_2_GPU::InitializeClassGPU(int N_neurons){
	cudaMalloc(&timeDrivenNeuronModel_GPU2, sizeof(TimeDrivenNeuronModel_GPU2 **));
	
	char * integrationNameGPU;
	cudaMalloc((void **)&integrationNameGPU,32*4);
	HANDLE_ERROR(cudaMemcpy(integrationNameGPU,integrationMethod_GPU->GetType(),32*4,cudaMemcpyHostToDevice));

	cudaDeviceProp prop;
	HANDLE_ERROR(cudaGetDeviceProperties( &prop, 0 ));	
	this->N_thread = 128;
	this->N_block=prop.multiProcessorCount*16;
	if((N_neurons+N_thread-1)/N_thread < N_block){
		N_block = (N_neurons+N_thread-1)/N_thread;
	}
	int Total_N_thread=N_thread*N_block;

	integrationMethod_GPU->InitializeMemoryGPU(N_neurons, Total_N_thread);

	LIFTimeDrivenModel_1_2_GPU_InitializeClassGPU<<<1,1>>>(timeDrivenNeuronModel_GPU2,eexc, einh, erest, vthr, 
		cm, texc, tinh, tref, grest, integrationNameGPU, N_neurons,Total_N_thread, integrationMethod_GPU->Buffer_GPU);

	cudaFree(integrationNameGPU);
}





__global__ void LIFTimeDrivenModel_1_2_GPU_DeleteClassGPU(TimeDrivenNeuronModel_GPU2 ** timeDrivenNeuronModel_GPU2){
	if(blockIdx.x==0 && threadIdx.x==0){
		delete (*timeDrivenNeuronModel_GPU2); 
	}
}


void LIFTimeDrivenModel_1_2_GPU::DeleteClassGPU(){
    LIFTimeDrivenModel_1_2_GPU_DeleteClassGPU<<<1,1>>>(timeDrivenNeuronModel_GPU2);
    cudaFree(timeDrivenNeuronModel_GPU2);
}

