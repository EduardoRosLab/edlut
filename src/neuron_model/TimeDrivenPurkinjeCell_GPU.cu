/***************************************************************************
 *                           TimeDrivenPurkinjeCell_GPU.cu                 *
 *                           -------------------                           *
 * copyright            : (C) 2015 by Richard Carrill, Niceto Luque and    *
						  Francisco Naveros								   *
 * email                : rcarrillo@ugr.es, nluque@ugr.es and			   *
						  fnaveros@atc.ugr.es							   *
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 3 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

#include "../../include/neuron_model/TimeDrivenPurkinjeCell_GPU.h"
#include "../../include/neuron_model/TimeDrivenPurkinjeCell_GPU2.h"
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

#include "../../include/openmp/openmp.h"

#include "../../include/cudaError.h"
//Library for CUDA
#include <helper_cuda.h>

void TimeDrivenPurkinjeCell_GPU::LoadNeuronModel(string ConfigFile) throw (EDLUTFileException){
	FILE *fh;
	long Currentline = 0L;
	fh=fopen(ConfigFile.c_str(),"rt");
	if(fh){

		this->State = (VectorNeuronState_GPU *) new VectorNeuronState_GPU(N_NeuronStateVariables);

  		//INTEGRATION METHOD
		this->integrationMethod_GPU=LoadIntegrationMethod_GPU::loadIntegrationMethod_GPU(fh, &Currentline, N_NeuronStateVariables, N_DifferentialNeuronState, N_TimeDependentNeuronState);

		//TIME DRIVEN STEP
		this->TimeDrivenStep_GPU = LoadTimeEvent_GPU::loadTimeEvent_GPU(fh, &Currentline);

	}
}

void TimeDrivenPurkinjeCell_GPU::SynapsisEffect(int index, VectorNeuronState_GPU * state, Interconnection * InputConnection){
	state->AuxStateCPU[InputConnection->GetType()*state->GetSizeState() + index]+=1e-6f*InputConnection->GetWeight();
}

TimeDrivenPurkinjeCell_GPU::TimeDrivenPurkinjeCell_GPU(string NeuronTypeID, string NeuronModelID): TimeDrivenNeuronModel_GPU(NeuronTypeID, NeuronModelID), g_L(0.02f),
		g_Ca(0.001f), g_M(0.75f), Cylinder_length_of_the_soma(0.0015f), Radius_of_the_soma(0.0008f), Area(3.141592f*0.0015f*2.0f*0.0008f),
		inv_Area(1.0f/(3.141592f*0.0015f*2.0f*0.0008f)), Membrane_capacitance(1.0f), inv_Membrane_capacitance(1.0f/1.0f){

	eexc=0.0f;
	einh=-80.0f ;
	vthr= -35.0f;
	erest=-65.0f;
	texc=1.0f;
	inv_texc=1.0f/texc;
	tinh=2;
	inv_tinh=1.0f/tinh;
	tref=1.35f;
	tref_0_5=tref*0.5f;
	inv_tref_0_5=1.0f/tref_0_5;
	spkpeak=31.0f;
}

TimeDrivenPurkinjeCell_GPU::~TimeDrivenPurkinjeCell_GPU(void){
	DeleteClassGPU2();
}

void TimeDrivenPurkinjeCell_GPU::LoadNeuronModel() throw (EDLUTFileException){
	this->LoadNeuronModel(this->GetModelID()+".cfg");
}

VectorNeuronState * TimeDrivenPurkinjeCell_GPU::InitializeState(){
	return this->GetVectorNeuronState();
}


InternalSpike * TimeDrivenPurkinjeCell_GPU::ProcessInputSpike(Interconnection * inter, Neuron * target, double time){
	int indexGPU =target->GetIndex_VectorNeuronState();

	VectorNeuronState_GPU * state = (VectorNeuronState_GPU *) this->State;

	// Add the effect of the input spike
	this->SynapsisEffect(target->GetIndex_VectorNeuronState(), state, inter);

	return 0;
}


__global__ void TimeDrivenPurkinjeCell_GPU_UpdateState(TimeDrivenPurkinjeCell_GPU2 ** NeuronModel_GPU2, double CurrentTime){
	(*NeuronModel_GPU2)->UpdateState(CurrentTime);
}

		
bool TimeDrivenPurkinjeCell_GPU::UpdateState(int index, VectorNeuronState * State, double CurrentTime){
	VectorNeuronState_GPU *state = (VectorNeuronState_GPU *) State;

	//----------------------------------------------
	if(prop.canMapHostMemory){
		TimeDrivenPurkinjeCell_GPU_UpdateState<<<N_block,N_thread>>>(NeuronModel_GPU2, CurrentTime);
	}else{
		HANDLE_ERROR(cudaMemcpy(state->AuxStateGPU,state->AuxStateCPU,4*state->SizeStates*sizeof(float),cudaMemcpyHostToDevice));
		TimeDrivenPurkinjeCell_GPU_UpdateState<<<N_block,N_thread>>>(NeuronModel_GPU2, CurrentTime);
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

ostream & TimeDrivenPurkinjeCell_GPU::PrintInfo(ostream & out){
	//out << "- Leaky Time-Driven Model 1_2: " << this->GetModelID() << endl;

	//out << "\tExc. Reversal Potential: " << this->eexc << "V\tInh. Reversal Potential: " << this->einh << "V\tResting potential: " << this->erest << "V" << endl;

	//out << "\tFiring threshold: " << this->vthr << "V\tMembrane capacitance: " << this->cm << "nS\tExcitatory Time Constant: " << this->texc << "s" << endl;

	//out << "\tInhibitory time constant: " << this->tinh << "s\tRefractory Period: " << this->tref << "s\tResting Conductance: " << this->grest << "nS" << endl;

	return out;
}	


void TimeDrivenPurkinjeCell_GPU::InitializeStates(int N_neurons, int OpenMPQueueIndex){

	//Select the correnpondent device. 
	HANDLE_ERROR(cudaSetDevice(GPUsIndex[OpenMPQueueIndex % NumberOfGPUs]));  
	HANDLE_ERROR(cudaEventCreate(&stop));
	HANDLE_ERROR(cudaGetDeviceProperties( &prop, GPUsIndex[OpenMPQueueIndex % NumberOfGPUs]));


	VectorNeuronState_GPU * state = (VectorNeuronState_GPU *) this->State;


	float alpha_ca=1.6f/(1+exp(-0.072f*(erest-5.0f)));

	float beta_ca=(0.02f*(erest+8.9f))/(exp((erest+8.9f)*0.2f)-1.0f);
	float inv_tau_ca=alpha_ca+beta_ca;

	float alpha_M=0.3f/(1+exp((-erest-2.0f)*0.2f));

	float beta_M=0.001f*exp((-erest-60.0f)*0.055555555555555f);
	float inv_tau_M=alpha_M+beta_M;

	//c_inf
	float c_inf=alpha_ca/inv_tau_ca;

	//M_inf
	float M_inf=alpha_M/inv_tau_M;

	float initialization[] = {erest,c_inf,M_inf,0.0f,0.0f};

	state->InitializeStatesGPU(N_neurons, initialization, N_TimeDependentNeuronState, prop);

	//INITIALIZE CLASS IN GPU
	this->InitializeClassGPU2(N_neurons);


	InitializeVectorNeuronState_GPU2();
}



__global__ void TimeDrivenPurkinjeCell_GPU_InitializeClassGPU2(TimeDrivenPurkinjeCell_GPU2 ** NeuronModel_GPU2, double new_elapsed_time,
	float new_g_L, float new_g_Ca, float new_g_M, float new_Cylinder_length_of_the_soma, float new_Radius_of_the_soma, float new_Area,
	float new_inv_Area, float new_Membrane_capacitance, float new_inv_Membrane_capacitance, float new_eexc,	float new_einh, 
	float new_vthr, float new_erest, float new_texc, float new_inv_texc, float new_tinh, float new_inv_tinh, float new_tref, 
	float new_tref_0_5, float new_inv_tref_0_5, float new_spkpeak, char const* integrationName, int N_neurons, void ** Buffer_GPU)
{
	if(blockIdx.x==0 && threadIdx.x==0){
		(*NeuronModel_GPU2)=new TimeDrivenPurkinjeCell_GPU2(new_elapsed_time, new_g_L, new_g_Ca, new_g_M, new_Cylinder_length_of_the_soma, 
			new_Radius_of_the_soma, new_Area, new_inv_Area, new_Membrane_capacitance, new_inv_Membrane_capacitance, new_eexc, new_einh,	
			new_vthr, new_erest, new_texc, new_inv_texc, new_tinh, new_inv_tinh, new_tref, new_tref_0_5, new_inv_tref_0_5, new_spkpeak, 
			integrationName, N_neurons, Buffer_GPU);
	}
}

void TimeDrivenPurkinjeCell_GPU::InitializeClassGPU2(int N_neurons){
	cudaMalloc(&NeuronModel_GPU2, sizeof(TimeDrivenPurkinjeCell_GPU2 **));
	
	char * integrationNameGPU;
	cudaMalloc((void **)&integrationNameGPU,32*4);
	HANDLE_ERROR(cudaMemcpy(integrationNameGPU,integrationMethod_GPU->GetType(),32*4,cudaMemcpyHostToDevice));

	this->N_thread = 128;
	this->N_block=prop.multiProcessorCount*16;
	if((N_neurons+N_thread-1)/N_thread < N_block){
		N_block = (N_neurons+N_thread-1)/N_thread;
	}
	int Total_N_thread=N_thread*N_block;

	integrationMethod_GPU->InitializeMemoryGPU(N_neurons, Total_N_thread);


	TimeDrivenPurkinjeCell_GPU_InitializeClassGPU2<<<1,1>>>(NeuronModel_GPU2,TimeDrivenStep_GPU, g_L, g_Ca, g_M,Cylinder_length_of_the_soma, 
		Radius_of_the_soma, Area, inv_Area, Membrane_capacitance, inv_Membrane_capacitance, eexc, einh,	vthr, erest, texc, inv_texc, tinh, 
		inv_tinh, tref, tref_0_5, inv_tref_0_5, spkpeak, integrationNameGPU, N_neurons, integrationMethod_GPU->Buffer_GPU);

	cudaFree(integrationNameGPU);
}



__global__ void initializeVectorNeuronState_GPU2(TimeDrivenPurkinjeCell_GPU2 ** NeuronModel_GPU2, float * AuxStateGPU, float * StateGPU, double * LastUpdateGPU, double * LastSpikeTimeGPU, bool * InternalSpikeGPU, int SizeStates){
	if(blockIdx.x==0 && threadIdx.x==0){
		(*NeuronModel_GPU2)->InitializeVectorNeuronState_GPU2(AuxStateGPU, StateGPU, LastUpdateGPU, LastSpikeTimeGPU, InternalSpikeGPU, SizeStates);
	}
}

void TimeDrivenPurkinjeCell_GPU::InitializeVectorNeuronState_GPU2(){
	VectorNeuronState_GPU *state = (VectorNeuronState_GPU *) State;
	initializeVectorNeuronState_GPU2<<<1,1>>>(NeuronModel_GPU2, state->AuxStateGPU, state->VectorNeuronStates_GPU, state->LastUpdateGPU, state->LastSpikeTimeGPU, state->InternalSpikeGPU, state->SizeStates);
}


__global__ void DeleteClass_GPU2(TimeDrivenPurkinjeCell_GPU2 ** NeuronModel_GPU2){
	if(blockIdx.x==0 && threadIdx.x==0){
		delete (*NeuronModel_GPU2); 
	}
}


void TimeDrivenPurkinjeCell_GPU::DeleteClassGPU2(){
    DeleteClass_GPU2<<<1,1>>>(NeuronModel_GPU2);
    cudaFree(NeuronModel_GPU2);
}


int TimeDrivenPurkinjeCell_GPU::CheckSynapseTypeNumber(int Type){
	if(Type<N_TimeDependentNeuronState && Type>=0){
		return Type;
	}else{
		cout<<"Neuron model "<<this->GetTypeID()<<", "<<this->GetModelID()<<" does not support input synapses of type "<<Type<<endl;
		return 0;
	}
}