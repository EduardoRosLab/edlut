/***************************************************************************
 *                           EgidioGranuleCell_TimeDriven_GPU_C_INTERFACE.cu *
 *                           -------------------                           *
 * copyright            : (C) 2013 by Francisco Naveros                    *
 * email                : fnaveros@ugr.es                                  *
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 3 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

#include "../../include/neuron_model/EgidioGranuleCell_TimeDriven_GPU_C_INTERFACE.cuh"
#include "../../include/neuron_model/EgidioGranuleCell_TimeDriven_GPU2.cuh"
#include "../../include/neuron_model/VectorNeuronState.h"
#include "../../include/neuron_model/VectorNeuronState_GPU_C_INTERFACE.cuh"
#include "../../include/neuron_model/CurrentSynapseModel.h"

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
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "integration_method/IntegrationMethodFactory_GPU_C_INTERFACE.cuh"

void EgidioGranuleCell_TimeDriven_GPU_C_INTERFACE::InitializeCurrentSynapsis(int N_neurons){
	this->CurrentSynapsis = new CurrentSynapseModel(N_neurons);
}

EgidioGranuleCell_TimeDriven_GPU_C_INTERFACE::EgidioGranuleCell_TimeDriven_GPU_C_INTERFACE() : TimeDrivenNeuronModel_GPU_C_INTERFACE(MilisecondScale), gMAXNa_f(0.013f), gMAXNa_r(0.0005f), gMAXNa_p(0.0002f), gMAXK_V(0.003f), gMAXK_A(0.004f), gMAXK_IR(0.0009f), gMAXK_Ca(0.004f),
		gMAXCa(0.00046f), gMAXK_sl(0.00035f), gLkg1(5.68e-5f), gLkg2(2.17e-5f), VNa(87.39f), VK(-84.69f), VLkg1(-58.0f), VLkg2(-65.0f), V0_xK_Ai(-46.7f),
		K_xK_Ai(-19.8f), V0_yK_Ai(-78.8f), K_yK_Ai(8.4f), V0_xK_sli(-30.0f), B_xK_sli(6.0f), F(96485.309f), A(1e-04f), d(0.2f), betaCa(1.5f),
		Ca0(1e-04f), R(8.3134f), cao(2.0f), Cm(1.0e-3f), temper(30.0f), Q10_20 ( pow(3,((temper-20.0f)/10.0f))), Q10_22 ( pow(3,((temper-22.0f)/10.0f))),
		Q10_30 ( pow(3,((temper-30.0f)/10.0f))), Q10_6_3 ( pow(3,((temper-6.3f)/10.0f))), e_exc(0.0f), e_inh(-80.0f), tau_exc(0.5f), tau_inh(10.0f), tau_nmda(15.0f),
		v_thr(0.0f), NeuronModel_GPU2(0), EXC(false), INH(false), NMDA(false), EXT_I(false){
	std::map<std::string, boost::any> param_map = EgidioGranuleCell_TimeDriven_GPU_C_INTERFACE::GetDefaultParameters();
	param_map["name"] = EgidioGranuleCell_TimeDriven_GPU_C_INTERFACE::GetName();
	this->SetParameters(param_map);

	this->State = (VectorNeuronState_GPU_C_INTERFACE *) new VectorNeuronState_GPU_C_INTERFACE(N_NeuronStateVariables);
}

EgidioGranuleCell_TimeDriven_GPU_C_INTERFACE::~EgidioGranuleCell_TimeDriven_GPU_C_INTERFACE(void){
	DeleteClassGPU2();
}

VectorNeuronState * EgidioGranuleCell_TimeDriven_GPU_C_INTERFACE::InitializeState(){
	return this->GetVectorNeuronState();
}


InternalSpike * EgidioGranuleCell_TimeDriven_GPU_C_INTERFACE::ProcessInputSpike(Interconnection * inter, double time){
	this->State_GPU->AuxStateCPU[inter->GetType()*State_GPU->GetSizeState() + inter->GetTargetNeuronModelIndex()] += inter->GetWeight();

	return 0;
}

void EgidioGranuleCell_TimeDriven_GPU_C_INTERFACE::ProcessInputCurrent(Interconnection * inter, Neuron * target, float current){
	//Update the external current in the corresponding input synapse of type EXT_I (defined in pA).
	this->CurrentSynapsis->SetInputCurrent(target->GetIndex_VectorNeuronState(), inter->GetSubindexType(), current);

	//Update the total external current that receive the neuron coming from all its EXT_I synapsis (defined in pA).
	float total_ext_I = this->CurrentSynapsis->GetTotalCurrent(target->GetIndex_VectorNeuronState());
	this->State_GPU->AuxStateCPU[inter->GetType()*State_GPU->GetSizeState() + inter->GetTargetNeuronModelIndex()] = total_ext_I;
}

__global__ void EgidioGranuleCell_TimeDriven_GPU_C_INTERFACE_UpdateState(EgidioGranuleCell_TimeDriven_GPU2 ** NeuronModel_GPU2, double CurrentTime){
	(*NeuronModel_GPU2)->UpdateState(CurrentTime);
}

bool EgidioGranuleCell_TimeDriven_GPU_C_INTERFACE::UpdateState(int index, double CurrentTime){
	if(prop.canMapHostMemory){
		EgidioGranuleCell_TimeDriven_GPU_C_INTERFACE_UpdateState<<<N_block,N_thread>>>(NeuronModel_GPU2, CurrentTime);
	}else{
		HANDLE_ERROR(cudaMemcpy(State_GPU->AuxStateGPU,State_GPU->AuxStateCPU,this->N_TimeDependentNeuronState*State_GPU->SizeStates*sizeof(float),cudaMemcpyHostToDevice));
		EgidioGranuleCell_TimeDriven_GPU_C_INTERFACE_UpdateState<<<N_block,N_thread>>>(NeuronModel_GPU2, CurrentTime);
		HANDLE_ERROR(cudaMemcpy(State_GPU->InternalSpikeCPU,State_GPU->InternalSpikeGPU,State_GPU->SizeStates*sizeof(bool),cudaMemcpyDeviceToHost));
	}

	if(this->GetVectorNeuronState()->Get_Is_Monitored()){
		HANDLE_ERROR(cudaMemcpy(State_GPU->VectorNeuronStates,State_GPU->VectorNeuronStates_GPU,State_GPU->GetNumberOfVariables()*State_GPU->SizeStates*sizeof(float),cudaMemcpyDeviceToHost));
		HANDLE_ERROR(cudaMemcpy(State_GPU->LastUpdate,State_GPU->LastUpdateGPU,State_GPU->SizeStates*sizeof(double),cudaMemcpyDeviceToHost));
		HANDLE_ERROR(cudaMemcpy(State_GPU->LastSpikeTime,State_GPU->LastSpikeTimeGPU,State_GPU->SizeStates*sizeof(double),cudaMemcpyDeviceToHost));
	}

	HANDLE_ERROR(cudaEventRecord(stop, 0));
	HANDLE_ERROR(cudaEventSynchronize(stop));

	//The external current value it is not reset (N_TimeDependentNeuronState-1)
	memset(State_GPU->AuxStateCPU,0,(N_TimeDependentNeuronState-1)*State_GPU->SizeStates*sizeof(float));

	return false;
}


enum NeuronModelOutputActivityType EgidioGranuleCell_TimeDriven_GPU_C_INTERFACE::GetModelOutputActivityType(){
	return OUTPUT_SPIKE;
}

enum NeuronModelInputActivityType EgidioGranuleCell_TimeDriven_GPU_C_INTERFACE::GetModelInputActivityType(){
	return INPUT_SPIKE_AND_CURRENT;
}

ostream & EgidioGranuleCell_TimeDriven_GPU_C_INTERFACE::PrintInfo(ostream & out){
	out << "- EgidioGranuleCell Time-Driven Model GPU: " << EgidioGranuleCell_TimeDriven_GPU_C_INTERFACE::GetName() << endl;
	out << "\tExcitatory reversal potential (e_exc): " << this->e_exc << "mV" << endl;
	out << "\tInhibitory reversal potential (e_inh): " << this->e_inh << "mV" << endl;
	out << "\tAMPA (excitatory) receptor time constant (tau_exc): " << this->tau_exc << "ms" << endl;
	out << "\tGABA (inhibitory) receptor time constant (tau_inh): " << this->tau_inh << "ms" << endl;
	out << "\tNMDA (excitatory) receptor time constant (tau_nmda): " << this->tau_nmda << "ms" << endl;
	out << "\tEffective threshold potential (v_thr): " << this->v_thr << "mV" << endl;

	this->integration_method_GPU->PrintInfo(out);
	return out;
}


void EgidioGranuleCell_TimeDriven_GPU_C_INTERFACE::InitializeStates(int N_neurons, int OpenMPQueueIndex){

	//Select the correnpondent device.
	this->GPU_index = OpenMPQueueIndex % NumberOfGPUs;
	HANDLE_ERROR(cudaSetDevice(GPUsIndex[GPU_index]));
	HANDLE_ERROR(cudaEventCreate(&stop));
	HANDLE_ERROR(cudaGetDeviceProperties( &prop, GPUsIndex[OpenMPQueueIndex % NumberOfGPUs]));

	this->State_GPU = (VectorNeuronState_GPU_C_INTERFACE *) this->State;

	//Initial State
	float V=-80.0f;
	float xNa_f=0.00047309535f;
	float yNa_f=1.0f;
	float xNa_r=0.00013423511f;
	float yNa_r=0.96227829f;
	float xNa_p=0.00050020111f;
	float xK_V=0.010183001f;
	float xK_A=0.15685486f;
	float yK_A=0.53565367f;
	float xK_IR=0.37337035f;
	float xK_Ca=0.00012384122f;
	float xCa=0.0021951104f;
	float yCa=0.89509747f;
	float xK_sl=0.00024031171f;
	float Ca=Ca0;
	float gexc=0.0f;
	float ginh=0.0f;
	float gnmda=0.0f;
	float External_current = 0.0f;

	////Initial State
	//float V = -59.959327697754f;
	//float xNa_f = 0.007151846774f;
	//float yNa_f = 0.999994099140f;
	//float xNa_r = 0.000353821204f;
	//float yNa_r = 0.841639518738f;
	//float xNa_p = 0.026804497465f;
	//float xK_V = 0.060160592198f;
	//float xK_A = 0.338571757078f;
	//float yK_A = 0.096005745232f;
	//float xK_IR = 0.130047351122f;
	//float xK_Ca = 0.000681858859f;
	//float xCa = 0.016704728827f;
	//float yCa = 0.692971527576f;
	//float xK_sl = 0.006732143927f;
	//float Ca = Ca0;
	//float gexc = 0.0f;
	//float ginh = 0.0f;
	//float gnmda = 0.0f;
	//float External_current = 0.0f;


	//Initialize neural state variables.
	float initialization[] = {V,xNa_f,yNa_f,xNa_r,yNa_r,xNa_p,xK_V,xK_A,yK_A,xK_IR,xK_Ca,xCa,yCa,xK_sl,Ca,gexc,ginh, gnmda,External_current};
	State_GPU->InitializeStatesGPU(N_neurons, initialization, N_TimeDependentNeuronState, prop);

	//INITIALIZE CLASS IN GPU
	this->InitializeClassGPU2(N_neurons);

	InitializeVectorNeuronState_GPU2();

	//Initialize the array that stores the number of input current synapses for each neuron in the model
	InitializeCurrentSynapsis(N_neurons);
}


__global__ void EgidioGranuleCell_TimeDriven_GPU_C_INTERFACE_InitializeClassGPU2(EgidioGranuleCell_TimeDriven_GPU2 ** NeuronModel_GPU2,
		float gMAXNa_f, float gMAXNa_r, float gMAXNa_p, float gMAXK_V,
		float gMAXK_A,float gMAXK_IR,float gMAXK_Ca,float gMAXCa,float gMAXK_sl, char const* integrationName, int N_neurons, void ** Buffer_GPU){
	if(blockIdx.x==0 && threadIdx.x==0){
		(*NeuronModel_GPU2) = new EgidioGranuleCell_TimeDriven_GPU2(gMAXNa_f, gMAXNa_r, gMAXNa_p, gMAXK_V,
			gMAXK_A,gMAXK_IR,gMAXK_Ca,gMAXCa,gMAXK_sl,integrationName, N_neurons, Buffer_GPU);
	}
}


void EgidioGranuleCell_TimeDriven_GPU_C_INTERFACE::InitializeClassGPU2(int N_neurons){
	cudaMalloc(&NeuronModel_GPU2, sizeof(EgidioGranuleCell_TimeDriven_GPU2 **));

	char * integrationNameGPU;
	cudaMalloc((void **)&integrationNameGPU,32*4);
//REVISAR
	HANDLE_ERROR(cudaMemcpy(integrationNameGPU, &integration_method_GPU->name[0], 32 * 4, cudaMemcpyHostToDevice));

	this->N_thread = 128;
	this->N_block=prop.multiProcessorCount*4;
	if((N_neurons+N_thread-1)/N_thread < N_block){
		N_block = (N_neurons+N_thread-1)/N_thread;
	}
	int Total_N_thread=N_thread*N_block;

	integration_method_GPU->InitializeMemoryGPU(N_neurons, Total_N_thread);


	EgidioGranuleCell_TimeDriven_GPU_C_INTERFACE_InitializeClassGPU2<<<1,1>>>(NeuronModel_GPU2, gMAXNa_f, gMAXNa_r, gMAXNa_p, gMAXK_V,
			gMAXK_A,gMAXK_IR,gMAXK_Ca,gMAXCa,gMAXK_sl,integrationNameGPU, N_neurons, integration_method_GPU->Buffer_GPU);

	cudaFree(integrationNameGPU);
}


__global__ void initializeVectorNeuronState_GPU2(EgidioGranuleCell_TimeDriven_GPU2 ** NeuronModel_GPU2, int NumberOfVariables, float * InitialStateGPU, float * AuxStateGPU, float * StateGPU, double * LastUpdateGPU, double * LastSpikeTimeGPU, bool * InternalSpikeGPU, int SizeStates){
	if(blockIdx.x==0 && threadIdx.x==0){
		(*NeuronModel_GPU2)->InitializeVectorNeuronState_GPU2(NumberOfVariables, InitialStateGPU, AuxStateGPU, StateGPU, LastUpdateGPU, LastSpikeTimeGPU, InternalSpikeGPU, SizeStates);
	}
}


void EgidioGranuleCell_TimeDriven_GPU_C_INTERFACE::InitializeVectorNeuronState_GPU2(){
	VectorNeuronState_GPU_C_INTERFACE *state = (VectorNeuronState_GPU_C_INTERFACE *) State;
	initializeVectorNeuronState_GPU2<<<1,1>>>(NeuronModel_GPU2, state->NumberOfVariables, state->InitialStateGPU, state->AuxStateGPU, state->VectorNeuronStates_GPU, state->LastUpdateGPU, state->LastSpikeTimeGPU, state->InternalSpikeGPU, state->SizeStates);
}


__global__ void DeleteClass_GPU2(EgidioGranuleCell_TimeDriven_GPU2 ** NeuronModel_GPU2){
	if(blockIdx.x==0 && threadIdx.x==0){
		delete (*NeuronModel_GPU2);
	}
}


void EgidioGranuleCell_TimeDriven_GPU_C_INTERFACE::DeleteClassGPU2(){
	if (NeuronModel_GPU2 != 0){
		DeleteClass_GPU2 << <1, 1 >> >(NeuronModel_GPU2);
		cudaFree(NeuronModel_GPU2);
	}
}


__global__ void SetEnabledSynapsis_GPU2(EgidioGranuleCell_TimeDriven_GPU2 ** NeuronModel_GPU2, bool new_EXC, bool new_INH, bool new_NMDA, bool new_EXT_I){
	if (blockIdx.x == 0 && threadIdx.x == 0){
		(*NeuronModel_GPU2)->SetEnabledSynapsis(new_EXC, new_INH, new_NMDA, new_EXT_I);
	}
}


bool EgidioGranuleCell_TimeDriven_GPU_C_INTERFACE::CheckSynapseType(Interconnection * connection){
	int Type = connection->GetType();
	if (Type<N_TimeDependentNeuronState && Type >= 0){
		//activaty synapse type
		if (Type == 0 && EXC == false){
			EXC = true;
			HANDLE_ERROR(cudaSetDevice(GPUsIndex[GPU_index]));
			SetEnabledSynapsis_GPU2 << <1, 1 >> >(NeuronModel_GPU2, EXC, INH, NMDA, EXT_I);
		}
		if (Type == 1 && INH == false){
			INH = true;
			HANDLE_ERROR(cudaSetDevice(GPUsIndex[GPU_index]));
			SetEnabledSynapsis_GPU2 << <1, 1 >> >(NeuronModel_GPU2, EXC, INH, NMDA, EXT_I);
		}
		if (Type == 2 && NMDA == false){
			NMDA = true;
			HANDLE_ERROR(cudaSetDevice(GPUsIndex[GPU_index]));
			SetEnabledSynapsis_GPU2 << <1, 1 >> >(NeuronModel_GPU2, EXC, INH, NMDA, EXT_I);
		}
		if (Type == 3 && EXT_I == false){
			EXT_I = true;
			HANDLE_ERROR(cudaSetDevice(GPUsIndex[GPU_index]));
			SetEnabledSynapsis_GPU2 << <1, 1 >> >(NeuronModel_GPU2, EXC, INH, NMDA, EXT_I);
		}

		NeuronModel * model = connection->GetSource()->GetNeuronModel();
		//Synapse types that process input spikes
		if (Type < N_TimeDependentNeuronState - 1){
			if (model->GetModelOutputActivityType() == OUTPUT_SPIKE){
				return true;
			}
			else{
			cout << "Synapses type " << Type << " of neuron model " << EgidioGranuleCell_TimeDriven_GPU_C_INTERFACE::GetName() << " must receive spikes. The source model generates currents." << endl;
				return false;
			}
		}
		//Synapse types that process input current
		if (Type == N_TimeDependentNeuronState - 1){
			if (model->GetModelOutputActivityType() == OUTPUT_CURRENT){
				connection->SetSubindexType(this->CurrentSynapsis->GetNInputCurrentSynapsesPerNeuron(connection->GetTarget()->GetIndex_VectorNeuronState()));
				this->CurrentSynapsis->IncrementNInputCurrentSynapsesPerNeuron(connection->GetTarget()->GetIndex_VectorNeuronState());
				return true;
			}
			else{
				cout << "Synapses type " << Type << " of neuron model " << EgidioGranuleCell_TimeDriven_GPU_C_INTERFACE::GetName() << " must receive current. The source model generates spikes." << endl;
				return false;
			}
		}
	}
	cout << "Neuron model " << EgidioGranuleCell_TimeDriven_GPU_C_INTERFACE::GetName() << " does not support input synapses of type " << Type << ". Just defined " << N_TimeDependentNeuronState << " synapses types." << endl;
	return false;
}


std::map<std::string,boost::any> EgidioGranuleCell_TimeDriven_GPU_C_INTERFACE::GetParameters() const {
	// Return a dictionary with the parameters
	std::map<std::string,boost::any> newMap = TimeDrivenNeuronModel_GPU_C_INTERFACE::GetParameters();
	return newMap;
}

std::map<std::string, boost::any> EgidioGranuleCell_TimeDriven_GPU_C_INTERFACE::GetSpecificNeuronParameters(int index) const noexcept(false){
	return GetParameters();
}

void EgidioGranuleCell_TimeDriven_GPU_C_INTERFACE::SetParameters(std::map<std::string, boost::any> param_map) noexcept(false){

	// Search for the parameters in the dictionary
	TimeDrivenNeuronModel_GPU_C_INTERFACE::SetParameters(param_map);
	return;
}

IntegrationMethod_GPU_C_INTERFACE * EgidioGranuleCell_TimeDriven_GPU_C_INTERFACE::CreateIntegrationMethod(ModelDescription imethodDescription) noexcept(false){
	return IntegrationMethodFactory_GPU_C_INTERFACE<EgidioGranuleCell_TimeDriven_GPU_C_INTERFACE>::CreateIntegrationMethod_GPU(imethodDescription, (EgidioGranuleCell_TimeDriven_GPU_C_INTERFACE*) this);
}

std::map<std::string,boost::any> EgidioGranuleCell_TimeDriven_GPU_C_INTERFACE::GetDefaultParameters() {
	// Return a dictionary with the parameters
	std::map<std::string, boost::any> newMap = TimeDrivenNeuronModel_GPU_C_INTERFACE::GetDefaultParameters<EgidioGranuleCell_TimeDriven_GPU_C_INTERFACE>();
	return newMap;
}

NeuronModel* EgidioGranuleCell_TimeDriven_GPU_C_INTERFACE::CreateNeuronModel(ModelDescription nmDescription){
	EgidioGranuleCell_TimeDriven_GPU_C_INTERFACE * nmodel = new EgidioGranuleCell_TimeDriven_GPU_C_INTERFACE();
	nmodel->SetParameters(nmDescription.param_map);
	return nmodel;
}

ModelDescription EgidioGranuleCell_TimeDriven_GPU_C_INTERFACE::ParseNeuronModel(std::string FileName) noexcept(false){
	FILE *fh;
	ModelDescription nmodel;
	nmodel.model_name = EgidioGranuleCell_TimeDriven_GPU_C_INTERFACE::GetName();
	long Currentline = 0L;
	fh=fopen(FileName.c_str(),"rt");
	if(!fh) {
		throw EDLUTFileException(TASK_EGIDIO_GRANULE_CELL_TIME_DRIVEN_GPU_LOAD, ERROR_NEURON_MODEL_OPEN, REPAIR_NEURON_MODEL_NAME, Currentline, FileName.c_str());
	}

	Currentline = 1L;
	skip_comments(fh, Currentline);
	try {
		ModelDescription intMethodDescription = TimeDrivenNeuronModel_GPU_C_INTERFACE::ParseIntegrationMethod<EgidioGranuleCell_TimeDriven_GPU_C_INTERFACE>(fh, Currentline);
		nmodel.param_map["int_meth"] = boost::any(intMethodDescription);
	} catch (EDLUTException exc) {
		throw EDLUTFileException(exc, Currentline, FileName.c_str());
	}

	nmodel.param_map["name"] = boost::any(EgidioGranuleCell_TimeDriven_GPU_C_INTERFACE::GetName());

	fclose(fh);

	return nmodel;
}

std::string EgidioGranuleCell_TimeDriven_GPU_C_INTERFACE::GetName(){
	return "EgidioGranuleCell_TimeDriven_GPU";
}

std::map<std::string, std::string> EgidioGranuleCell_TimeDriven_GPU_C_INTERFACE::GetNeuronModelInfo() {
	// Return a dictionary with the parameters
	std::map<std::string, std::string> newMap;
	newMap["info"] = std::string("GPU Time-driven complex neuron model representing a cerebellar granular cell with fifteen differential equations(membrane potential (v) and several ionic-channel variables) and four types of input synapses: AMPA (excitatory), GABA (inhibitory), NMDA (excitatory) and external input current (set on pA)");
	newMap["int_meth"] = std::string("Integraton method dictionary (from the list of available integration methods in GPU)");
	return newMap;
}
