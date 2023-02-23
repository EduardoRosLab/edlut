/***************************************************************************
 *                           AdExTimeDrivenModel_GPU_C_INTERFACE.cu        *
 *                           -------------------                           *
 * copyright            : (C) 2015 by Francisco Naveros                    *
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

#include "../../include/neuron_model/AdExTimeDrivenModel_GPU_C_INTERFACE.cuh"
#include "../../include/neuron_model/AdExTimeDrivenModel_GPU2.cuh"
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

void AdExTimeDrivenModel_GPU_C_INTERFACE::InitializeCurrentSynapsis(int N_neurons){
	this->CurrentSynapsis = new CurrentSynapseModel(N_neurons);
}

AdExTimeDrivenModel_GPU_C_INTERFACE::AdExTimeDrivenModel_GPU_C_INTERFACE(): TimeDrivenNeuronModel_GPU_C_INTERFACE(MilisecondScale), a(0), b(0),
	thr_slo_fac(0), v_thr(0), tau_w(0), e_exc(0), e_inh(0), e_reset(0), e_leak(0), g_leak(0), c_m(0), tau_exc(0), tau_inh(0), NeuronModel_GPU2(0),
	EXC(false), INH(false), NMDA(false), EXT_I(false){

	std::map<std::string, boost::any> param_map = AdExTimeDrivenModel_GPU_C_INTERFACE::GetDefaultParameters();
	param_map["name"] = AdExTimeDrivenModel_GPU_C_INTERFACE::GetName();
	this->SetParameters(param_map);

	this->State = (VectorNeuronState_GPU_C_INTERFACE *) new VectorNeuronState_GPU_C_INTERFACE(N_NeuronStateVariables);
}


AdExTimeDrivenModel_GPU_C_INTERFACE::~AdExTimeDrivenModel_GPU_C_INTERFACE(void){
	DeleteClassGPU2();
}

VectorNeuronState * AdExTimeDrivenModel_GPU_C_INTERFACE::InitializeState(){
	return this->GetVectorNeuronState();
}


InternalSpike * AdExTimeDrivenModel_GPU_C_INTERFACE::ProcessInputSpike(Interconnection * inter, double time){
	this->State_GPU->AuxStateCPU[inter->GetType()*State_GPU->GetSizeState() + inter->GetTargetNeuronModelIndex()] += inter->GetWeight();

	return 0;
}

void AdExTimeDrivenModel_GPU_C_INTERFACE::ProcessInputCurrent(Interconnection * inter, Neuron * target, float current){
	//Update the external current in the corresponding input synapse of type EXT_I (defined in pA).
	this->CurrentSynapsis->SetInputCurrent(target->GetIndex_VectorNeuronState(), inter->GetSubindexType(), current);

	//Update the total external current that receive the neuron coming from all its EXT_I synapsis (defined in pA).
	float total_ext_I = this->CurrentSynapsis->GetTotalCurrent(target->GetIndex_VectorNeuronState());
	this->State_GPU->AuxStateCPU[inter->GetType()*State_GPU->GetSizeState() + inter->GetTargetNeuronModelIndex()] = total_ext_I;
}


__global__ void AdExTimeDrivenModel_GPU_C_INTERFACE_UpdateState(AdExTimeDrivenModel_GPU2 ** NeuronModel_GPU2, double CurrentTime){
	(*NeuronModel_GPU2)->UpdateState(CurrentTime);
}


bool AdExTimeDrivenModel_GPU_C_INTERFACE::UpdateState(int index, double CurrentTime){
	if(prop.canMapHostMemory){
		AdExTimeDrivenModel_GPU_C_INTERFACE_UpdateState<<<N_block,N_thread>>>(NeuronModel_GPU2, CurrentTime);
	}else{
		HANDLE_ERROR(cudaMemcpy(State_GPU->AuxStateGPU,State_GPU->AuxStateCPU,this->N_TimeDependentNeuronState*State_GPU->SizeStates*sizeof(float),cudaMemcpyHostToDevice));
		AdExTimeDrivenModel_GPU_C_INTERFACE_UpdateState<<<N_block,N_thread>>>(NeuronModel_GPU2, CurrentTime);
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


enum NeuronModelOutputActivityType AdExTimeDrivenModel_GPU_C_INTERFACE::GetModelOutputActivityType(){
	return OUTPUT_SPIKE;
}

enum NeuronModelInputActivityType AdExTimeDrivenModel_GPU_C_INTERFACE::GetModelInputActivityType(){
	return INPUT_SPIKE_AND_CURRENT;
}


ostream & AdExTimeDrivenModel_GPU_C_INTERFACE::PrintInfo(ostream & out){
	out << "- AdEx Time-Driven Model GPU: " << AdExTimeDrivenModel_GPU_C_INTERFACE::GetName() << endl;
	out << "\tConductance (a): " << this->a << "nS" << endl;
	out << "\tSpike trigger adaptation (b): " << this->b << "pA" << endl;
	out << "\tThreshold slope factor (thr_slo_fac): " << this->thr_slo_fac << "mV" << endl;
	out << "\tEffective threshold potential (v_thr): " << this->v_thr << "mV" << endl;
	out << "\tAdaptation time constant (tau_w): " << this->tau_w << "ms" << endl;
	out << "\tExcitatory reversal potential (e_exc): " << this->e_exc << "mV" << endl;
	out << "\tInhibitory reversal potential (e_inh): " << this->e_inh << "mV" << endl;
	out << "\tReset potential (e_reset): " << this->e_reset << "mV" << endl;
	out << "\tEffective leak potential (e_leak): " << this->e_leak << "mV" << endl;
	out << "\tLeak conductance (g_leak): " << this->g_leak << "nS" << endl;
	out << "\tMembrane capacitance (c_m): " << this->c_m << "pF" << endl;
	out << "\tAMPA (excitatory) receptor time constant (tau_exc): " << this->tau_exc << "ms" << endl;
	out << "\tGABA (inhibitory) receptor time constant (tau_inh): " << this->tau_inh << "ms" << endl;
	out << "\tNMDA (excitatory) receptor time constant (tau_nmda): " << this->tau_nmda << "ms" << endl;

	this->integration_method_GPU->PrintInfo(out);
	return out;
}


void AdExTimeDrivenModel_GPU_C_INTERFACE::InitializeStates(int N_neurons, int OpenMPQueueIndex){

	//Select the correnpondent device.
	this->GPU_index = OpenMPQueueIndex % NumberOfGPUs;
	HANDLE_ERROR(cudaSetDevice(GPUsIndex[GPU_index]));
	HANDLE_ERROR(cudaEventCreate(&stop));
	HANDLE_ERROR(cudaGetDeviceProperties( &prop, GPUsIndex[OpenMPQueueIndex % NumberOfGPUs]));

	this->State_GPU = (VectorNeuronState_GPU_C_INTERFACE *) this->State;

	//Initialize neural state variables.
	float initialization[] = { e_leak, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
	State_GPU->InitializeStatesGPU(N_neurons, initialization, N_TimeDependentNeuronState, prop);

	//INITIALIZE CLASS IN GPU
	this->InitializeClassGPU2(N_neurons);

	InitializeVectorNeuronState_GPU2();

	//Initialize the array that stores the number of input current synapses for each neuron in the model
	InitializeCurrentSynapsis(N_neurons);
}



__global__ void AdExTimeDrivenModel_GPU_C_INTERFACE_InitializeClassGPU2(AdExTimeDrivenModel_GPU2 ** NeuronModel_GPU2, float a, float b, float thr_slo_fac,
	float v_thr, float tau_w, float e_exc, float e_inh, float e_reset, float e_leak, float g_leak, float c_m, float tau_exc, float tau_inh, float tau_nmda,
		char const* integrationName, int N_neurons, void ** Buffer_GPU){
	if(blockIdx.x==0 && threadIdx.x==0){
		(*NeuronModel_GPU2)=new AdExTimeDrivenModel_GPU2(a, b, thr_slo_fac, v_thr, tau_w, e_exc, e_inh, e_reset, e_leak, g_leak, c_m, tau_exc, tau_inh, tau_nmda,
			integrationName, N_neurons, Buffer_GPU);
	}
}

void AdExTimeDrivenModel_GPU_C_INTERFACE::InitializeClassGPU2(int N_neurons){
	cudaMalloc(&NeuronModel_GPU2, sizeof(AdExTimeDrivenModel_GPU2 **));

	char * integrationNameGPU;
	cudaMalloc((void **)&integrationNameGPU,32*4);
//REVISAR
	HANDLE_ERROR(cudaMemcpy(integrationNameGPU, &integration_method_GPU->name[0], 32 * 4, cudaMemcpyHostToDevice));

	this->N_thread = 128;
	this->N_block=prop.multiProcessorCount*16;
	if((N_neurons+N_thread-1)/N_thread < N_block){
		N_block = (N_neurons+N_thread-1)/N_thread;
	}
	int Total_N_thread=N_thread*N_block;

	integration_method_GPU->InitializeMemoryGPU(N_neurons, Total_N_thread);

	AdExTimeDrivenModel_GPU_C_INTERFACE_InitializeClassGPU2<<<1,1>>>(NeuronModel_GPU2, a, b, thr_slo_fac, v_thr, tau_w, e_exc, e_inh, e_reset, e_leak, g_leak, c_m,
		tau_exc, tau_inh, tau_nmda, integrationNameGPU, N_neurons, integration_method_GPU->Buffer_GPU);

	cudaFree(integrationNameGPU);
}


__global__ void initializeVectorNeuronState_GPU2(AdExTimeDrivenModel_GPU2 ** NeuronModel_GPU2, int NumberOfVariables, float * InitialStateGPU, float * AuxStateGPU, float * StateGPU, double * LastUpdateGPU, double * LastSpikeTimeGPU, bool * InternalSpikeGPU, int SizeStates){
	if(blockIdx.x==0 && threadIdx.x==0){
		(*NeuronModel_GPU2)->InitializeVectorNeuronState_GPU2(NumberOfVariables, InitialStateGPU, AuxStateGPU, StateGPU, LastUpdateGPU, LastSpikeTimeGPU, InternalSpikeGPU, SizeStates);
	}
}


void AdExTimeDrivenModel_GPU_C_INTERFACE::InitializeVectorNeuronState_GPU2(){
	VectorNeuronState_GPU_C_INTERFACE *state = (VectorNeuronState_GPU_C_INTERFACE *) State;
	initializeVectorNeuronState_GPU2<<<1,1>>>(NeuronModel_GPU2, state->NumberOfVariables, state->InitialStateGPU, state->AuxStateGPU, state->VectorNeuronStates_GPU, state->LastUpdateGPU, state->LastSpikeTimeGPU, state->InternalSpikeGPU, state->SizeStates);
}


__global__ void DeleteClass_GPU2(AdExTimeDrivenModel_GPU2 ** NeuronModel_GPU2){
	if(blockIdx.x==0 && threadIdx.x==0){
		delete (*NeuronModel_GPU2);
	}
}


void AdExTimeDrivenModel_GPU_C_INTERFACE::DeleteClassGPU2(){
	if (NeuronModel_GPU2 != 0){
		DeleteClass_GPU2 << <1, 1 >> >(NeuronModel_GPU2);
		cudaFree(NeuronModel_GPU2);
	}
}


__global__ void SetEnabledSynapsis_GPU2(AdExTimeDrivenModel_GPU2 ** NeuronModel_GPU2, bool new_EXC, bool new_INH, bool new_NMDA, bool new_EXT_I){
	if (blockIdx.x == 0 && threadIdx.x == 0){
		(*NeuronModel_GPU2)->SetEnabledSynapsis(new_EXC, new_INH, new_NMDA, new_EXT_I);
	}
}


bool AdExTimeDrivenModel_GPU_C_INTERFACE::CheckSynapseType(Interconnection * connection){
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
			cout << "Synapses type " << Type << " of neuron model " << AdExTimeDrivenModel_GPU_C_INTERFACE::GetName() << " must receive spikes. The source model generates currents." << endl;
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
				cout << "Synapses type " << Type << " of neuron model " << AdExTimeDrivenModel_GPU_C_INTERFACE::GetName() << " must receive current. The source model generates spikes." << endl;
				return false;
			}
		}
	}
	cout << "Neuron model " << AdExTimeDrivenModel_GPU_C_INTERFACE::GetName() << " does not support input synapses of type " << Type << ". Just defined " << N_TimeDependentNeuronState << " synapses types." << endl;
	return false;
}

std::map<std::string, boost::any> AdExTimeDrivenModel_GPU_C_INTERFACE::GetParameters() const {
	// Return a dictionary with the parameters
	std::map<std::string, boost::any> newMap = TimeDrivenNeuronModel_GPU_C_INTERFACE::GetParameters();
	newMap["a"] = boost::any(this->a); //conductance (nS)
	newMap["b"] = boost::any(this->b); //spike trigger adaptation (pA)
	newMap["thr_slo_fac"] = boost::any(this->thr_slo_fac); //threshold slope factor (mV)
	newMap["v_thr"] = boost::any(this->v_thr); //effective threshold potential (mV)
	newMap["tau_w"] = boost::any(this->tau_w); //adaptation time constant (ms)
	newMap["e_exc"] = boost::any(this->e_exc); //excitatory reversal potential (mV)
	newMap["e_inh"] = boost::any(this->e_inh); //inhibitory reversal potential (mV)
	newMap["e_reset"] = boost::any(this->e_reset); //reset potential (mV)
	newMap["e_leak"] = boost::any(this->e_leak); //effective leak potential (mV)
	newMap["g_leak"] = boost::any(this->g_leak); //leak conductance (nS)
	newMap["c_m"] = boost::any(this->c_m); //membrane capacitance (pF)
	newMap["tau_exc"] = boost::any(this->tau_exc); //AMPA (excitatory) receptor time constant (ms)
	newMap["tau_inh"] = boost::any(this->tau_inh); //GABA (inhibitory) receptor time constant (ms)
	newMap["tau_nmda"] = boost::any(this->tau_nmda); //NMDA (excitatory) receptor time constant (ms)
	return newMap;
}

std::map<std::string, boost::any> AdExTimeDrivenModel_GPU_C_INTERFACE::GetSpecificNeuronParameters(int index) const noexcept(false){
	return GetParameters();
}

void AdExTimeDrivenModel_GPU_C_INTERFACE::SetParameters(std::map<std::string, boost::any> param_map) noexcept(false){

	// Search for the parameters in the dictionary
	std::map<std::string, boost::any>::iterator it = param_map.find("a");
	if (it != param_map.end()){
		float new_param = boost::any_cast<float>(it->second);
		this->a = new_param;
		param_map.erase(it);
	}

	it = param_map.find("b");
	if (it != param_map.end()){
		float new_param = boost::any_cast<float>(it->second);
		this->b = new_param;
		param_map.erase(it);
	}

	it = param_map.find("thr_slo_fac");
	if (it != param_map.end()){
		float new_param = boost::any_cast<float>(it->second);
		this->thr_slo_fac = new_param;
		param_map.erase(it);
	}

	it = param_map.find("v_thr");
	if (it != param_map.end()){
		float new_param = boost::any_cast<float>(it->second);
		this->v_thr = new_param;
		param_map.erase(it);
	}

	it = param_map.find("tau_w");
	if (it != param_map.end()){
		float new_param = boost::any_cast<float>(it->second);
		this->tau_w = new_param;
		param_map.erase(it);
	}

	it = param_map.find("e_exc");
	if (it != param_map.end()){
		float new_param = boost::any_cast<float>(it->second);
		this->e_exc = new_param;
		param_map.erase(it);
	}

	it = param_map.find("e_inh");
	if (it != param_map.end()){
		float new_param = boost::any_cast<float>(it->second);
		this->e_inh = new_param;
		param_map.erase(it);
	}

	it = param_map.find("e_reset");
	if (it != param_map.end()){
		float new_param = boost::any_cast<float>(it->second);
		this->e_reset = new_param;
		param_map.erase(it);
	}

	it = param_map.find("e_leak");
	if (it != param_map.end()){
		float new_param = boost::any_cast<float>(it->second);
		this->e_leak = new_param;
		param_map.erase(it);
	}

	it = param_map.find("g_leak");
	if (it != param_map.end()){
		float new_param = boost::any_cast<float>(it->second);
		this->g_leak = new_param;
		param_map.erase(it);
	}

	it = param_map.find("c_m");
	if (it != param_map.end()){
		float new_param = boost::any_cast<float>(it->second);
		this->c_m = new_param;
		param_map.erase(it);
	}

	it = param_map.find("tau_exc");
	if (it != param_map.end()){
		float new_param = boost::any_cast<float>(it->second);
		this->tau_exc = new_param;
		param_map.erase(it);
	}

	it = param_map.find("tau_inh");
	if (it != param_map.end()){
		float new_param = boost::any_cast<float>(it->second);
		this->tau_inh = new_param;
		param_map.erase(it);
	}


	it=param_map.find("tau_nmda");
	if (it!=param_map.end()){
		float new_param = boost::any_cast<float>(it->second);
		this->tau_nmda = new_param;
		param_map.erase(it);
	}

	// Search for the parameters in the dictionary
	TimeDrivenNeuronModel_GPU_C_INTERFACE::SetParameters(param_map);
	return;
}


IntegrationMethod_GPU_C_INTERFACE * AdExTimeDrivenModel_GPU_C_INTERFACE::CreateIntegrationMethod(ModelDescription imethodDescription) noexcept(false){
	return IntegrationMethodFactory_GPU_C_INTERFACE<AdExTimeDrivenModel_GPU_C_INTERFACE>::CreateIntegrationMethod_GPU(imethodDescription, (AdExTimeDrivenModel_GPU_C_INTERFACE*) this);
}


std::map<std::string, boost::any> AdExTimeDrivenModel_GPU_C_INTERFACE::GetDefaultParameters() {
	// Return a dictionary with the parameters
	std::map<std::string, boost::any> newMap = TimeDrivenNeuronModel_GPU_C_INTERFACE::GetDefaultParameters<AdExTimeDrivenModel_GPU_C_INTERFACE>();
	newMap["a"] = boost::any(1.0f); //conductance (nS)
	newMap["b"] = boost::any(9.0f); //spike trigger adaptation (pA)
	newMap["thr_slo_fac"] = boost::any(2.0f); //threshold slope factor (mV)
	newMap["v_thr"] = boost::any(-50.0f); //effective threshold potential (mV)
	newMap["tau_w"] = boost::any(50.0f); //adaptation time constant (ms)
	newMap["e_exc"] = boost::any(0.0f); //excitatory reversal potential (mV)
	newMap["e_inh"] = boost::any(-80.0f); //inhibitory reversal potential (mV)
	newMap["e_reset"] = boost::any(-80.0f); //reset potential (mV)
	newMap["e_leak"] = boost::any(-65.0f); //effective leak potential (mV)
	newMap["g_leak"] = boost::any(10.0f); //leak conductance (nS)
	newMap["c_m"] = boost::any(110.0f); //membrane capacitance (pF)
	newMap["tau_exc"] = boost::any(5.0f); //AMPA (excitatory) receptor time constant (ms)
	newMap["tau_inh"] = boost::any(10.0f); //GABA (inhibitory) receptor time constant (ms)
	newMap["tau_nmda"] = boost::any(20.0f); //NMDA (excitatory) receptor time constant (ms)
	return newMap;
}

NeuronModel* AdExTimeDrivenModel_GPU_C_INTERFACE::CreateNeuronModel(ModelDescription nmDescription){
	AdExTimeDrivenModel_GPU_C_INTERFACE * nmodel = new AdExTimeDrivenModel_GPU_C_INTERFACE();
	nmodel->SetParameters(nmDescription.param_map);
	return nmodel;
}

ModelDescription AdExTimeDrivenModel_GPU_C_INTERFACE::ParseNeuronModel(std::string FileName) noexcept(false){
	FILE *fh;
	ModelDescription nmodel;
	nmodel.model_name = AdExTimeDrivenModel_GPU_C_INTERFACE::GetName();
	long Currentline = 0L;
	fh = fopen(FileName.c_str(), "rt");
	if (!fh) {
		throw EDLUTFileException(TASK_ADEX_TIME_DRIVEN_MODEL_GPU_LOAD, ERROR_NEURON_MODEL_OPEN, REPAIR_NEURON_MODEL_NAME, Currentline, FileName.c_str());
	}

	Currentline = 1L;
	skip_comments(fh, Currentline);

	float param;
	if (fscanf(fh, "%f", &param) != 1) {
		throw EDLUTFileException(TASK_ADEX_TIME_DRIVEN_MODEL_GPU_LOAD, ERROR_ADEX_TIME_DRIVEN_MODEL_A, REPAIR_NEURON_MODEL_VALUES, Currentline, FileName.c_str());
	}
	nmodel.param_map["a"] = boost::any(param);

	skip_comments(fh, Currentline);
	if (fscanf(fh, "%f", &param) != 1) {
		throw EDLUTFileException(TASK_ADEX_TIME_DRIVEN_MODEL_GPU_LOAD, ERROR_ADEX_TIME_DRIVEN_MODEL_B, REPAIR_NEURON_MODEL_VALUES, Currentline, FileName.c_str());
	}
	nmodel.param_map["b"] = boost::any(param);

	skip_comments(fh, Currentline);
	if (fscanf(fh, "%f", &param) != 1 || param <= 0.0f) {
		throw EDLUTFileException(TASK_ADEX_TIME_DRIVEN_MODEL_GPU_LOAD, ERROR_ADEX_TIME_DRIVEN_MODEL_THR_SLO_FAC, REPAIR_NEURON_MODEL_VALUES, Currentline, FileName.c_str());
	}
	nmodel.param_map["thr_slo_fac"] = boost::any(param);

	skip_comments(fh, Currentline);
	if (fscanf(fh, "%f", &param) != 1) {
		throw EDLUTFileException(TASK_ADEX_TIME_DRIVEN_MODEL_GPU_LOAD, ERROR_ADEX_TIME_DRIVEN_MODEL_V_THR, REPAIR_NEURON_MODEL_VALUES, Currentline, FileName.c_str());
	}
	nmodel.param_map["v_thr"] = boost::any(param);

	skip_comments(fh, Currentline);
	if (fscanf(fh, "%f", &param) != 1 || param <= 0.0f) {
		throw EDLUTFileException(TASK_ADEX_TIME_DRIVEN_MODEL_GPU_LOAD, ERROR_ADEX_TIME_DRIVEN_MODEL_TAU_W, REPAIR_NEURON_MODEL_VALUES, Currentline, FileName.c_str());
	}
	nmodel.param_map["tau_w"] = boost::any(param);

	skip_comments(fh, Currentline);
	if (fscanf(fh, "%f", &param) != 1) {
		throw EDLUTFileException(TASK_ADEX_TIME_DRIVEN_MODEL_GPU_LOAD, ERROR_ADEX_TIME_DRIVEN_MODEL_E_EXC, REPAIR_NEURON_MODEL_VALUES, Currentline, FileName.c_str());
	}
	nmodel.param_map["e_exc"] = boost::any(param);

	skip_comments(fh, Currentline);
	if (fscanf(fh, "%f", &param) != 1) {
		throw EDLUTFileException(TASK_ADEX_TIME_DRIVEN_MODEL_GPU_LOAD, ERROR_ADEX_TIME_DRIVEN_MODEL_E_INH, REPAIR_NEURON_MODEL_VALUES, Currentline, FileName.c_str());
	}
	nmodel.param_map["e_inh"] = boost::any(param);

	skip_comments(fh, Currentline);
	if (fscanf(fh, "%f", &param) != 1) {
		throw EDLUTFileException(TASK_ADEX_TIME_DRIVEN_MODEL_GPU_LOAD, ERROR_ADEX_TIME_DRIVEN_MODEL_E_RESET, REPAIR_NEURON_MODEL_VALUES, Currentline, FileName.c_str());
	}
	nmodel.param_map["e_reset"] = boost::any(param);

	skip_comments(fh, Currentline);
	if (fscanf(fh, "%f", &param) != 1) {
		throw EDLUTFileException(TASK_ADEX_TIME_DRIVEN_MODEL_GPU_LOAD, ERROR_ADEX_TIME_DRIVEN_MODEL_E_LEAK, REPAIR_NEURON_MODEL_VALUES, Currentline, FileName.c_str());
	}
	nmodel.param_map["e_leak"] = boost::any(param);

	skip_comments(fh, Currentline);
	if (fscanf(fh, "%f", &param) != 1 || param <= 0.0f) {
		throw EDLUTFileException(TASK_ADEX_TIME_DRIVEN_MODEL_GPU_LOAD, ERROR_ADEX_TIME_DRIVEN_MODEL_G_LEAK, REPAIR_NEURON_MODEL_VALUES, Currentline, FileName.c_str());
	}
	nmodel.param_map["g_leak"] = boost::any(param);

	skip_comments(fh, Currentline);
	if (fscanf(fh, "%f", &param) != 1 || param <= 0.0f) {
		throw EDLUTFileException(TASK_ADEX_TIME_DRIVEN_MODEL_GPU_LOAD, ERROR_ADEX_TIME_DRIVEN_MODEL_C_M, REPAIR_NEURON_MODEL_VALUES, Currentline, FileName.c_str());
	}
	nmodel.param_map["c_m"] = boost::any(param);

	skip_comments(fh, Currentline);
	if (fscanf(fh, "%f", &param) != 1 || param <= 0.0f) {
		throw EDLUTFileException(TASK_ADEX_TIME_DRIVEN_MODEL_GPU_LOAD, ERROR_ADEX_TIME_DRIVEN_MODEL_TAU_EXC, REPAIR_NEURON_MODEL_VALUES, Currentline, FileName.c_str());
	}
	nmodel.param_map["tau_exc"] = boost::any(param);

	skip_comments(fh, Currentline);
	if (fscanf(fh, "%f", &param) != 1 || param <= 0.0f) {
		throw EDLUTFileException(TASK_ADEX_TIME_DRIVEN_MODEL_GPU_LOAD, ERROR_ADEX_TIME_DRIVEN_MODEL_TAU_INH, REPAIR_NEURON_MODEL_VALUES, Currentline, FileName.c_str());
	}
	nmodel.param_map["tau_inh"] = boost::any(param);

	skip_comments(fh, Currentline);
	if (fscanf(fh, "%f", &param) != 1 || param <= 0.0f) {
		throw EDLUTFileException(TASK_ADEX_TIME_DRIVEN_MODEL_GPU_LOAD, ERROR_ADEX_TIME_DRIVEN_MODEL_TAU_NMDA, REPAIR_NEURON_MODEL_VALUES, Currentline, FileName.c_str());
	}
	nmodel.param_map["tau_nmda"] = boost::any(param);

	skip_comments(fh, Currentline);
	try {
		ModelDescription intMethodDescription = TimeDrivenNeuronModel_GPU_C_INTERFACE::ParseIntegrationMethod<AdExTimeDrivenModel_GPU_C_INTERFACE>(fh, Currentline);
		nmodel.param_map["int_meth"] = boost::any(intMethodDescription);
	}
	catch (EDLUTException exc) {
		throw EDLUTFileException(exc, Currentline, FileName.c_str());
	}

	nmodel.param_map["name"] = boost::any(AdExTimeDrivenModel_GPU_C_INTERFACE::GetName());

	fclose(fh);

	return nmodel;
}

std::string AdExTimeDrivenModel_GPU_C_INTERFACE::GetName(){
	return "AdExTimeDrivenModel_GPU";
}

std::map<std::string, std::string> AdExTimeDrivenModel_GPU_C_INTERFACE::GetNeuronModelInfo() {
	// Return a dictionary with the parameters
	std::map<std::string, std::string> newMap;
	newMap["info"] = std::string("GPU Time-driven Adaptative Exponential Integrate and Fire (AdEx) neuron model with two differential equations(membrane potential (v) and membrane recovery (w)) and four types of input synapses: AMPA (excitatory), GABA (inhibitory), NMDA (excitatory) and external input current (set on pA)");
	newMap["a"] = std::string("Conductance (nS)");
	newMap["b"] = std::string("Spike trigger adaptation (pA)");
	newMap["thr_slo_fac"] = std::string("Threshold slope factor (mV)");
	newMap["v_thr"] = std::string("Effective threshold potential (mV)");
	newMap["tau_w"] = std::string("Adaptation time constant (ms)");
	newMap["e_exc"] = std::string("Excitatory reversal potential (mV)");
	newMap["e_inh"] = std::string("Inhibitory reversal potential (mV)");
	newMap["e_reset"] = std::string("Reset potential (mV)");
	newMap["e_leak"] = std::string("Effective leak potential (mV)");
	newMap["g_leak"] = std::string("Leak conductance (nS)");
	newMap["c_m"] = std::string("Membrane capacitance (pF)");
	newMap["tau_exc"] = std::string("AMPA (excitatory) receptor time constant (ms)");
	newMap["tau_inh"] = std::string("GABA (inhibitory) receptor time constant (ms)");
	newMap["tau_nmda"] = std::string("NMDA (excitatory) receptor time constant (ms)");
	newMap["int_meth"] = std::string("Integraton method dictionary (from the list of available integration methods in GPU)");

	return newMap;
}
