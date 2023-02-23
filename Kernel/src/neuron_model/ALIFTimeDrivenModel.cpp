/***************************************************************************
 *                           ALIFTimeDrivenModel.cpp                        *
 *                           -------------------                           *
 * copyright            : (C) 2022 by Álvaro González-Redondo              *
 * email                : alvarogr@ugr.es                                  *
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 3 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

#include "neuron_model/ALIFTimeDrivenModel.h"
#include "neuron_model/VectorNeuronState.h"
#include "neuron_model/CurrentSynapseModel.h"

#include "spike/Neuron.h"
#include "spike/Interconnection.h"

#include "integration_method/IntegrationMethodFactory.h"

// #include <iostream>
// #include <fstream>

// ofstream myfile;


void ALIFTimeDrivenModel::Generate_g_nmda_inf_values(){
	auxNMDA = (TableSizeNMDA - 1) / (e_exc - e_inh);
	for (int i = 0; i<TableSizeNMDA; i++){
		float V = e_inh + ((e_exc - e_inh)*i) / (TableSizeNMDA - 1);

		//g_nmda_inf
		g_nmda_inf_values[i] = 1.0f / (1.0f + exp(-0.062f*V)*(1.2f / 3.57f));
	}
}


float ALIFTimeDrivenModel::Get_g_nmda_inf(float V_m){
	int position = int((V_m - e_inh)*auxNMDA);
		if(position<0){
			position=0;
		}
		else if (position>(TableSizeNMDA - 1)){
			position = TableSizeNMDA - 1;
		}
		return g_nmda_inf_values[position];
}


void ALIFTimeDrivenModel::InitializeCurrentSynapsis(int N_neurons){
	this->CurrentSynapsis = new CurrentSynapseModel(N_neurons);
}


//this neuron model is implemented in a second scale.
ALIFTimeDrivenModel::ALIFTimeDrivenModel(): TimeDrivenNeuronModel(MilisecondScale), EXC(false), INH(false), NMDA(false), EXT_I(false){
	std::map<std::string, boost::any> param_map = ALIFTimeDrivenModel::GetDefaultParameters();
	param_map["name"] = ALIFTimeDrivenModel::GetName();
	this->SetParameters(param_map);

	this->State = (VectorNeuronState *) new VectorNeuronState(N_NeuronStateVariables, true);
}


ALIFTimeDrivenModel::~ALIFTimeDrivenModel(void){
}


VectorNeuronState * ALIFTimeDrivenModel::InitializeState(){
	return this->GetVectorNeuronState();
}


InternalSpike * ALIFTimeDrivenModel::ProcessInputSpike(Interconnection * inter, double time){
	// Add the effect of the input spike
	this->GetVectorNeuronState()->IncrementStateVariableAtCPU(inter->GetTargetNeuronModelIndex(), N_DifferentialNeuronState + inter->GetType(), inter->GetWeight());

	return 0;
}


void ALIFTimeDrivenModel::ProcessInputCurrent(Interconnection * inter, Neuron * target, float current){
	//Update the external current in the corresponding input synapse of type EXT_I (defined in pA).
	this->CurrentSynapsis->SetInputCurrent(target->GetIndex_VectorNeuronState(), inter->GetSubindexType(), current);

	//Update the total external current that receive the neuron coming from all its EXT_I synapsis (defined in pA).
	float total_ext_I = this->CurrentSynapsis->GetTotalCurrent(target->GetIndex_VectorNeuronState());
	State->SetStateVariableAt(target->GetIndex_VectorNeuronState(), EXT_I_index, total_ext_I);
}


bool ALIFTimeDrivenModel::UpdateState(int index, double CurrentTime){
	//Reset the number of internal spikes in this update period
	this->State->NInternalSpikeIndexs = 0;

	this->integration_method->NextDifferentialEquationValues();

	this->CheckValidIntegration(CurrentTime, this->integration_method->GetValidIntegrationVariable());

	return false;
}


enum NeuronModelOutputActivityType ALIFTimeDrivenModel::GetModelOutputActivityType(){
	return OUTPUT_SPIKE;
}


enum NeuronModelInputActivityType ALIFTimeDrivenModel::GetModelInputActivityType(){
	return INPUT_SPIKE_AND_CURRENT;
}


ostream & ALIFTimeDrivenModel::PrintInfo(ostream & out){
	out << "- Leaky Time-Driven Model: " << ALIFTimeDrivenModel::GetName() << endl;
	out << "\tExcitatory reversal potential (e_exc): " << this->e_exc << "mV" << endl;
	out << "\tInhibitory reversal potential (e_inh): " << this->e_inh << "mV" << endl;
	out << "\tEffective leak potential (e_leak): " << this->e_leak << "mV" << endl;
	out << "\tEffective threshold potential (v_thr): " << this->v_thr << "mV" << endl;
	out << "\tMembrane capacitance (c_m): " << this->c_m << "pF" << endl;
	out << "\tAMPA (excitatory) receptor time constant (tau_exc): " << this->tau_exc << "ms" << endl;
	out << "\tGABA (inhibitory) receptor time constant (tau_inh): " << this->tau_inh << "ms" << endl;
	out << "\tRefractory period (tau_ref): " << this->tau_ref << "ms" << endl;
	out << "\tLeak conductance (g_leak): " << this->g_leak << "nS" << endl;
	out << "\tNMDA (excitatory) receptor time constant (tau_nmda): " << this->tau_nmda << "ms" << endl;
	out << "\tThreshold time constant (tau_thr): " << this->tau_thr << "s" << endl;
	out << "\tTarget firing rate (tar_fir_rat): " << this->tar_fir_rat << "Hz" << endl;

	this->integration_method->PrintInfo(out);
	return out;
}


void ALIFTimeDrivenModel::InitializeStates(int N_neurons, int OpenMPQueueIndex){
	//Initialize neural state variables.
 float initialization[] = {e_leak,0.0f,0.0f,0.0f,0.0f,1.0f,v_thr-e_leak};
	// float initialization[] = {e_leak,0.0f,0.0f,0.0f,0.0f,1.0f,0.0f};
	State->InitializeStates(N_neurons, initialization);

	//Initialize integration method state variables.
	this->integration_method->SetBifixedStepParameters((e_leak + v_thr) / 2.0, (e_leak + v_thr) / 2.0, 0);
	this->integration_method->Calculate_conductance_exp_values();
	this->integration_method->InitializeStates(N_neurons, initialization);

	//Initialize the array that stores the number of input current synapses for each neuron in the model
	InitializeCurrentSynapsis(N_neurons);
}


void ALIFTimeDrivenModel::GetBifixedStepParameters(float & startVoltageThreshold, float & endVoltageThreshold, float & timeAfterEndVoltageThreshold){
	startVoltageThreshold = (e_leak+4*v_thr)/5;
	endVoltageThreshold = (e_leak+4*v_thr)/5;
	timeAfterEndVoltageThreshold = 0.0f;
	return;
}


void ALIFTimeDrivenModel::EvaluateSpikeCondition(float previous_V, float * NeuronState, int index, float elapsedTimeInNeuronModelScale){
	float thr = this->e_leak + NeuronState[DYN_THR_index];

	if (NeuronState[V_m_index] > thr) {
		NeuronState[V_m_index] = this->e_leak;
		State->NewFiredSpike(index);
		this->integration_method->resetState(index);
		this->State->InternalSpikeIndexs[this->State->NInternalSpikeIndexs] = index;
		this->State->NInternalSpikeIndexs++;

		// Increment the threshold
		if (this->tau_thr>0) {
		    NeuronState[DYN_THR_index] += 396e-3 * this->inv_tau_thr / this->tar_fir_rat;
			// NeuronState[DYN_THR_index] += 5000.0 * (this->inv_tau_thr / this->tar_fir_rat);
		}
	}
}


void ALIFTimeDrivenModel::EvaluateDifferentialEquation(float * NeuronState, float * AuxNeuronState, int index, float elapsed_time){
	for (int i=0;i<N_NeuronStateVariables;i++) AuxNeuronState[i] = 0.0f;

	float current = 0.0;
	if(EXC){
		current += NeuronState[EXC_index] * (this->e_exc - NeuronState[V_m_index]);
	}
	if(INH){
		current += NeuronState[INH_index] * (this->e_inh - NeuronState[V_m_index]);
	}
	if(NMDA){
		//float g_nmda_inf = 1.0f/(1.0f + ExponentialTable::GetResult(-62.0f*NeuronState[V_m_index])*(1.2f/3.57f));
		float g_nmda_inf = Get_g_nmda_inf(NeuronState[V_m_index]);
		current += NeuronState[NMDA_index] * g_nmda_inf*(this->e_exc - NeuronState[V_m_index]);
	}
	current += NeuronState[EXT_I_index]; // (defined in pA).

	if (this->GetVectorNeuronState()->GetLastSpikeTime(index)  * this->GetTimeScale()>this->tau_ref){
		AuxNeuronState[V_m_index] = (current + this->g_leak* (this->e_leak - NeuronState[V_m_index])) * this->inv_c_m;
	}
	else if ((this->GetVectorNeuronState()->GetLastSpikeTime(index)  * this->GetTimeScale() + elapsed_time)>this->tau_ref){
		float fraction = (this->GetVectorNeuronState()->GetLastSpikeTime(index)  * this->GetTimeScale() + elapsed_time - this->tau_ref) / elapsed_time;
		AuxNeuronState[V_m_index] = fraction*((current + this->g_leak * (this->e_leak - NeuronState[V_m_index])) * this->inv_c_m);
	}
	else{
		AuxNeuronState[V_m_index] = 0;
	}
}

void ALIFTimeDrivenModel::EvaluateTimeDependentEquation(float * NeuronState, int index, int elapsed_time_index){
	float limit=1e-9;
	float * Conductance_values=this->Get_conductance_exponential_values(elapsed_time_index);

	if(EXC){
		if (NeuronState[EXC_index]<limit){
			NeuronState[EXC_index] = 0.0f;
		}else{
			NeuronState[EXC_index] *= Conductance_values[0];
		}
	}
	if(INH){
		if (NeuronState[INH_index]<limit){
			NeuronState[INH_index] = 0.0f;
		}else{
			NeuronState[INH_index] *= Conductance_values[1];
		}
	}
	if(NMDA){
		if (NeuronState[NMDA_index]<limit){
			NeuronState[NMDA_index] = 0.0f;
		}else{
			NeuronState[NMDA_index] *= Conductance_values[2];
		}
	}
	if (this->tau_thr > 0.0f) {
		if (NeuronState[DYN_THR_index]<limit){
			NeuronState[DYN_THR_index] = 0.0f;
		}else{
			// NeuronState[DYN_THR_index] *= Conductance_values[4];
			NeuronState[TIME_index] -= Conductance_values[3];
			if (NeuronState[TIME_index] < 0.0f) {
				NeuronState[TIME_index] = 1.0f;
				NeuronState[DYN_THR_index] *= Conductance_values[4];
			}

      // Save the state of the adaptive threshold
      // if (myfile.is_open()) {
      //   myfile << NeuronState[DYN_THR_index] << "\t" << NeuronState[V_m_index] << "\n";
      // } else {
      //   myfile.open("/home/alvaro/data.tmp");
      // }

		}
	}
}

void ALIFTimeDrivenModel::Calculate_conductance_exp_values(int index, float elapsed_time){
	//excitatory synapse.
	Set_conductance_exp_values(index, 0, expf(-elapsed_time*this->inv_tau_exc));
	//inhibitory synapse.
	Set_conductance_exp_values(index, 1, expf(-elapsed_time*this->inv_tau_inh));
	//nmda synapse.
	Set_conductance_exp_values(index, 2, expf(-elapsed_time*this->inv_tau_nmda));
	//Elapsed time
	Set_conductance_exp_values(index, 3, elapsed_time);
	//Dynamic threshold.
	Set_conductance_exp_values(index, 4, expf(-elapsed_time*1e-3*this->inv_tau_thr));
}


bool ALIFTimeDrivenModel::CheckSynapseType(Interconnection * connection){
	int Type = connection->GetType();
	if (Type<N_TimeDependentNeuronState && Type >= 0){
		//activaty synapse type
		if (Type == 0){
			EXC = true;
		}
		if (Type == 1){
			INH = true;
		}
		if (Type == 2){
			NMDA = true;
		}
		if (Type == 3){
			EXT_I = true;
		}

		NeuronModel * model = connection->GetSource()->GetNeuronModel();
		//Synapse types that process input spikes
		if (Type != EXT_I_index-N_DifferentialNeuronState) {
			if (model->GetModelOutputActivityType() == OUTPUT_SPIKE){
				return true;
			}
			else{
			cout << "Synapses type " << Type << " of neuron model " << ALIFTimeDrivenModel::GetName() << " must receive spikes. The source model generates currents." << endl;
				return false;
			}
		}
		//Synapse types that process input current
		if (Type == EXT_I_index-N_DifferentialNeuronState) {
			if (model->GetModelOutputActivityType() == OUTPUT_CURRENT){
				connection->SetSubindexType(this->CurrentSynapsis->GetNInputCurrentSynapsesPerNeuron(connection->GetTarget()->GetIndex_VectorNeuronState()));
				this->CurrentSynapsis->IncrementNInputCurrentSynapsesPerNeuron(connection->GetTarget()->GetIndex_VectorNeuronState());
				return true;
			}
			else{
				cout << "Synapses type " << Type << " of neuron model " << ALIFTimeDrivenModel::GetName() << " must receive current. The source model generates spikes." << endl;
				return false;
			}
		}
	}
	cout << "Neuron model " << ALIFTimeDrivenModel::GetName() << " does not support input synapses of type " << Type << ". Just defined " << N_TimeDependentNeuronState << " synapses types." << endl;
	return false;
}

std::map<std::string,boost::any> ALIFTimeDrivenModel::GetParameters() const {
	// Return a dictionary with the parameters
	std::map<std::string,boost::any> newMap = TimeDrivenNeuronModel::GetParameters();
	newMap["e_exc"] = boost::any(this->e_exc); // Excitatory reversal potential (mV)
	newMap["e_inh"] = boost::any(this->e_inh); // Inhibitory reversal potential (mV)
	newMap["e_leak"] = boost::any(this->e_leak); // Effective leak potential (mV)
	newMap["v_thr"] = boost::any(this->v_thr); // Effective threshold potential (mV)
	newMap["c_m"] = boost::any(float(this->c_m)); // Membrane capacitance (pF)
	newMap["tau_exc"] = boost::any(this->tau_exc); // AMPA (excitatory) receptor time constant (ms)
	newMap["tau_inh"] = boost::any(this->tau_inh); // GABA (inhibitory) receptor time constant (ms)
	newMap["tau_ref"] = boost::any(this->tau_ref); // Refractory period (ms)
	newMap["g_leak"] = boost::any(float(this->g_leak)); // Leak conductance (nS)
	newMap["tau_nmda"] = boost::any(this->tau_nmda); // NMDA (excitatory) receptor time constant (ms)
	newMap["tau_thr"] = boost::any(this->tau_thr); // Threshold time constant (s)
	newMap["tar_fir_rat"] = boost::any(this->tar_fir_rat); // Target firing rate (Hz)

	return newMap;
}

std::map<std::string, boost::any> ALIFTimeDrivenModel::GetSpecificNeuronParameters(int index) const noexcept(false){
	return GetParameters();
}

void ALIFTimeDrivenModel::SetParameters(std::map<std::string, boost::any> param_map) noexcept(false){

	// Search for the parameters in the dictionary
	std::map<std::string,boost::any>::iterator it=param_map.find("e_exc");
	if (it!=param_map.end()){
		float new_param = boost::any_cast<float>(it->second);
		this->e_exc = new_param;
		param_map.erase(it);
	}

	it=param_map.find("e_inh");
	if (it!=param_map.end()){
		float new_param = boost::any_cast<float>(it->second);
		this->e_inh = new_param;
		param_map.erase(it);
	}

	it=param_map.find("e_leak");
	if (it!=param_map.end()){
		float new_param = boost::any_cast<float>(it->second);
		this->e_leak = new_param;
		param_map.erase(it);
	}

	it=param_map.find("v_thr");
	if (it!=param_map.end()){
		float new_param = boost::any_cast<float>(it->second);
		this->v_thr = new_param;
		param_map.erase(it);
	}

	it=param_map.find("c_m");
	if (it!=param_map.end()){
		float new_param = boost::any_cast<float>(it->second);
		this->c_m = new_param;
		this->inv_c_m = 1. / (new_param);
		param_map.erase(it);
	}

	it=param_map.find("tau_exc");
	if (it!=param_map.end()){
		float new_param = boost::any_cast<float>(it->second);
		this->tau_exc = new_param;
		this->inv_tau_exc = 1.0/new_param;
		param_map.erase(it);
	}

	it=param_map.find("tau_inh");
	if (it!=param_map.end()){
		float new_param = boost::any_cast<float>(it->second);
		this->tau_inh = new_param;
		this->inv_tau_inh = 1.0/new_param;
		param_map.erase(it);
	}

	it=param_map.find("tau_ref");
	if (it!=param_map.end()){
		float new_param = boost::any_cast<float>(it->second);
		this->tau_ref = new_param;
		param_map.erase(it);
	}

	it = param_map.find("g_leak");
	if (it != param_map.end()){
		float new_param = boost::any_cast<float>(it->second);
		this->g_leak = new_param;
		param_map.erase(it);
	}

	it=param_map.find("tau_nmda");
	if (it!=param_map.end()){
		float new_param = boost::any_cast<float>(it->second);
		this->tau_nmda = new_param;
		this->inv_tau_nmda = 1.0/new_param;
		param_map.erase(it);
	}

	it=param_map.find("tau_thr");
	if (it!=param_map.end()){
		float new_param = boost::any_cast<float>(it->second);
		this->tau_thr = new_param;
		this->inv_tau_thr = (this->tau_thr>0.0) ? (1.0 / this->tau_thr) : 0.0;
		param_map.erase(it);
	}

	it=param_map.find("tar_fir_rat");
	if (it!=param_map.end()){
		float new_param = boost::any_cast<float>(it->second);
		this->tar_fir_rat = new_param;
		param_map.erase(it);
	}

	// Search for the parameters in the dictionary
	TimeDrivenNeuronModel::SetParameters(param_map);

	//Set the new g_nmda_inf values based on the e_exc and e_inh parameters
	Generate_g_nmda_inf_values();

	return;
}


IntegrationMethod * ALIFTimeDrivenModel::CreateIntegrationMethod(ModelDescription imethodDescription) noexcept(false){
	return IntegrationMethodFactory<ALIFTimeDrivenModel>::CreateIntegrationMethod(imethodDescription, (ALIFTimeDrivenModel*) this);
}


std::map<std::string,boost::any> ALIFTimeDrivenModel::GetDefaultParameters() {
	// Return a dictionary with the parameters
	std::map<std::string, boost::any> newMap = TimeDrivenNeuronModel::GetDefaultParameters<ALIFTimeDrivenModel>();
	newMap["e_exc"] = boost::any(0.0f); // Excitatory reversal potential (mV)
	newMap["e_inh"] = boost::any(-80.0f); // Inhibitory reversal potential (mV)
	newMap["e_leak"] = boost::any(-65.0f); // Effective leak potential (mV)
	newMap["v_thr"] = boost::any(-50.0f); // Effective threshold potential (mV)
	newMap["c_m"] = boost::any(2.0f); // Membrane capacitance (pF)
	newMap["tau_exc"] = boost::any(5.0f); // AMPA (excitatory) receptor time constant (ms)
	newMap["tau_inh"] = boost::any(10.0f); // GABA (inhibitory) receptor time constant (ms)
	newMap["tau_ref"] = boost::any(1.0f); // Refractory period (ms)
	newMap["g_leak"] = boost::any(0.2f); // Leak conductance (nS)
	newMap["tau_nmda"] = boost::any(20.0f); // NMDA (excitatory) receptor time constant (ms)
	newMap["tau_thr"] = boost::any(0.0f); // Threshold time constant (s)
	newMap["tar_fir_rat"] = boost::any(5.0f); // Target firing rate (Hz)
	return newMap;
}

NeuronModel* ALIFTimeDrivenModel::CreateNeuronModel(ModelDescription nmDescription){
	ALIFTimeDrivenModel * nmodel = new ALIFTimeDrivenModel();
	nmodel->SetParameters(nmDescription.param_map);
	return nmodel;
}

ModelDescription ALIFTimeDrivenModel::ParseNeuronModel(std::string FileName) noexcept(false){
	FILE *fh;
	ModelDescription nmodel;
	nmodel.model_name = ALIFTimeDrivenModel::GetName();
	long Currentline = 0L;
	fh=fopen(FileName.c_str(),"rt");
	if(!fh) {
		throw EDLUTFileException(TASK_LIF_TIME_DRIVEN_MODEL_LOAD, ERROR_NEURON_MODEL_OPEN, REPAIR_NEURON_MODEL_NAME, Currentline, FileName.c_str());
	}

	Currentline = 1L;
	skip_comments(fh, Currentline);

	float param;
	if (fscanf(fh, "%f", &param) != 1) {
		throw EDLUTFileException(TASK_LIF_TIME_DRIVEN_MODEL_LOAD, ERROR_LIF_TIME_DRIVEN_MODEL_E_EXC, REPAIR_NEURON_MODEL_VALUES, Currentline, FileName.c_str());
	}
	nmodel.param_map["e_exc"] = boost::any(param);

	skip_comments(fh, Currentline);
	if (fscanf(fh, "%f", &param) != 1) {
		throw EDLUTFileException(TASK_LIF_TIME_DRIVEN_MODEL_LOAD, ERROR_LIF_TIME_DRIVEN_MODEL_E_INH, REPAIR_NEURON_MODEL_VALUES, Currentline, FileName.c_str());
	}
	nmodel.param_map["e_inh"] = boost::any(param);

	skip_comments(fh, Currentline);
	if (fscanf(fh, "%f", &param) != 1) {
		throw EDLUTFileException(TASK_LIF_TIME_DRIVEN_MODEL_LOAD, ERROR_LIF_TIME_DRIVEN_MODEL_E_LEAK, REPAIR_NEURON_MODEL_VALUES, Currentline, FileName.c_str());
	}
	nmodel.param_map["e_leak"] = boost::any(param);

	skip_comments(fh, Currentline);
	if (fscanf(fh, "%f", &param) != 1) {
		throw EDLUTFileException(TASK_LIF_TIME_DRIVEN_MODEL_LOAD, ERROR_LIF_TIME_DRIVEN_MODEL_V_THR, REPAIR_NEURON_MODEL_VALUES, Currentline, FileName.c_str());
	}
	nmodel.param_map["v_thr"] = boost::any(param);

	skip_comments(fh, Currentline);
	if (fscanf(fh, "%f", &param) != 1 || param <= 0.0f) {
		throw EDLUTFileException(TASK_LIF_TIME_DRIVEN_MODEL_LOAD, ERROR_LIF_TIME_DRIVEN_MODEL_C_M, REPAIR_NEURON_MODEL_VALUES, Currentline, FileName.c_str());
	}
	nmodel.param_map["c_m"] = boost::any(param);

	skip_comments(fh, Currentline);
	if (fscanf(fh, "%f", &param) != 1 || param <= 0.0f) {
		throw EDLUTFileException(TASK_LIF_TIME_DRIVEN_MODEL_LOAD, ERROR_LIF_TIME_DRIVEN_MODEL_TAU_EXC, REPAIR_NEURON_MODEL_VALUES, Currentline, FileName.c_str());
	}
	nmodel.param_map["tau_exc"] = boost::any(param);

	skip_comments(fh, Currentline);
	if (fscanf(fh, "%f", &param) != 1 || param <= 0.0f) {
		throw EDLUTFileException(TASK_LIF_TIME_DRIVEN_MODEL_LOAD, ERROR_LIF_TIME_DRIVEN_MODEL_TAU_INH, REPAIR_NEURON_MODEL_VALUES, Currentline, FileName.c_str());
	}
	nmodel.param_map["tau_inh"] = boost::any(param);

	skip_comments(fh, Currentline);
	if (fscanf(fh, "%f", &param) != 1 || param <= 0.0f) {
		throw EDLUTFileException(TASK_LIF_TIME_DRIVEN_MODEL_LOAD, ERROR_LIF_TIME_DRIVEN_MODEL_TAU_REF, REPAIR_NEURON_MODEL_VALUES, Currentline, FileName.c_str());
	}
	nmodel.param_map["tau_ref"] = boost::any(param);

	skip_comments(fh, Currentline);
	if (fscanf(fh, "%f", &param) != 1 || param <= 0.0f) {
		throw EDLUTFileException(TASK_LIF_TIME_DRIVEN_MODEL_LOAD, ERROR_LIF_TIME_DRIVEN_MODEL_G_LEAK, REPAIR_NEURON_MODEL_VALUES, Currentline, FileName.c_str());
	}
	nmodel.param_map["g_leak"] = boost::any(param);

	skip_comments(fh, Currentline);
	if (fscanf(fh, "%f", &param) != 1 || param <= 0.0f) {
		throw EDLUTFileException(TASK_LIF_TIME_DRIVEN_MODEL_LOAD, ERROR_LIF_TIME_DRIVEN_MODEL_TAU_NMDA, REPAIR_NEURON_MODEL_VALUES, Currentline, FileName.c_str());
	}
	nmodel.param_map["tau_nmda"] = boost::any(param);

	skip_comments(fh, Currentline);
  // if (fscanf(fh, "%f", &param) != 1 || param <= 0.0f) {
  //   throw EDLUTFileException(TASK_LIF_TIME_DRIVEN_MODEL_LOAD, ERROR_LIF_TIME_DRIVEN_MODEL_TAU_DA, REPAIR_NEURON_MODEL_VALUES, Currentline, FileName.c_str());
  // }
  // nmodel.param_map["tau_da"] = boost::any(param);
  //
  // skip_comments(fh, Currentline);
  // if (fscanf(fh, "%f", &param) != 1 || param <= 0.0f) {
  //   throw EDLUTFileException(TASK_LIF_TIME_DRIVEN_MODEL_LOAD, ERROR_LIF_TIME_DRIVEN_MODEL_DA_THR_FAC, REPAIR_NEURON_MODEL_VALUES, Currentline, FileName.c_str());
  // }
  // nmodel.param_map["da_thr_fac"] = boost::any(param);
  //
  // skip_comments(fh, Currentline);
  // if (fscanf(fh, "%f", &param) != 1 || param <= 0.0f) {
  //   throw EDLUTFileException(TASK_LIF_TIME_DRIVEN_MODEL_LOAD, ERROR_LIF_TIME_DRIVEN_MODEL_MAX_DA, REPAIR_NEURON_MODEL_VALUES, Currentline, FileName.c_str());
  // }
  // nmodel.param_map["max_da"] = boost::any(param);
  //
  // skip_comments(fh, Currentline);
  // if (fscanf(fh, "%f", &param) != 1 || param < 0.0f) {
  //   throw EDLUTFileException(TASK_LIF_TIME_DRIVEN_MODEL_LOAD, ERROR_LIF_TIME_DRIVEN_MODEL_MAX_DA, REPAIR_NEURON_MODEL_VALUES, Currentline, FileName.c_str());
  // }
  // nmodel.param_map["tau_thr"] = boost::any(param);
  //
  // skip_comments(fh, Currentline);
  // if (fscanf(fh, "%f", &param) != 1 || param <= 0.0f) {
  //   throw EDLUTFileException(TASK_LIF_TIME_DRIVEN_MODEL_LOAD, ERROR_LIF_TIME_DRIVEN_MODEL_MAX_DA, REPAIR_NEURON_MODEL_VALUES, Currentline, FileName.c_str());
  // }
  // nmodel.param_map["tar_fir_rat"] = boost::any(param);
  //
  //
  // skip_comments(fh, Currentline);
	try {
		ModelDescription intMethodDescription = TimeDrivenNeuronModel::ParseIntegrationMethod<ALIFTimeDrivenModel>(fh, Currentline);
		nmodel.param_map["int_meth"] = boost::any(intMethodDescription);
	} catch (EDLUTException exc) {
		throw EDLUTFileException(exc, Currentline, FileName.c_str());
	}

	nmodel.param_map["name"] = boost::any(ALIFTimeDrivenModel::GetName());

	fclose(fh);

	return nmodel;
}

std::string ALIFTimeDrivenModel::GetName(){
	return "ALIFTimeDrivenModel";
}

std::map<std::string, std::string> ALIFTimeDrivenModel::GetNeuronModelInfo() {
	// Return a dictionary with the parameters
	std::map<std::string, std::string> newMap;
	newMap["info"] = std::string("CPU Time-driven Leaky Integrate and Fire (LIF) neuron model with one differential equations(membrane potential (v)) and four types of input synapses: AMPA (excitatory), GABA (inhibitory), NMDA (excitatory) and external input current (set on pA)");
	newMap["e_exc"] = std::string("Excitatory reversal potential (mV)");
	newMap["e_inh"] = std::string("Inhibitory reversal potential (mV)");
	newMap["e_leak"] = std::string("Effective leak potential (mV)");
	newMap["v_thr"] = std::string("Effective threshold potential (mV)");
	newMap["c_m"] = std::string("Membrane capacitance (pF)");
	newMap["tau_exc"] = std::string("AMPA (excitatory) receptor time constant (ms)");
	newMap["tau_inh"] = std::string("GABA (inhibitory) receptor time constant (ms)");
	newMap["tau_ref"] = std::string("Refractory period (ms)");
	newMap["g_leak"] = std::string("Leak conductance (nS)");
	newMap["tau_nmda"] = std::string("NMDA (excitatory) receptor time constant (ms)");
	newMap["tau_thr"] = std::string("Threshold time constant (s)");
	newMap["tar_fir_rat"] = std::string("Target firing rate (Hz)");
	newMap["int_meth"] = std::string("Integraton method dictionary (from the list of available integration methods)");

	return newMap;
}
