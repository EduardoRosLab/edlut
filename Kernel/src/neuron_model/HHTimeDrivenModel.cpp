/***************************************************************************
 *                           HHTimeDrivenModel.cpp		                   *
 *                           -------------------                           *
 * copyright            : (C) 2015 by Francisco Naveros					   *
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

#include "neuron_model/HHTimeDrivenModel.h"
#include "neuron_model/VectorNeuronState.h"
#include "neuron_model/CurrentSynapseModel.h"
#include "simulation/ExponentialTable.h"
#include "spike/Neuron.h"
#include "spike/Interconnection.h"

#include "integration_method/IntegrationMethodFactory.h"


const float HHTimeDrivenModel::Max_V=50.0f;
const float HHTimeDrivenModel::Min_V=-90.0f;

const float HHTimeDrivenModel::aux=(HHTimeDrivenModel::TableSize-1)/( HHTimeDrivenModel::Max_V- HHTimeDrivenModel::Min_V);


void HHTimeDrivenModel::Generate_g_nmda_inf_values(){
	auxNMDA = (TableSizeNMDA - 1) / (e_exc - e_inh);
	for (int i = 0; i<TableSizeNMDA; i++){
		float V = e_inh + ((e_exc - e_inh)*i) / (TableSizeNMDA - 1);

		//g_nmda_inf
		g_nmda_inf_values[i] = 1.0f / (1.0f + exp(-0.062f*V)*(1.2f / 3.57f));
	}
}


float HHTimeDrivenModel::Get_g_nmda_inf(float V_m){
	int position = int((V_m - e_inh)*auxNMDA);
		if(position<0){
			position=0;
		}
		else if (position>(TableSizeNMDA - 1)){
			position = TableSizeNMDA - 1;
		}
		return g_nmda_inf_values[position];
}


void HHTimeDrivenModel::InitializeCurrentSynapsis(int N_neurons){
	this->CurrentSynapsis = new CurrentSynapseModel(N_neurons);
}


//this neuron model is implemented in a milisecond scale.
HHTimeDrivenModel::HHTimeDrivenModel() : TimeDrivenNeuronModel(MilisecondScale), EXC(false), INH(false), NMDA(false), EXT_I(false), channel_values(0)
{
	std::map<std::string, boost::any> param_map = HHTimeDrivenModel::GetDefaultParameters();
	param_map["name"] = HHTimeDrivenModel::GetName();
	this->SetParameters(param_map);

	this->State = (VectorNeuronState *) new VectorNeuronState(N_NeuronStateVariables, true);
}


HHTimeDrivenModel::~HHTimeDrivenModel()
{
	if (channel_values != 0){
		delete channel_values;
	}
}

VectorNeuronState * HHTimeDrivenModel::InitializeState(){
	return this->GetVectorNeuronState();
}


InternalSpike * HHTimeDrivenModel::ProcessInputSpike(Interconnection * inter, double time){
	// Add the effect of the input spike
	this->GetVectorNeuronState()->IncrementStateVariableAtCPU(inter->GetTargetNeuronModelIndex(), N_DifferentialNeuronState + inter->GetType(), inter->GetWeight());

	return 0;
}


void HHTimeDrivenModel::ProcessInputCurrent(Interconnection * inter, Neuron * target, float current){
	//Update the external current in the corresponding input synapse of type EXT_I (defined in pA).
	this->CurrentSynapsis->SetInputCurrent(target->GetIndex_VectorNeuronState(), inter->GetSubindexType(), current);

	//Update the total external current that receive the neuron coming from all its EXT_I synapsis (defined in pA).
	float total_ext_I = this->CurrentSynapsis->GetTotalCurrent(target->GetIndex_VectorNeuronState());
	State->SetStateVariableAt(target->GetIndex_VectorNeuronState(), EXT_I_index, total_ext_I);
}


bool HHTimeDrivenModel::UpdateState(int index, double CurrentTime){
	//Reset the number of internal spikes in this update period
	this->State->NInternalSpikeIndexs = 0;

	this->integration_method->NextDifferentialEquationValues();

	this->CheckValidIntegration(CurrentTime, this->integration_method->GetValidIntegrationVariable());

	return false;
}


enum NeuronModelOutputActivityType HHTimeDrivenModel::GetModelOutputActivityType(){
	return OUTPUT_SPIKE;
}


enum NeuronModelInputActivityType HHTimeDrivenModel::GetModelInputActivityType(){
	return INPUT_SPIKE_AND_CURRENT;
}


ostream & HHTimeDrivenModel::PrintInfo(ostream & out){
	out << "- HH Time-Driven Model: " << HHTimeDrivenModel::GetName() << endl;
	out << "\tExcitatory reversal potential (e_exc): " << this->e_exc << "mV" << endl;
	out << "\tInhibitory reversal potential (e_inh): " << this->e_inh << "mV" << endl;
	out << "\tEffective leak potential (e_leak): " << this->e_leak << "mV" << endl;
	out << "\tLeak conductance (g_leak): " << this->g_leak << "nS" << endl;
	out << "\tMembrane capacitance (c_m): " << this->c_m << "pF" << endl;
	out << "\tEffective threshold potential (v_thr): " << this->v_thr << "mV" << endl;
	out << "\tAMPA (excitatory) receptor time constant (tau_exc): " << this->tau_exc << "ms" << endl;
	out << "\tGABA (inhibitory) receptor time constant (tau_inh): " << this->tau_inh << "ms" << endl;
	out << "\tNMDA (excitatory) receptor time constant (tau_nmda): " << this->tau_nmda << "ms" << endl;
	out << "\tMaximum value of sodium conductance (g_na): " << this->g_na << "nS" << endl;
	out << "\tMaximum value of potassium conductance (g_kd): " << this->g_kd << "nS" << endl;
	out << "\tSodium potential (e_na): " << this->e_na << "mV" << endl;
	out << "\tPotassium potential (e_k): " << this->e_k << "mV" << endl;

	this->integration_method->PrintInfo(out);
	return out;
}


void HHTimeDrivenModel::InitializeStates(int N_neurons, int OpenMPQueueIndex){
	//Initialize neural state variables.
	float * values=Get_channel_values(e_leak);

	//m
	float alpha_m=values[0];
	float beta_m=values[1];
	float m_inf=alpha_m/(alpha_m+beta_m);

	//h
	float alpha_h=values[2];
	float beta_h=values[3];
	float h_inf=alpha_h/(alpha_h+beta_h);


	//n
	float alpha_n=values[4];
	float beta_n=values[5];
	float n_inf=alpha_n/(alpha_n+beta_n);

	float initialization[] = {e_leak, m_inf, h_inf, n_inf, 0.0f, 0.0f, 0.0f, 0.0f};
	State->InitializeStates(N_neurons, initialization);

	//Initialize integration method state variables.
	this->integration_method->SetBifixedStepParameters(v_thr, v_thr, 2.0f);
	this->integration_method->Calculate_conductance_exp_values();
	this->integration_method->InitializeStates(N_neurons, initialization);

	//Initialize the array that stores the number of input current synapses for each neuron in the model
	InitializeCurrentSynapsis(N_neurons);
}


void HHTimeDrivenModel::GetBifixedStepParameters(float & startVoltageThreshold, float & endVoltageThreshold, float & timeAfterEndVoltageThreshold){
	startVoltageThreshold = this->v_thr;
	endVoltageThreshold = this->v_thr;
	timeAfterEndVoltageThreshold = 2.0f;
	return;
}



void HHTimeDrivenModel::EvaluateSpikeCondition(float previous_V, float * NeuronState, int index, float elapsedTimeInNeuronModelScale){
	if (NeuronState[V_m_index] > -30.0f && previous_V < -30.0f){
		State->NewFiredSpike(index);
		this->State->InternalSpikeIndexs[this->State->NInternalSpikeIndexs] = index;
		this->State->NInternalSpikeIndexs++;
	}
}



void HHTimeDrivenModel::EvaluateDifferentialEquation(float * NeuronState, float * AuxNeuronState, int index, float elapsed_time){
	if (NeuronState[V_m_index]<e_k){
		NeuronState[V_m_index] = e_k;
	}
	if (NeuronState[V_m_index]>e_na - 1.0f){
		NeuronState[V_m_index] = e_na - 1.0f;
	}

	if (NeuronState[m_index]<0){
		NeuronState[m_index] = 0.0f;
	}
	if (NeuronState[m_index]>1){
		NeuronState[m_index] = 1.0f;
	}
	if (NeuronState[h_index]<0){
		NeuronState[h_index] = 0.0f;
	}
	if (NeuronState[h_index]>1){
		NeuronState[h_index] = 1.0f;
	}
	if (NeuronState[n_index]<0){
		NeuronState[n_index] = 0.0f;
	}
	if (NeuronState[n_index]>0.72){
		NeuronState[n_index] = 0.72f;
	}

	float V = NeuronState[V_m_index];
	float m = NeuronState[m_index];
	float h = NeuronState[h_index];
	float n = NeuronState[n_index];

	//Precomputed valued used to update m, h, and n variables.
	float * values = Get_channel_values(V);

	float current = 0.0;
	if(EXC){
		current += NeuronState[EXC_index] * (this->e_exc - NeuronState[V_m_index]);
	}
	if(INH){
		current += NeuronState[INH_index] * (this->e_inh - NeuronState[V_m_index]);
	}
	if(NMDA){
		//float g_nmda_inf = 1.0f/(1.0f + ExponentialTable::GetResult(-0.062f*NeuronState[V_m_index])*(1.2f/3.57f));
		float g_nmda_inf = Get_g_nmda_inf(NeuronState[V_m_index]);
		current += NeuronState[NMDA_index] * g_nmda_inf*(this->e_exc - NeuronState[V_m_index]);
	}
	current+=NeuronState[EXT_I_index]; // (defined in pA).

	//V
	AuxNeuronState[0]=(current + g_leak * (this->e_leak - V) + g_na*m*m*m*h*(e_na - V) + g_kd*n*n*n*n*(e_k - V))*this->inv_c_m;


	//m
	float alpha_m=values[0];
	float beta_m=values[1];
	AuxNeuronState[1]=(alpha_m*(1.0f-m)-beta_m*m);

	//h
	float alpha_h=values[2];
	float beta_h=values[3];
	AuxNeuronState[2]=(alpha_h*(1.0f-h)-beta_h*h);

	//n
	float alpha_n=values[4];
	float beta_n=values[5];
	AuxNeuronState[3]=(alpha_n*(1.0f-n)-beta_n*n);
}



void HHTimeDrivenModel::EvaluateTimeDependentEquation(float * NeuronState, int index, int elapsed_time_index){
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
}

void HHTimeDrivenModel::Calculate_conductance_exp_values(int index, float elapsed_time){
	//excitatory synapse.
	Set_conductance_exp_values(index, 0, expf(-elapsed_time*this->inv_tau_exc));
	//inhibitory synapse.
	Set_conductance_exp_values(index, 1, expf(-elapsed_time*this->inv_tau_inh));
	//nmda synapse.
	Set_conductance_exp_values(index, 2, expf(-elapsed_time*this->inv_tau_nmda));
}


bool HHTimeDrivenModel::CheckSynapseType(Interconnection * connection){
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
		if (Type < N_TimeDependentNeuronState - 1){
			if (model->GetModelOutputActivityType() == OUTPUT_SPIKE){
				return true;
			}
			else{
			cout << "Synapses type " << Type << " of neuron model " << HHTimeDrivenModel::GetName() << " must receive spikes. The source model generates currents." << endl;
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
				cout << "Synapses type " << Type << " of neuron model " << HHTimeDrivenModel::GetName() << " must receive current. The source model generates spikes." << endl;
				return false;
			}
		}
	}
	cout << "Neuron model " << HHTimeDrivenModel::GetName() << " does not support input synapses of type " << Type << ". Just defined " << N_TimeDependentNeuronState << " synapses types." << endl;
	return false;
}

std::map<std::string,boost::any> HHTimeDrivenModel::GetParameters() const {
	// Return a dictionary with the parameters
	std::map<std::string,boost::any> newMap = TimeDrivenNeuronModel::GetParameters();
	newMap["e_exc"] = boost::any(this->e_exc); //Excitatory reversal potential (mV)
	newMap["e_inh"] = boost::any(this->e_inh); // Inhibitory reversal potential mV)
	newMap["e_leak"] = boost::any(this->e_leak); // Effective leak potential(mV)
	newMap["g_leak"] = boost::any(this->g_leak); // Leak conductance (nS)
	newMap["c_m"] = boost::any(this->c_m); // Membrane capacitance (pF)
	newMap["v_thr"] = boost::any(this->v_thr); // Effective threshold potential (mV)
	newMap["tau_exc"] = boost::any(this->tau_exc); // AMPA (excitatory) receptor time constant (ms)
	newMap["tau_inh"] = boost::any(this->tau_inh); // GABA (inhibitory) receptor time constant (ms)
	newMap["tau_nmda"] = boost::any(this->tau_nmda); // NMDA (excitatory) receptor time constant (ms)
	newMap["g_na"] = boost::any(this->g_na); // Maximum value of sodium conductance (nS)
	newMap["g_kd"] = boost::any(this->g_kd); // Maximum value of potassium conductance (nS)
	newMap["e_na"] = boost::any(this->e_na); // Sodium potential (mV)
	newMap["e_k"] = boost::any(this->e_k); // Potassium potential (mV)
	return newMap;
}

std::map<std::string, boost::any> HHTimeDrivenModel::GetSpecificNeuronParameters(int index) const noexcept(false){
	return GetParameters();
}

void HHTimeDrivenModel::SetParameters(std::map<std::string, boost::any> param_map) noexcept(false){

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

	it=param_map.find("g_leak");
	if (it!=param_map.end()){
		float new_param = boost::any_cast<float>(it->second);
		this->g_leak = new_param;
		param_map.erase(it);
	}

	it=param_map.find("c_m");
	if (it!=param_map.end()){
		float new_param = boost::any_cast<float>(it->second);
		this->c_m = new_param;
		this->inv_c_m = 1./this->c_m;
		param_map.erase(it);
	}

	it = param_map.find("v_thr");
	if (it != param_map.end()){
		float new_param = boost::any_cast<float>(it->second);
		this->v_thr = new_param;
		param_map.erase(it);
	}

	it=param_map.find("tau_exc");
	if (it!=param_map.end()){
		float new_param = boost::any_cast<float>(it->second);
		this->tau_exc = new_param;
		this->inv_tau_exc = 1.0 / new_param;
		param_map.erase(it);
	}

	it=param_map.find("tau_inh");
	if (it!=param_map.end()){
		float new_param = boost::any_cast<float>(it->second);
		this->tau_inh = new_param;
		this->inv_tau_inh = 1.0 / new_param;
		param_map.erase(it);
	}

	it=param_map.find("tau_nmda");
	if (it!=param_map.end()){
		float new_param = boost::any_cast<float>(it->second);
		this->tau_nmda = new_param;
		this->inv_tau_nmda = 1.0/new_param;
		param_map.erase(it);
	}

	it=param_map.find("g_na");
	if (it!=param_map.end()){
		float new_param = boost::any_cast<float>(it->second);
		this->g_na = new_param;
		param_map.erase(it);
	}

	it=param_map.find("g_kd");
	if (it!=param_map.end()){
		float new_param = boost::any_cast<float>(it->second);
		this->g_kd = new_param;
		param_map.erase(it);
	}

	it=param_map.find("e_na");
	if (it!=param_map.end()){
		float new_param = boost::any_cast<float>(it->second);
		this->e_na = new_param;
		param_map.erase(it);
	}

	it=param_map.find("e_k");
	if (it!=param_map.end()){
		float new_param = boost::any_cast<float>(it->second);
		this->e_k = new_param;
		param_map.erase(it);
	}

	// Search for the parameters in the dictionary
	TimeDrivenNeuronModel::SetParameters(param_map);

	//Set the new g_nmda_inf values based on the e_exc and e_inh parameters
	Generate_g_nmda_inf_values();

	//Precompute in look-up tables the ion-channel variables
	this->Generate_channel_values();

	return;
}


IntegrationMethod * HHTimeDrivenModel::CreateIntegrationMethod(ModelDescription imethodDescription) noexcept(false){
	return IntegrationMethodFactory<HHTimeDrivenModel>::CreateIntegrationMethod(imethodDescription, (HHTimeDrivenModel*) this);
}


std::map<std::string,boost::any> HHTimeDrivenModel::GetDefaultParameters() {
	// Return a dictionary with the parameters
	std::map<std::string,boost::any> newMap = TimeDrivenNeuronModel::GetDefaultParameters<HHTimeDrivenModel>();
	newMap["e_exc"] = boost::any(0.0f); //Excitatory reversal potential (mV)
	newMap["e_inh"] = boost::any(-80.0f); // Inhibitory reversal potential mV)
	newMap["e_leak"] = boost::any(-65.0f); // Effective leak potential(mV)
	newMap["g_leak"] = boost::any(10.0f); // Leak conductance (nS)
	newMap["c_m"] = boost::any(120.0f); // Membrane capacitance (pF)
	newMap["v_thr"] = boost::any(-52.0f); // Effective threshold potential (mV)
	newMap["tau_exc"] = boost::any(5.0f); // AMPA (excitatory) receptor time constant (ms)
	newMap["tau_inh"] = boost::any(10.0f); // GABA (inhibitory) receptor time constant (ms)
	newMap["tau_nmda"] = boost::any(20.0f); // NMDA (excitatory) receptor time constant (ms)
	newMap["g_na"] = boost::any(20000.0f); // Maximum value of sodium conductance (nS)
	newMap["g_kd"] = boost::any(6000.0f); // Maximum value of potassium conductance (nS)
	newMap["e_na"] = boost::any(50.0f); // Sodium potential (mV)
	newMap["e_k"] = boost::any(-90.0f); // Potassium potential (mV)
	return newMap;
}

NeuronModel* HHTimeDrivenModel::CreateNeuronModel(ModelDescription nmDescription){
	HHTimeDrivenModel * nmodel = new HHTimeDrivenModel();
	nmodel->SetParameters(nmDescription.param_map);
	return nmodel;
}

ModelDescription HHTimeDrivenModel::ParseNeuronModel(std::string FileName) noexcept(false){
	FILE *fh;
	ModelDescription nmodel;
	nmodel.model_name = HHTimeDrivenModel::GetName();
	long Currentline = 0L;
	fh=fopen(FileName.c_str(),"rt");
	if(!fh) {
		throw EDLUTFileException(TASK_HH_TIME_DRIVEN_MODEL_LOAD, ERROR_NEURON_MODEL_OPEN, REPAIR_NEURON_MODEL_NAME, Currentline, FileName.c_str());
	}

	Currentline = 1L;
	float param;

	skip_comments(fh, Currentline);
	if (fscanf(fh, "%f", &param) != 1) {
		throw EDLUTFileException(TASK_HH_TIME_DRIVEN_MODEL_LOAD, ERROR_HH_TIME_DRIVEN_MODEL_E_EXC, REPAIR_NEURON_MODEL_VALUES, Currentline, FileName.c_str());
	}
	nmodel.param_map["e_exc"] = boost::any(param);

	skip_comments(fh, Currentline);
	if (fscanf(fh, "%f", &param) != 1) {
		throw EDLUTFileException(TASK_HH_TIME_DRIVEN_MODEL_LOAD, ERROR_HH_TIME_DRIVEN_MODEL_E_INH, REPAIR_NEURON_MODEL_VALUES, Currentline, FileName.c_str());
	}
	nmodel.param_map["e_inh"] = boost::any(param);

	skip_comments(fh, Currentline);
	if (fscanf(fh, "%f", &param) != 1) {
		throw EDLUTFileException(TASK_HH_TIME_DRIVEN_MODEL_LOAD, ERROR_HH_TIME_DRIVEN_MODEL_E_LEAK, REPAIR_NEURON_MODEL_VALUES, Currentline, FileName.c_str());
	}
	nmodel.param_map["e_leak"] = boost::any(param);

	skip_comments(fh, Currentline);
	if (fscanf(fh, "%f", &param) != 1 || param <= 0.0f) {
		throw EDLUTFileException(TASK_HH_TIME_DRIVEN_MODEL_LOAD, ERROR_HH_TIME_DRIVEN_MODEL_G_LEAK, REPAIR_NEURON_MODEL_VALUES, Currentline, FileName.c_str());
	}
	nmodel.param_map["g_leak"] = boost::any(param);

	skip_comments(fh, Currentline);
	if (fscanf(fh, "%f", &param) != 1 || param <= 0.0f) {
		throw EDLUTFileException(TASK_HH_TIME_DRIVEN_MODEL_LOAD, ERROR_HH_TIME_DRIVEN_MODEL_C_M, REPAIR_NEURON_MODEL_VALUES, Currentline, FileName.c_str());
	}
	nmodel.param_map["c_m"] = boost::any(param);

	skip_comments(fh, Currentline);
	if (fscanf(fh, "%f", &param) != 1) {
		throw EDLUTFileException(TASK_HH_TIME_DRIVEN_MODEL_LOAD, ERROR_HH_TIME_DRIVEN_MODEL_V_THR, REPAIR_NEURON_MODEL_VALUES, Currentline, FileName.c_str());
	}
	nmodel.param_map["v_thr"] = boost::any(param);

	skip_comments(fh, Currentline);
	if (fscanf(fh, "%f", &param) != 1 || param <= 0.0f) {
		throw EDLUTFileException(TASK_HH_TIME_DRIVEN_MODEL_LOAD, ERROR_HH_TIME_DRIVEN_MODEL_TAU_EXC, REPAIR_NEURON_MODEL_VALUES, Currentline, FileName.c_str());
	}
	nmodel.param_map["tau_exc"] = boost::any(param);

	skip_comments(fh, Currentline);
	if (fscanf(fh, "%f", &param) != 1 || param <= 0.0f) {
		throw EDLUTFileException(TASK_HH_TIME_DRIVEN_MODEL_LOAD, ERROR_HH_TIME_DRIVEN_MODEL_TAU_INH, REPAIR_NEURON_MODEL_VALUES, Currentline, FileName.c_str());
	}
	nmodel.param_map["tau_inh"] = boost::any(param);

	skip_comments(fh, Currentline);
	if (fscanf(fh, "%f", &param) != 1 || param <= 0.0f) {
		throw EDLUTFileException(TASK_HH_TIME_DRIVEN_MODEL_LOAD, ERROR_HH_TIME_DRIVEN_MODEL_TAU_NMDA, REPAIR_NEURON_MODEL_VALUES, Currentline, FileName.c_str());
	}
	nmodel.param_map["tau_nmda"] = boost::any(param);

	skip_comments(fh, Currentline);
	if (fscanf(fh, "%f", &param) != 1 || param <= 0.0f) {
		throw EDLUTFileException(TASK_HH_TIME_DRIVEN_MODEL_LOAD, ERROR_HH_TIME_DRIVEN_MODEL_G_NA, REPAIR_NEURON_MODEL_VALUES, Currentline, FileName.c_str());
	}
	nmodel.param_map["g_na"] = boost::any(param);

	skip_comments(fh, Currentline);
	if (fscanf(fh, "%f", &param) != 1 || param <= 0.0f) {
		throw EDLUTFileException(TASK_HH_TIME_DRIVEN_MODEL_LOAD, ERROR_HH_TIME_DRIVEN_MODEL_G_KD, REPAIR_NEURON_MODEL_VALUES, Currentline, FileName.c_str());
	}
	nmodel.param_map["g_kd"] = boost::any(param);

	skip_comments(fh, Currentline);
	if (fscanf(fh, "%f", &param) != 1) {
		throw EDLUTFileException(TASK_HH_TIME_DRIVEN_MODEL_LOAD, ERROR_HH_TIME_DRIVEN_MODEL_E_NA, REPAIR_NEURON_MODEL_VALUES, Currentline, FileName.c_str());
	}
	nmodel.param_map["e_na"] = boost::any(param);

	skip_comments(fh, Currentline);
	if (fscanf(fh, "%f", &param) != 1) {
		throw EDLUTFileException(TASK_HH_TIME_DRIVEN_MODEL_LOAD, ERROR_HH_TIME_DRIVEN_MODEL_E_K, REPAIR_NEURON_MODEL_VALUES, Currentline, FileName.c_str());
	}
	nmodel.param_map["e_k"] = boost::any(param);

	skip_comments(fh, Currentline);
	try {
		ModelDescription intMethodDescription = TimeDrivenNeuronModel::ParseIntegrationMethod<HHTimeDrivenModel>(fh, Currentline);
		nmodel.param_map["int_meth"] = boost::any(intMethodDescription);
	} catch (EDLUTException exc) {
		throw EDLUTFileException(exc, Currentline, FileName.c_str());
	}

	nmodel.param_map["name"] = boost::any(HHTimeDrivenModel::GetName());

	fclose(fh);

	return nmodel;
}

std::string HHTimeDrivenModel::GetName(){
	return "HHTimeDrivenModel";
}

std::map<std::string, std::string> HHTimeDrivenModel::GetNeuronModelInfo() {
	// Return a dictionary with the parameters
	std::map<std::string, std::string> newMap;
	newMap["info"] = std::string("CPU Time-driven Hodgkin and Huxley (HH) neuron model with four differential equations (membrane potential (v) and three ionic-channel variables (m, h and n)) and four types of input synapses: AMPA (excitatory), GABA (inhibitory), NMDA (excitatory) and external input current (set on pA)");
	newMap["e_exc"] = std::string("Excitatory reversal potential (mV)");
	newMap["e_inh"] = std::string("Inhibitory reversal potential mV)");
	newMap["e_leak"] = std::string("Effective leak potential(mV)");
	newMap["g_leak"] = std::string("Leak conductance (nS)");
	newMap["c_m"] = std::string("Membrane capacitance (pF)");
	newMap["v_thr"] = std::string("Effective threshold potential (mV)");
	newMap["tau_exc"] = std::string("AMPA (excitatory) receptor time constant (ms)");
	newMap["tau_inh"] = std::string("GABA (inhibitory) receptor time constant (ms)");
	newMap["tau_nmda"] = std::string("NMDA (excitatory) receptor time constant (ms)");
	newMap["g_na"] = std::string("Maximum value of sodium conductance (nS)");
	newMap["g_kd"] = std::string("Maximum value of potassium conductance (nS)");
	newMap["e_na"] = std::string("Sodium potential (mV)");
	newMap["e_k"] = std::string("Potassium potential (mV)");
	newMap["int_meth"] = std::string("Integraton method dictionary (from the list of available integration methods)");

	return newMap;
}
