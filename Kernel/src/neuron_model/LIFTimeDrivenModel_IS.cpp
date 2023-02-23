/***************************************************************************
 *                           LIFTimeDrivenModel_IS.cpp                     *
 *                           -------------------                           *
 * copyright            : (C) 2019 by Francisco Naveros                    *
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

#include "neuron_model/LIFTimeDrivenModel_IS.h"
#include "neuron_model/VectorNeuronState.h"
#include "neuron_model/CurrentSynapseModel.h"

#include "spike/Neuron.h"
#include "spike/Interconnection.h"

#include "integration_method/IntegrationMethodFactory.h"


void LIFTimeDrivenModel_IS::Generate_g_nmda_inf_values(){
	auxNMDA = (TableSizeNMDA - 1) / (e_exc - e_inh);
	for (int i = 0; i<TableSizeNMDA; i++){
		float V = e_inh + ((e_exc - e_inh)*i) / (TableSizeNMDA - 1);

		//g_nmda_inf
		g_nmda_inf_values[i] = 1.0f / (1.0f + exp(-62.0f*V)*(1.2f / 3.57f));
	}
}


float LIFTimeDrivenModel_IS::Get_g_nmda_inf(float V_m){
	int position = int((V_m - e_inh)*auxNMDA);
		if(position<0){
			position=0;
		}
		else if (position>(TableSizeNMDA - 1)){
			position = TableSizeNMDA - 1;
		}
		return g_nmda_inf_values[position];
}


void LIFTimeDrivenModel_IS::InitializeCurrentSynapsis(int N_neurons){
	this->CurrentSynapsis = new CurrentSynapseModel(N_neurons);
}


//this neuron model is implemented in a second scale.
LIFTimeDrivenModel_IS::LIFTimeDrivenModel_IS(): TimeDrivenNeuronModel(SecondScale), EXC(false), INH(false), NMDA(false), EXT_I(false){
	std::map<std::string, boost::any> param_map = LIFTimeDrivenModel_IS::GetDefaultParameters();
	param_map["name"] = LIFTimeDrivenModel_IS::GetName();
	this->SetParameters(param_map);

	this->State = (VectorNeuronState *) new VectorNeuronState(N_NeuronStateVariables, true);
}


LIFTimeDrivenModel_IS::~LIFTimeDrivenModel_IS(void){
}


VectorNeuronState * LIFTimeDrivenModel_IS::InitializeState(){
	return this->GetVectorNeuronState();
}


InternalSpike * LIFTimeDrivenModel_IS::ProcessInputSpike(Interconnection * inter, double time){
	// Add the effect of the input spike
	this->GetVectorNeuronState()->IncrementStateVariableAtCPU(inter->GetTargetNeuronModelIndex(), N_DifferentialNeuronState + inter->GetType(), inter->GetWeight());

	return 0;
}


void LIFTimeDrivenModel_IS::ProcessInputCurrent(Interconnection * inter, Neuron * target, float current){
	//Update the external current in the corresponding input synapse of type EXT_I (defined in pA).
	this->CurrentSynapsis->SetInputCurrent(target->GetIndex_VectorNeuronState(), inter->GetSubindexType(), current);

	//Update the total external current that receive the neuron coming from all its EXT_I synapsis (defined in pA).
	float total_ext_I = this->CurrentSynapsis->GetTotalCurrent(target->GetIndex_VectorNeuronState());
	State->SetStateVariableAt(target->GetIndex_VectorNeuronState(), EXT_I_index, total_ext_I);
}


bool LIFTimeDrivenModel_IS::UpdateState(int index, double CurrentTime){
	//Reset the number of internal spikes in this update period
	this->State->NInternalSpikeIndexs = 0;

	this->integration_method->NextDifferentialEquationValues();

	this->CheckValidIntegration(CurrentTime, this->integration_method->GetValidIntegrationVariable());

	return false;
}


enum NeuronModelOutputActivityType LIFTimeDrivenModel_IS::GetModelOutputActivityType(){
	return OUTPUT_SPIKE;
}


enum NeuronModelInputActivityType LIFTimeDrivenModel_IS::GetModelInputActivityType(){
	return INPUT_SPIKE_AND_CURRENT;
}


ostream & LIFTimeDrivenModel_IS::PrintInfo(ostream & out){
	out << "- Leaky Time-Driven Model: " << LIFTimeDrivenModel_IS::GetName() << endl;
	out << "\tExcitatory reversal potential (e_exc): " << this->e_exc << "V" << endl;
	out << "\tInhibitory reversal potential (e_inh): " << this->e_inh << "V" << endl;
	out << "\tEffective leak potential (e_leak): " << this->e_leak << "V" << endl;
	out << "\tEffective threshold potential (v_thr): " << this->v_thr << "V" << endl;
	out << "\tMembrane capacitance (c_m): " << this->c_m << "F" << endl;
	out << "\tAMPA (excitatory) receptor time constant (tau_exc): " << this->tau_exc << "s" << endl;
	out << "\tGABA (inhibitory) receptor time constant (tau_inh): " << this->tau_inh << "s" << endl;
	out << "\tRefractory period (tau_ref): " << this->tau_ref << "s" << endl;
	out << "\tLeak conductance (g_leak): " << this->g_leak << "S" << endl;
	out << "\tNMDA (excitatory) receptor time constant (tau_nmda): " << this->tau_nmda << "s" << endl;

	this->integration_method->PrintInfo(out);
	return out;
}


void LIFTimeDrivenModel_IS::InitializeStates(int N_neurons, int OpenMPQueueIndex){
	//Initialize neural state variables.
	float initialization[] = {e_leak,0.0f,0.0f,0.0f,0.0f};
	State->InitializeStates(N_neurons, initialization);

	//Initialize integration method state variables.
	this->integration_method->SetBifixedStepParameters((e_leak + v_thr) / 2.0, (e_leak + v_thr) / 2.0, 0);
	this->integration_method->Calculate_conductance_exp_values();
	this->integration_method->InitializeStates(N_neurons, initialization);

	//Initialize the array that stores the number of input current synapses for each neuron in the model
	InitializeCurrentSynapsis(N_neurons);
}


void LIFTimeDrivenModel_IS::GetBifixedStepParameters(float & startVoltageThreshold, float & endVoltageThreshold, float & timeAfterEndVoltageThreshold){
	startVoltageThreshold = (e_leak+4*v_thr)/5;
	endVoltageThreshold = (e_leak+4*v_thr)/5;
	timeAfterEndVoltageThreshold = 0.0f;
	return;
}


void LIFTimeDrivenModel_IS::EvaluateSpikeCondition(float previous_V, float * NeuronState, int index, float elapsedTimeInNeuronModelScale){
	if (NeuronState[V_m_index] > this->v_thr){
		NeuronState[V_m_index] = this->e_leak;
		State->NewFiredSpike(index);
		this->integration_method->resetState(index);
		this->State->InternalSpikeIndexs[this->State->NInternalSpikeIndexs] = index;
		this->State->NInternalSpikeIndexs++;
	}
}


void LIFTimeDrivenModel_IS::EvaluateDifferentialEquation(float * NeuronState, float * AuxNeuronState, int index, float elapsed_time){
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
	float external_current_nA = 0.001 * NeuronState[EXT_I_index]; // (defined in pA).

	current += external_current_nA; // (defined in nA).

	if (this->GetVectorNeuronState()->GetLastSpikeTime(index)>this->tau_ref){
		AuxNeuronState[V_m_index] = (current + this->g_leak_nS* (this->e_leak - NeuronState[V_m_index])) * this->inv_c_m_nF;
	}
	else if ((this->GetVectorNeuronState()->GetLastSpikeTime(index) + elapsed_time)>this->tau_ref){
		float fraction = (this->GetVectorNeuronState()->GetLastSpikeTime(index) + elapsed_time - this->tau_ref) / elapsed_time;
		AuxNeuronState[V_m_index] = fraction*((current + this->g_leak_nS* (this->e_leak - NeuronState[V_m_index])) * this->inv_c_m_nF);
	}
	else{
		AuxNeuronState[V_m_index] = 0;
	}
}

void LIFTimeDrivenModel_IS::EvaluateTimeDependentEquation(float * NeuronState, int index, int elapsed_time_index){
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

void LIFTimeDrivenModel_IS::Calculate_conductance_exp_values(int index, float elapsed_time){
	//excitatory synapse.
	Set_conductance_exp_values(index, 0, expf(-elapsed_time*this->inv_tau_exc));
	//inhibitory synapse.
	Set_conductance_exp_values(index, 1, expf(-elapsed_time*this->inv_tau_inh));
	//nmda synapse.
	Set_conductance_exp_values(index, 2, expf(-elapsed_time*this->inv_tau_nmda));
}


bool LIFTimeDrivenModel_IS::CheckSynapseType(Interconnection * connection){
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
			cout << "Synapses type " << Type << " of neuron model " << LIFTimeDrivenModel_IS::GetName() << " must receive spikes. The source model generates currents." << endl;
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
				cout << "Synapses type " << Type << " of neuron model " << LIFTimeDrivenModel_IS::GetName() << " must receive current. The source model generates spikes." << endl;
				return false;
			}
		}
	}
	cout << "Neuron model " << LIFTimeDrivenModel_IS::GetName() << " does not support input synapses of type " << Type << ". Just defined " << N_TimeDependentNeuronState << " synapses types." << endl;
	return false;
}

std::map<std::string,boost::any> LIFTimeDrivenModel_IS::GetParameters() const {
	// Return a dictionary with the parameters
	std::map<std::string,boost::any> newMap = TimeDrivenNeuronModel::GetParameters();
	newMap["e_exc"] = boost::any(this->e_exc); // Excitatory reversal potential (V)
	newMap["e_inh"] = boost::any(this->e_inh); // Inhibitory reversal potential (V)
	newMap["e_leak"] = boost::any(this->e_leak); // Effective leak potential (V)
	newMap["v_thr"] = boost::any(this->v_thr); // Effective threshold potential (V)
	newMap["c_m"] = boost::any(float(this->c_m)); // Membrane capacitance (F)
	newMap["tau_exc"] = boost::any(this->tau_exc); // AMPA (excitatory) receptor time constant (s)
	newMap["tau_inh"] = boost::any(this->tau_inh); // GABA (inhibitory) receptor time constant (s)
	newMap["tau_ref"] = boost::any(this->tau_ref); // Refractory period (s)
	newMap["g_leak"] = boost::any(float(this->g_leak)); // Leak conductance (S)
	newMap["tau_nmda"] = boost::any(this->tau_nmda); // NMDA (excitatory) receptor time constant (s)
	return newMap;
}

std::map<std::string, boost::any> LIFTimeDrivenModel_IS::GetSpecificNeuronParameters(int index) const noexcept(false){
	return GetParameters();
}

void LIFTimeDrivenModel_IS::SetParameters(std::map<std::string, boost::any> param_map) noexcept(false){

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
		this->inv_c_m_nF = 1. / (new_param*1.e9);
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
		this->g_leak_nS = new_param*1.e9;
		param_map.erase(it);
	}

	it=param_map.find("tau_nmda");
	if (it!=param_map.end()){
		float new_param = boost::any_cast<float>(it->second);
		this->tau_nmda = new_param;
		this->inv_tau_nmda = 1.0/new_param;
		param_map.erase(it);
	}

	// Search for the parameters in the dictionary
	TimeDrivenNeuronModel::SetParameters(param_map);

	//Set the new g_nmda_inf values based on the e_exc and e_inh parameters
	Generate_g_nmda_inf_values();

	return;
}


IntegrationMethod * LIFTimeDrivenModel_IS::CreateIntegrationMethod(ModelDescription imethodDescription) noexcept(false){
	return IntegrationMethodFactory<LIFTimeDrivenModel_IS>::CreateIntegrationMethod(imethodDescription, (LIFTimeDrivenModel_IS*) this);
}


std::map<std::string,boost::any> LIFTimeDrivenModel_IS::GetDefaultParameters() {
	// Return a dictionary with the parameters
	std::map<std::string, boost::any> newMap = TimeDrivenNeuronModel::GetDefaultParameters<LIFTimeDrivenModel_IS>();
	newMap["e_exc"] = boost::any(0.0f); // Excitatory reversal potential (V)
	newMap["e_inh"] = boost::any(-80.0e-3f); // Inhibitory reversal potential (V)
	newMap["e_leak"] = boost::any(-65.0e-3f); // Effective leak potential (V)
	newMap["v_thr"] = boost::any(-50.0e-3f); // Effective threshold potential (V)
	newMap["c_m"] = boost::any(2.0e-12f); // Membrane capacitance (F)
	newMap["tau_exc"] = boost::any(5.0e-3f); // AMPA (excitatory) receptor time constant (s)
	newMap["tau_inh"] = boost::any(10.0e-3f); // GABA (inhibitory) receptor time constant (s)
	newMap["tau_ref"] = boost::any(1.0e-3f); // Refractory period (s)
	newMap["g_leak"] = boost::any(0.2e-9f); // Leak conductance (S)
	newMap["tau_nmda"] = boost::any(20.0e-3f); // NMDA (excitatory) receptor time constant (s)
	return newMap;
}

NeuronModel* LIFTimeDrivenModel_IS::CreateNeuronModel(ModelDescription nmDescription){
	LIFTimeDrivenModel_IS * nmodel = new LIFTimeDrivenModel_IS();
	nmodel->SetParameters(nmDescription.param_map);
	return nmodel;
}

ModelDescription LIFTimeDrivenModel_IS::ParseNeuronModel(std::string FileName) noexcept(false){
	FILE *fh;
	ModelDescription nmodel;
	nmodel.model_name = LIFTimeDrivenModel_IS::GetName();
	long Currentline = 0L;
	fh=fopen(FileName.c_str(),"rt");
	if(!fh) {
		throw EDLUTFileException(TASK_LIF_TIME_DRIVEN_MODEL_IS_LOAD, ERROR_NEURON_MODEL_OPEN, REPAIR_NEURON_MODEL_NAME, Currentline, FileName.c_str());
	}

	Currentline = 1L;
	skip_comments(fh, Currentline);

	float param;
	if (fscanf(fh, "%f", &param) != 1) {
		throw EDLUTFileException(TASK_LIF_TIME_DRIVEN_MODEL_IS_LOAD, ERROR_LIF_TIME_DRIVEN_MODEL_IS_E_EXC, REPAIR_NEURON_MODEL_VALUES, Currentline, FileName.c_str());
	}
	nmodel.param_map["e_exc"] = boost::any(param);

	skip_comments(fh, Currentline);
	if (fscanf(fh, "%f", &param) != 1) {
		throw EDLUTFileException(TASK_LIF_TIME_DRIVEN_MODEL_IS_LOAD, ERROR_LIF_TIME_DRIVEN_MODEL_IS_E_INH, REPAIR_NEURON_MODEL_VALUES, Currentline, FileName.c_str());
	}
	nmodel.param_map["e_inh"] = boost::any(param);

	skip_comments(fh, Currentline);
	if (fscanf(fh, "%f", &param) != 1) {
		throw EDLUTFileException(TASK_LIF_TIME_DRIVEN_MODEL_IS_LOAD, ERROR_LIF_TIME_DRIVEN_MODEL_IS_E_LEAK, REPAIR_NEURON_MODEL_VALUES, Currentline, FileName.c_str());
	}
	nmodel.param_map["e_leak"] = boost::any(param);

	skip_comments(fh, Currentline);
	if (fscanf(fh, "%f", &param) != 1) {
		throw EDLUTFileException(TASK_LIF_TIME_DRIVEN_MODEL_IS_LOAD, ERROR_LIF_TIME_DRIVEN_MODEL_IS_V_THR, REPAIR_NEURON_MODEL_VALUES, Currentline, FileName.c_str());
	}
	nmodel.param_map["v_thr"] = boost::any(param);

	skip_comments(fh, Currentline);
	if (fscanf(fh, "%f", &param) != 1 || param <= 0.0f) {
		throw EDLUTFileException(TASK_LIF_TIME_DRIVEN_MODEL_IS_LOAD, ERROR_LIF_TIME_DRIVEN_MODEL_IS_C_M, REPAIR_NEURON_MODEL_VALUES, Currentline, FileName.c_str());
	}
	nmodel.param_map["c_m"] = boost::any(param);

	skip_comments(fh, Currentline);
	if (fscanf(fh, "%f", &param) != 1 || param <= 0.0f) {
		throw EDLUTFileException(TASK_LIF_TIME_DRIVEN_MODEL_IS_LOAD, ERROR_LIF_TIME_DRIVEN_MODEL_IS_TAU_EXC, REPAIR_NEURON_MODEL_VALUES, Currentline, FileName.c_str());
	}
	nmodel.param_map["tau_exc"] = boost::any(param);

	skip_comments(fh, Currentline);
	if (fscanf(fh, "%f", &param) != 1 || param <= 0.0f) {
		throw EDLUTFileException(TASK_LIF_TIME_DRIVEN_MODEL_IS_LOAD, ERROR_LIF_TIME_DRIVEN_MODEL_IS_TAU_INH, REPAIR_NEURON_MODEL_VALUES, Currentline, FileName.c_str());
	}
	nmodel.param_map["tau_inh"] = boost::any(param);

	skip_comments(fh, Currentline);
	if (fscanf(fh, "%f", &param) != 1 || param <= 0.0f) {
		throw EDLUTFileException(TASK_LIF_TIME_DRIVEN_MODEL_IS_LOAD, ERROR_LIF_TIME_DRIVEN_MODEL_IS_TAU_REF, REPAIR_NEURON_MODEL_VALUES, Currentline, FileName.c_str());
	}
	nmodel.param_map["tau_ref"] = boost::any(param);

	skip_comments(fh, Currentline);
	if (fscanf(fh, "%f", &param) != 1 || param <= 0.0f) {
		throw EDLUTFileException(TASK_LIF_TIME_DRIVEN_MODEL_IS_LOAD, ERROR_LIF_TIME_DRIVEN_MODEL_IS_G_LEAK, REPAIR_NEURON_MODEL_VALUES, Currentline, FileName.c_str());
	}
	nmodel.param_map["g_leak"] = boost::any(param);

	skip_comments(fh, Currentline);
	if (fscanf(fh, "%f", &param) != 1 || param <= 0.0f) {
		throw EDLUTFileException(TASK_LIF_TIME_DRIVEN_MODEL_IS_LOAD, ERROR_LIF_TIME_DRIVEN_MODEL_IS_TAU_NMDA, REPAIR_NEURON_MODEL_VALUES, Currentline, FileName.c_str());
	}
	nmodel.param_map["tau_nmda"] = boost::any(param);

	skip_comments(fh, Currentline);
	try {
		ModelDescription intMethodDescription = TimeDrivenNeuronModel::ParseIntegrationMethod<LIFTimeDrivenModel_IS>(fh, Currentline);
		nmodel.param_map["int_meth"] = boost::any(intMethodDescription);
	} catch (EDLUTException exc) {
		throw EDLUTFileException(exc, Currentline, FileName.c_str());
	}

	nmodel.param_map["name"] = boost::any(LIFTimeDrivenModel_IS::GetName());

	fclose(fh);

	return nmodel;
}

std::string LIFTimeDrivenModel_IS::GetName(){
	return "LIFTimeDrivenModel_IS";
}

std::map<std::string, std::string> LIFTimeDrivenModel_IS::GetNeuronModelInfo() {
	// Return a dictionary with the parameters
	std::map<std::string, std::string> newMap;
	newMap["info"] = std::string("CPU Time-driven Leaky Integrate and Fire (LIF) neuron model with one differential equations(membrane potential (v)) and four types of input synapses: AMPA (excitatory), GABA (inhibitory), NMDA (excitatory) and external input current (set on pA). THIS MODEL USES THE INTERNATIONAL SYSTEM UNITS IN ITS PARAMETERS.");
	newMap["e_exc"] = std::string("Excitatory reversal potential (V)");
	newMap["e_inh"] = std::string("Inhibitory reversal potential (V)");
	newMap["e_leak"] = std::string("Effective leak potential (V)");
	newMap["v_thr"] = std::string("Effective threshold potential (V)");
	newMap["c_m"] = std::string("Membrane capacitance (F)");
	newMap["tau_exc"] = std::string("AMPA (excitatory) receptor time constant (s)");
	newMap["tau_inh"] = std::string("GABA (inhibitory) receptor time constant (s)");
	newMap["tau_ref"] = std::string("Refractory period (s)");
	newMap["g_leak"] = std::string("Leak conductance (S)");
	newMap["tau_nmda"] = std::string("NMDA (excitatory) receptor time constant (s)");
	newMap["int_meth"] = std::string("Integraton method dictionary (from the list of available integration methods)");

	return newMap;
}
