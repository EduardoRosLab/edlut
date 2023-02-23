/***************************************************************************
 *                           IzhikevichTimeDrivenModel.cpp                 *
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

#include "neuron_model/IzhikevichTimeDrivenModel.h"
#include "neuron_model/VectorNeuronState.h"
#include "neuron_model/CurrentSynapseModel.h"
#include "simulation/ExponentialTable.h"
#include "spike/Neuron.h"
#include "spike/Interconnection.h"

#include "integration_method/IntegrationMethodFactory.h"




void IzhikevichTimeDrivenModel::Generate_g_nmda_inf_values(){
	auxNMDA = (TableSizeNMDA - 1) / (e_exc - e_inh);
	for (int i = 0; i<TableSizeNMDA; i++){
		float V = e_inh + ((e_exc - e_inh)*i) / (TableSizeNMDA - 1);

		//g_nmda_inf
		g_nmda_inf_values[i] = 1.0f / (1.0f + exp(-0.062f*V)*(1.2f / 3.57f));
	}
}


float IzhikevichTimeDrivenModel::Get_g_nmda_inf(float V_m){
	int position = int((V_m - e_inh)*auxNMDA);
		if(position<0){
			position=0;
		}
		else if (position>(TableSizeNMDA - 1)){
			position = TableSizeNMDA - 1;
		}
		return g_nmda_inf_values[position];
}


void IzhikevichTimeDrivenModel::InitializeCurrentSynapsis(int N_neurons){
	this->CurrentSynapsis = new CurrentSynapseModel(N_neurons);
}


//this neuron model is implemented in a milisecond scale.
IzhikevichTimeDrivenModel::IzhikevichTimeDrivenModel(): TimeDrivenNeuronModel(MilisecondScale), EXC(false), INH(false), NMDA(false), EXT_I(false){
	std::map<std::string, boost::any> param_map = IzhikevichTimeDrivenModel::GetDefaultParameters();
	param_map["name"] = IzhikevichTimeDrivenModel::GetName();
	this->SetParameters(param_map);

	this->State = (VectorNeuronState *) new VectorNeuronState(N_NeuronStateVariables, true);
}

IzhikevichTimeDrivenModel::~IzhikevichTimeDrivenModel(void)
{
}

VectorNeuronState * IzhikevichTimeDrivenModel::InitializeState(){
	return this->GetVectorNeuronState();
}


InternalSpike * IzhikevichTimeDrivenModel::ProcessInputSpike(Interconnection * inter, double time){
	// Add the effect of the input spike
	this->GetVectorNeuronState()->IncrementStateVariableAtCPU(inter->GetTargetNeuronModelIndex(), N_DifferentialNeuronState + inter->GetType(), inter->GetWeight());

	return 0;
}


void IzhikevichTimeDrivenModel::ProcessInputCurrent(Interconnection * inter, Neuron * target, float current){
	//Update the external current in the corresponding input synapse of type EXT_I (defined in pA).
	this->CurrentSynapsis->SetInputCurrent(target->GetIndex_VectorNeuronState(), inter->GetSubindexType(), current);

	//Update the total external current that receive the neuron coming from all its EXT_I synapsis (defined in pA).
	float total_ext_I = this->CurrentSynapsis->GetTotalCurrent(target->GetIndex_VectorNeuronState());
	State->SetStateVariableAt(target->GetIndex_VectorNeuronState(), EXT_I_index, total_ext_I);
}

bool IzhikevichTimeDrivenModel::UpdateState(int index, double CurrentTime){
	//Reset the number of internal spikes in this update period
	this->State->NInternalSpikeIndexs = 0;

	this->integration_method->NextDifferentialEquationValues();

	this->CheckValidIntegration(CurrentTime, this->integration_method->GetValidIntegrationVariable());

	return false;
}


enum NeuronModelOutputActivityType IzhikevichTimeDrivenModel::GetModelOutputActivityType(){
	return OUTPUT_SPIKE;
}


enum NeuronModelInputActivityType IzhikevichTimeDrivenModel::GetModelInputActivityType(){
	return INPUT_SPIKE_AND_CURRENT;
}


ostream & IzhikevichTimeDrivenModel::PrintInfo(ostream & out){
	out << "- Izhikevich Time-Driven Model: " << IzhikevichTimeDrivenModel::GetName() << endl;
	out << "\tTime scale of recovery variable u (a): " << this->a << "dimensionless" << endl;
	out << "\tSensitivity of the recovery variable u to the subthreshold fluctuations of the membrane potential v (b): " << this->b << "dimensionless" << endl;
	out << "\tAfter-spike reset value of the membrane potential v (c): " << this->c << "dimensionless" << endl;
	out << "\tAfter-spike reset of the recovery variable u (d): " << this->d << "dimensionless" << endl;
	out << "\tExcitatory reversal potential (e_exc): " << this->e_exc << "mV" << endl;
	out << "\tInhibitory reversal potential (e_inh): " << this->e_inh << "mV" << endl;
	out << "\tMembrane capacitance (c_m): " << this->c_m << "pF" << endl;
	out << "\tAMPA (excitatory) receptor time constant (tau_exc): " << this->tau_exc << "ms" << endl;
	out << "\tGABA (inhibitory) receptor time constant (tau_inh): " << this->tau_inh << "ms" << endl;
	out << "\tNMDA (excitatory) receptor time constant (tau_nmda): " << this->tau_nmda << "ms" << endl;

	this->integration_method->PrintInfo(out);
	return out;
}

void IzhikevichTimeDrivenModel::InitializeStates(int N_neurons, int OpenMPQueueIndex){
	//Initialize neural state variables.
	float Veq=(((b-5.0f)-sqrt((5.0f-b)*(5.0f-b)-22.4f))/0.08f);
	float Ueq=Veq*b;
	float initialization[] = {Veq, Ueq,0.0f,0.0f,0.0f,0.0f};
	State->InitializeStates(N_neurons, initialization);

	//Initialize integration method state variables.
	this->integration_method->SetBifixedStepParameters(-40.0f, this->c + 5.0f, 2.0f);
	this->integration_method->Calculate_conductance_exp_values();
	this->integration_method->InitializeStates(N_neurons, initialization);

	//Initialize the array that stores the number of input current synapses for each neuron in the model
	InitializeCurrentSynapsis(N_neurons);
}

void IzhikevichTimeDrivenModel::GetBifixedStepParameters(float & startVoltageThreshold, float & endVoltageThreshold, float & timeAfterEndVoltageThreshold){
    startVoltageThreshold = -40.0f;
    endVoltageThreshold = c+5.0f;
    timeAfterEndVoltageThreshold = 2.0f;
    return;
}



void IzhikevichTimeDrivenModel::EvaluateSpikeCondition(float previous_V, float * NeuronState, int index, float elpasedTimeInNeuronModelScale){
	if (NeuronState[V_m_index] > 30.0f){
		NeuronState[V_m_index] = this->c;//v
		NeuronState[u_index] += this->d;//u
		State->NewFiredSpike(index);
		this->integration_method->resetState(index);
		this->State->InternalSpikeIndexs[this->State->NInternalSpikeIndexs] = index;
		this->State->NInternalSpikeIndexs++;
	}
}



void IzhikevichTimeDrivenModel::EvaluateDifferentialEquation(float * NeuronState, float * AuxNeuronState, int index, float elapsed_time){
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

	if (NeuronState[V_m_index] <= 30.0f){
		//V
		AuxNeuronState[V_m_index] = 0.04f*NeuronState[V_m_index] * NeuronState[V_m_index] + 5 * NeuronState[V_m_index] + 140 - NeuronState[u_index] + (current*this->inv_c_m);
		//u
		AuxNeuronState[u_index]=a*(b*NeuronState[V_m_index] - NeuronState[u_index]);
	}else{
		//V
		AuxNeuronState[V_m_index]=0;
		//u
		AuxNeuronState[u_index]=0;
	}
}

void IzhikevichTimeDrivenModel::EvaluateTimeDependentEquation(float * NeuronState, int index, int elapsed_time_index){
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

void IzhikevichTimeDrivenModel::Calculate_conductance_exp_values(int index, float elapsed_time){
	//excitatory synapse.
	Set_conductance_exp_values(index, 0, expf(-elapsed_time*this->inv_tau_exc));
	//inhibitory synapse.
	Set_conductance_exp_values(index, 1, expf(-elapsed_time*this->inv_tau_inh));
	//nmda synapse.
	Set_conductance_exp_values(index, 2, expf(-elapsed_time*this->inv_tau_nmda));
}


bool IzhikevichTimeDrivenModel::CheckSynapseType(Interconnection * connection){
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
			cout << "Synapses type " << Type << " of neuron model " << IzhikevichTimeDrivenModel::GetName() << " must receive spikes. The source model generates currents." << endl;
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
				cout << "Synapses type " << Type << " of neuron model " << IzhikevichTimeDrivenModel::GetName() << " must receive current. The source model generates spikes." << endl;
				return false;
			}
		}
	}
	cout << "Neuron model " << IzhikevichTimeDrivenModel::GetName() << " does not support input synapses of type " << Type << ". Just defined " << N_TimeDependentNeuronState << " synapses types." << endl;
	return false;
}

std::map<std::string,boost::any> IzhikevichTimeDrivenModel::GetParameters() const {
	// Return a dictionary with the parameters
	std::map<std::string,boost::any> newMap = TimeDrivenNeuronModel::GetParameters();
	newMap["a"] = boost::any(this->a); // Time scale of recovery variable u (dimensionless)
	newMap["b"] = boost::any(this->b); // Sensitivity of the recovery variable u to the subthreshold fluctuations of the membrane potential v (dimensionless)
	newMap["c"] = boost::any(this->c);  // After-spike reset value of the membrane potential v (dimensionless)
	newMap["d"] = boost::any(this->d); // After-spike reset of the recovery variable u (dimensionless)
	newMap["e_exc"] = boost::any(this->e_exc); // Excitatory reversal potential (mV)
	newMap["e_inh"] = boost::any(this->e_inh); // Inhibitory reversal potential (mV)
	newMap["c_m"] = boost::any(this->c_m); // Membrane capacitance (pF)
	newMap["tau_exc"] = boost::any(this->tau_exc); // AMPA (excitatory) receptor time constant (ms)
	newMap["tau_inh"] = boost::any(this->tau_inh); // GABA (inhibitory) receptor time constant (ms)
	newMap["tau_nmda"] = boost::any(this->tau_nmda); // NMDA (excitatory) receptor time constant (ms)
	return newMap;
}

std::map<std::string, boost::any> IzhikevichTimeDrivenModel::GetSpecificNeuronParameters(int index) const noexcept(false){
	return GetParameters();
}

void IzhikevichTimeDrivenModel::SetParameters(std::map<std::string, boost::any> param_map) noexcept(false){

	// Search for the parameters in the dictionary
	std::map<std::string,boost::any>::iterator it=param_map.find("a");
	if (it!=param_map.end()){
		float new_param = boost::any_cast<float>(it->second);
		this->a = new_param;
		param_map.erase(it);
	}

	it=param_map.find("b");
	if (it!=param_map.end()){
		float new_param = boost::any_cast<float>(it->second);
		this->b = new_param;
		param_map.erase(it);
	}

	it=param_map.find("c");
	if (it!=param_map.end()){
		float new_param = boost::any_cast<float>(it->second);
		this->c = new_param;
		param_map.erase(it);
	}

	it=param_map.find("d");
	if (it!=param_map.end()){
		float new_param = boost::any_cast<float>(it->second);
		this->d = new_param;
		param_map.erase(it);
	}

	it=param_map.find("e_exc");
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

	it=param_map.find("c_m");
	if (it!=param_map.end()){
		float new_param = boost::any_cast<float>(it->second);
		this->c_m = new_param;
		this->inv_c_m = 1./this->c_m;
		param_map.erase(it);
	}

	it=param_map.find("tau_exc");
	if (it!=param_map.end()){
		float new_param = boost::any_cast<float>(it->second);
		this->tau_exc = new_param;
		this->inv_tau_exc = 1.0/this->tau_exc;
		param_map.erase(it);
	}

	it=param_map.find("tau_inh");
	if (it!=param_map.end()){
		float new_param = boost::any_cast<float>(it->second);
		this->tau_inh = new_param;
		this->inv_tau_inh = 1.0/this->tau_inh;
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

IntegrationMethod * IzhikevichTimeDrivenModel::CreateIntegrationMethod(ModelDescription imethodDescription) noexcept(false){
	return IntegrationMethodFactory<IzhikevichTimeDrivenModel>::CreateIntegrationMethod(imethodDescription, (IzhikevichTimeDrivenModel*) this);
}
std::map<std::string,boost::any> IzhikevichTimeDrivenModel::GetDefaultParameters() {
	// Return a dictionary with the parameters
	std::map<std::string,boost::any> newMap = TimeDrivenNeuronModel::GetDefaultParameters<IzhikevichTimeDrivenModel>();
	newMap["a"] = boost::any(0.1f); // Time scale of recovery variable u (dimensionless)
	newMap["b"] = boost::any(0.23f); // Sensitivity of the recovery variable u to the subthreshold fluctuations of the membrane potential v (dimensionless)
	newMap["c"] = boost::any(-65.0f); // After-spike reset value of the membrane potential v (dimensionless)
	newMap["d"] = boost::any(0.2f); // After-spike reset of the recovery variable u (dimensionless)
	newMap["e_exc"] = boost::any(0.0f); // Excitatory reversal potential (mV)
	newMap["e_inh"] = boost::any(-80.0f); // Inhibitory reversal potential (mV)
	newMap["c_m"] = boost::any(100.0f); // Membrane capacitance (pF)
	newMap["tau_exc"] = boost::any(5.0f); // AMPA (excitatory) receptor time constant (ms)
	newMap["tau_inh"] = boost::any(10.0f); // GABA (inhibitory) receptor time constant (ms)
	newMap["tau_nmda"] = boost::any(20.0f); // NMDA (excitatory) receptor time constant (ms)
	return newMap;
}

    NeuronModel* IzhikevichTimeDrivenModel::CreateNeuronModel(ModelDescription nmDescription){
	IzhikevichTimeDrivenModel * nmodel = new IzhikevichTimeDrivenModel();
	nmodel->SetParameters(nmDescription.param_map);
	return nmodel;
}

ModelDescription IzhikevichTimeDrivenModel::ParseNeuronModel(std::string FileName) noexcept(false){
	FILE *fh;
	ModelDescription nmodel;
	nmodel.model_name = IzhikevichTimeDrivenModel::GetName();
	long Currentline = 0L;
	fh=fopen(FileName.c_str(),"rt");
	if(!fh) {
		throw EDLUTFileException(TASK_IZHIKEVICH_TIME_DRIVEN_MODEL_LOAD, ERROR_NEURON_MODEL_OPEN, REPAIR_NEURON_MODEL_NAME, Currentline, FileName.c_str());
	}

	Currentline = 1L;
	skip_comments(fh, Currentline);

	float param;
	if (fscanf(fh, "%f", &param) != 1) {
		throw EDLUTFileException(TASK_IZHIKEVICH_TIME_DRIVEN_MODEL_LOAD, ERROR_IZHIKEVICH_TIME_DRIVEN_MODEL_A, REPAIR_NEURON_MODEL_VALUES, Currentline, FileName.c_str());
	}
	nmodel.param_map["a"] = boost::any(param);

	skip_comments(fh, Currentline);
	if (fscanf(fh, "%f", &param) != 1) {
		throw EDLUTFileException(TASK_IZHIKEVICH_TIME_DRIVEN_MODEL_LOAD, ERROR_IZHIKEVICH_TIME_DRIVEN_MODEL_B, REPAIR_NEURON_MODEL_VALUES, Currentline, FileName.c_str());
	}
	nmodel.param_map["b"] = boost::any(param);

	skip_comments(fh, Currentline);
	if (fscanf(fh, "%f", &param) != 1) {
		throw EDLUTFileException(TASK_IZHIKEVICH_TIME_DRIVEN_MODEL_LOAD, ERROR_IZHIKEVICH_TIME_DRIVEN_MODEL_C, REPAIR_NEURON_MODEL_VALUES, Currentline, FileName.c_str());
	}
	nmodel.param_map["c"] = boost::any(param);

	skip_comments(fh, Currentline);
	if (fscanf(fh, "%f", &param) != 1) {
		throw EDLUTFileException(TASK_IZHIKEVICH_TIME_DRIVEN_MODEL_LOAD, ERROR_IZHIKEVICH_TIME_DRIVEN_MODEL_D, REPAIR_NEURON_MODEL_VALUES, Currentline, FileName.c_str());
	}
	nmodel.param_map["d"] = boost::any(param);

	skip_comments(fh, Currentline);
	if (fscanf(fh, "%f", &param) != 1) {
		throw EDLUTFileException(TASK_IZHIKEVICH_TIME_DRIVEN_MODEL_LOAD, ERROR_IZHIKEVICH_TIME_DRIVEN_MODEL_E_EXC, REPAIR_NEURON_MODEL_VALUES, Currentline, FileName.c_str());
	}
	nmodel.param_map["e_exc"] = boost::any(param);

	skip_comments(fh, Currentline);
	if (fscanf(fh, "%f", &param) != 1) {
		throw EDLUTFileException(TASK_IZHIKEVICH_TIME_DRIVEN_MODEL_LOAD, ERROR_IZHIKEVICH_TIME_DRIVEN_MODEL_E_INH, REPAIR_NEURON_MODEL_VALUES, Currentline, FileName.c_str());
	}
	nmodel.param_map["e_inh"] = boost::any(param);

	skip_comments(fh, Currentline);
	if (fscanf(fh, "%f", &param) != 1 || param <= 0.0f) {
		throw EDLUTFileException(TASK_IZHIKEVICH_TIME_DRIVEN_MODEL_LOAD, ERROR_IZHIKEVICH_TIME_DRIVEN_MODEL_C_M, REPAIR_NEURON_MODEL_VALUES, Currentline, FileName.c_str());
	}
	nmodel.param_map["c_m"] = boost::any(param);

	skip_comments(fh, Currentline);
	if (fscanf(fh, "%f", &param) != 1 || param <= 0.0f) {
		throw EDLUTFileException(TASK_IZHIKEVICH_TIME_DRIVEN_MODEL_LOAD, ERROR_IZHIKEVICH_TIME_DRIVEN_MODEL_TAU_EXC, REPAIR_NEURON_MODEL_VALUES, Currentline, FileName.c_str());
	}
	nmodel.param_map["tau_exc"] = boost::any(param);

	skip_comments(fh, Currentline);
	if (fscanf(fh, "%f", &param) != 1 || param <= 0.0f) {
		throw EDLUTFileException(TASK_IZHIKEVICH_TIME_DRIVEN_MODEL_LOAD, ERROR_IZHIKEVICH_TIME_DRIVEN_MODEL_TAU_INH, REPAIR_NEURON_MODEL_VALUES, Currentline, FileName.c_str());
	}
	nmodel.param_map["tau_inh"] = boost::any(param);

	skip_comments(fh, Currentline);
	if (fscanf(fh, "%f", &param) != 1) {
		throw EDLUTFileException(TASK_IZHIKEVICH_TIME_DRIVEN_MODEL_LOAD, ERROR_IZHIKEVICH_TIME_DRIVEN_MODEL_TAU_NMDA, REPAIR_NEURON_MODEL_VALUES, Currentline, FileName.c_str());
	}
	nmodel.param_map["tau_nmda"] = boost::any(param);

	skip_comments(fh, Currentline);
	try {
		ModelDescription intMethodDescription = TimeDrivenNeuronModel::ParseIntegrationMethod<IzhikevichTimeDrivenModel>(fh, Currentline);
		nmodel.param_map["int_meth"] = boost::any(intMethodDescription);
	} catch (EDLUTException exc) {
		throw EDLUTFileException(exc, Currentline, FileName.c_str());
	}

	nmodel.param_map["name"] = boost::any(IzhikevichTimeDrivenModel::GetName());

	fclose(fh);

	return nmodel;
}

std::string IzhikevichTimeDrivenModel::GetName(){
	return "IzhikevichTimeDrivenModel";
}

std::map<std::string, std::string> IzhikevichTimeDrivenModel::GetNeuronModelInfo() {
	// Return a dictionary with the parameters
	std::map<std::string, std::string> newMap;
	newMap["info"] = std::string("CPU Time-driven Izhikevich (Izhikevich) neuron model with two differential equations(membrane potential (v) and membrane recovery (u)) and four types of input synapses: AMPA (excitatory), GABA (inhibitory), NMDA (excitatory) and external input current (set on pA)");
	newMap["a"] = std::string("Time scale of recovery variable u (dimensionless)");
	newMap["b"] = std::string("Sensitivity of the recovery variable u to the subthreshold fluctuations of the membrane potential v (dimensionless)");
	newMap["c"] = std::string("After-spike reset value of the membrane potential v (dimensionless)");
	newMap["d"] = std::string("After-spike reset of the recovery variable u (dimensionless)");
	newMap["e_exc"] = std::string("Excitatory reversal potential (mV)");
	newMap["e_inh"] = std::string("Inhibitory reversal potential (mV)");
	newMap["c_m"] = std::string("Membrane capacitance (pF)");
	newMap["tau_exc"] = std::string("AMPA (excitatory) receptor time constant (ms)");
	newMap["tau_inh"] = std::string("GABA (inhibitory) receptor time constant (ms)");
	newMap["tau_nmda"] = std::string("NMDA (excitatory) receptor time constant (ms)");
	newMap["int_meth"] = std::string("Integraton method dictionary (from the list of available integration methods)");

	return newMap;
}
