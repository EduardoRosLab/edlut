/***************************************************************************
 *                           TimeDrivenPurkinjeCell.cpp                    *
 *                           -------------------                           *
 * copyright            : (C) 2019 by Richard Carrillo, Niceto Luque and   *
						  Francisco Naveros								   *
 * email                : rcarrillo@ugr.es, nluque@ugr.es and			   *
						  fnaveros@ugr.es    							   *
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 3 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

#include "neuron_model/TimeDrivenPurkinjeCell.h"
#include "neuron_model/VectorNeuronState.h"
#include "neuron_model/CurrentSynapseModel.h"
#include "simulation/ExponentialTable.h"
#include "spike/Neuron.h"
#include "spike/Interconnection.h"

#include "integration_method/IntegrationMethodFactory.h"


const float TimeDrivenPurkinjeCell::Max_V=35.0f;
const float TimeDrivenPurkinjeCell::Min_V=-100.0f;

const float TimeDrivenPurkinjeCell::aux=(TimeDrivenPurkinjeCell::TableSize-1)/( TimeDrivenPurkinjeCell::Max_V- TimeDrivenPurkinjeCell::Min_V);

float * TimeDrivenPurkinjeCell::channel_values=Generate_channel_values();

void TimeDrivenPurkinjeCell::Generate_g_nmda_inf_values(){
	auxNMDA = (TableSizeNMDA - 1) / (e_exc - e_inh);
	for (int i = 0; i<TableSizeNMDA; i++){
		float V = e_inh + ((e_exc - e_inh)*i) / (TableSizeNMDA - 1);

		//g_nmda_inf
		g_nmda_inf_values[i] = 1.0f / (1.0f + exp(-0.062f*V)*(1.2f / 3.57f));
	}
}


float TimeDrivenPurkinjeCell::Get_g_nmda_inf(float V_m){
	int position = int((V_m - e_inh)*auxNMDA);
		if(position<0){
			position=0;
		}
		else if (position>(TableSizeNMDA - 1)){
			position = TableSizeNMDA - 1;
		}
		return g_nmda_inf_values[position];
}


void TimeDrivenPurkinjeCell::InitializeCurrentSynapsis(int N_neurons){
	this->CurrentSynapsis = new CurrentSynapseModel(N_neurons);
}


//this neuron model is implemented in a milisecond scale.
TimeDrivenPurkinjeCell::TimeDrivenPurkinjeCell(): TimeDrivenNeuronModel(MilisecondScale), g_leak(0.02f),
		g_Ca(0.001f), g_M(0.75f), cylinder_length_of_the_soma(0.0015f), radius_of_the_soma(0.0008f), area(3.141592f*0.0015f*2.0f*0.0008f),
		inv_area(1.0f/(3.141592f*0.0015f*2.0f*0.0008f)), c_m(0.95f), inv_c_m(1.0f/0.95f), spk_peak(31.0)
		{
	std::map<std::string, boost::any> param_map = TimeDrivenPurkinjeCell::GetDefaultParameters();
	param_map["name"] = TimeDrivenPurkinjeCell::GetName();
	this->SetParameters(param_map);

	this->State = (VectorNeuronState *) new VectorNeuronState(N_NeuronStateVariables, true);
}

TimeDrivenPurkinjeCell::~TimeDrivenPurkinjeCell(void)
{
}

VectorNeuronState * TimeDrivenPurkinjeCell::InitializeState(){
	return this->GetVectorNeuronState();
}


InternalSpike * TimeDrivenPurkinjeCell::ProcessInputSpike(Interconnection * inter, double time){
	// Add the effect of the input spike
	this->GetVectorNeuronState()->IncrementStateVariableAtCPU(inter->GetTargetNeuronModelIndex(), N_DifferentialNeuronState + inter->GetType(), inter->GetWeight());

	return 0;
}


void TimeDrivenPurkinjeCell::ProcessInputCurrent(Interconnection * inter, Neuron * target, float current){
	//Update the external current in the corresponding input synapse of type EXT_I (defined in pA).
	this->CurrentSynapsis->SetInputCurrent(target->GetIndex_VectorNeuronState(), inter->GetSubindexType(), current);

	//Update the total external current that receive the neuron coming from all its EXT_I synapsis (defined in pA).
	float total_ext_I = this->CurrentSynapsis->GetTotalCurrent(target->GetIndex_VectorNeuronState());
	State->SetStateVariableAt(target->GetIndex_VectorNeuronState(), EXT_I_index, total_ext_I);
}

bool TimeDrivenPurkinjeCell::UpdateState(int index, double CurrentTime){
	//Reset the number of internal spikes in this update period
	this->State->NInternalSpikeIndexs = 0;

	this->integration_method->NextDifferentialEquationValues();

	this->CheckValidIntegration(CurrentTime, this->integration_method->GetValidIntegrationVariable());

	return false;
}


enum NeuronModelOutputActivityType TimeDrivenPurkinjeCell::GetModelOutputActivityType(){
	return OUTPUT_SPIKE;
}

enum NeuronModelInputActivityType TimeDrivenPurkinjeCell::GetModelInputActivityType(){
	return INPUT_SPIKE_AND_CURRENT;
}

ostream & TimeDrivenPurkinjeCell::PrintInfo(ostream & out){
	out << "- Time-Driven Purkinje Cell Model: " << TimeDrivenPurkinjeCell::GetName() << endl;
	out << "\tExcitatory reversal potential (e_exc): " << this->e_exc << "mV" << endl;
	out << "\tInhibitory reversal potential (e_inh): " << this->e_inh << "mV" << endl;
	out << "\tEffective threshold potential (v_thr): " << this->v_thr << "mV" << endl;
	out << "\tEffective leak potential (e_leak): " << this->e_leak << "mV" << endl;
	out << "\tAMPA (excitatory) receptor time constant (tau_exc): " << this->tau_exc << "ms" << endl;
	out << "\tGABA (inhibitory) receptor time constant (tau_inh): " << this->tau_inh << "ms" << endl;
	out << "\tNMDA (excitatory) receptor time constant (tau_nmda): " << this->tau_nmda << "ms" << endl;
	out << "\tRefractory period (tau_ref): " << this->tau_ref << "ms" << endl;

	this->integration_method->PrintInfo(out);
	return out;
}



void TimeDrivenPurkinjeCell::InitializeStates(int N_neurons, int OpenMPQueueIndex){
	//Initialize neural state variables.
	float * values=Get_channel_values(e_leak);
	float alpha_ca=values[0];
	float inv_tau_ca=values[1];
	float alpha_M=values[2];
	float inv_tau_M=values[3];

	//c_inf
	float c_inf=alpha_ca/inv_tau_ca;

	//M_inf
	float M_inf=alpha_M/inv_tau_M;

	float initialization[] = {e_leak, c_inf, M_inf, 0.0f, 0.0f, 0.0f, 0.0f};
	State->InitializeStates(N_neurons, initialization);

	//Initialize integration method state variables.
	this->integration_method->SetBifixedStepParameters((e_leak + 3 * v_thr) / 4, (e_leak + 3 * v_thr) / 4, 2.0f);
	this->integration_method->Calculate_conductance_exp_values();
	this->integration_method->InitializeStates(N_neurons, initialization);

	//Initialize the array that stores the number of input current synapses for each neuron in the model
	InitializeCurrentSynapsis(N_neurons);
}

void TimeDrivenPurkinjeCell::GetBifixedStepParameters(float & startVoltageThreshold, float & endVoltageThreshold, float & timeAfterEndVoltageThreshold){
	startVoltageThreshold = (e_leak+3*v_thr)/4;
	endVoltageThreshold = (e_leak+3*v_thr)/4;
	timeAfterEndVoltageThreshold = 2.0f;
	return;
}

void TimeDrivenPurkinjeCell::EvaluateSpikeCondition(float previous_V, float * NeuronState, int index, float elapsedTimeInNeuronModelScale){
	if (NeuronState[V_m_index] >= this->v_thr && previous_V < this->v_thr){
		State->NewFiredSpike(index);
		this->State->InternalSpikeIndexs[this->State->NInternalSpikeIndexs] = index;
		this->State->NInternalSpikeIndexs++;
	}

	double last_spike = State->GetLastSpikeTime(index) * this->time_scale;

	if(last_spike < tau_ref){
		if(last_spike <= tau_ref_0_5){
			NeuronState[V_m_index]=v_thr+(spk_peak-v_thr)*(last_spike*inv_tau_ref_0_5);
		}else{
			NeuronState[V_m_index]=spk_peak-(spk_peak-e_leak)*((last_spike-tau_ref_0_5)*inv_tau_ref_0_5);
		}
	}else if((last_spike - tau_ref)<elapsedTimeInNeuronModelScale){
		NeuronState[V_m_index]=e_leak;
	}
}


void TimeDrivenPurkinjeCell::EvaluateDifferentialEquation(float * NeuronState, float * AuxNeuronState, int index, float elapsed_time){
	float V=NeuronState[V_m_index];
	float ca=NeuronState[Ca_index];
	float M=NeuronState[M_index];
	float last_spike=this->time_scale*State->GetLastSpikeTime(index);

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

	//We must transform the external current defined in pA to uA/cm^2
	current = current * (1e-6 * inv_area);


	//V
	if(last_spike >= tau_ref){
		AuxNeuronState[V_m_index]=(-g_leak*(V+70.0f)-g_Ca*ca*ca*(V-125.0f)-g_M*M*(V+95.0f) + current)*inv_c_m;
	}else if(last_spike <= tau_ref_0_5){
		AuxNeuronState[V_m_index]=(spk_peak-v_thr)*inv_tau_ref_0_5;
	}else{
		AuxNeuronState[V_m_index]=(e_leak-spk_peak)*inv_tau_ref_0_5;
	}

	float * values=Get_channel_values(V);

	//ca
	float alpha_ca=values[0];
	float inv_tau_ca=values[1];
	AuxNeuronState[Ca_index]=alpha_ca - ca*inv_tau_ca;

	//M
	float alpha_M=values[2];
	float inv_tau_M=values[3];
	AuxNeuronState[M_index]=alpha_M - M*inv_tau_M;


}



void TimeDrivenPurkinjeCell::EvaluateTimeDependentEquation(float * NeuronState, int index, int elapsed_time_index){
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

void TimeDrivenPurkinjeCell::Calculate_conductance_exp_values(int index, float elapsed_time){
	//excitatory synapse.
	Set_conductance_exp_values(index, 0, expf(-elapsed_time*this->inv_tau_exc));
	//inhibitory synapse.
	Set_conductance_exp_values(index, 1, expf(-elapsed_time*this->inv_tau_inh));
	//nmda synapse.
	Set_conductance_exp_values(index, 2, expf(-elapsed_time*this->inv_tau_nmda));
}


bool TimeDrivenPurkinjeCell::CheckSynapseType(Interconnection * connection){
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
			cout << "Synapses type " << Type << " of neuron model " << TimeDrivenPurkinjeCell::GetName() << " must receive spikes. The source model generates currents." << endl;
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
				cout << "Synapses type " << Type << " of neuron model " << TimeDrivenPurkinjeCell::GetName() << " must receive current. The source model generates spikes." << endl;
				return false;
			}
		}
	}
	cout << "Neuron model " << TimeDrivenPurkinjeCell::GetName() << " does not support input synapses of type " << Type << ". Just defined " << N_TimeDependentNeuronState << " synapses types." << endl;
	return false;
}

std::map<std::string,boost::any> TimeDrivenPurkinjeCell::GetParameters() const {
	// Return a dictionary with the parameters
	std::map<std::string,boost::any> newMap = TimeDrivenNeuronModel::GetParameters();
	newMap["e_exc"] = boost::any(this->e_exc); // Excitatory reversal potential (mV)
	newMap["e_inh"] = boost::any(this->e_inh); // Inhibitory reversal potential (mV)
	newMap["v_thr"] = boost::any(this->v_thr); // Effective threshold potential (mV)
	newMap["e_leak"] = boost::any(this->e_leak); // Effective leak potential (mV)
	newMap["tau_exc"] = boost::any(this->tau_exc); // AMPA (excitatory) receptor time constant (ms)
	newMap["tau_inh"] = boost::any(this->tau_inh); // GABA (inhibitory) receptor time constant (ms)
	newMap["tau_nmda"] = boost::any(this->tau_nmda); // NMDA (excitatory) receptor time constant (ms)
	newMap["tau_ref"] = boost::any(this->tau_ref); // Refractory period (ms)
	return newMap;
}

std::map<std::string, boost::any> TimeDrivenPurkinjeCell::GetSpecificNeuronParameters(int index) const noexcept(false){
	return GetParameters();
}

void TimeDrivenPurkinjeCell::SetParameters(std::map<std::string, boost::any> param_map) noexcept(false){

	// Search for the parameters in the dictionary
	std::map<std::string, boost::any>::iterator it = param_map.find("e_exc");
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

	it=param_map.find("v_thr");
	if (it!=param_map.end()){
		float new_param = boost::any_cast<float>(it->second);
		this->v_thr = new_param;
		param_map.erase(it);
	}

	it=param_map.find("e_leak");
	if (it!=param_map.end()){
		float new_param = boost::any_cast<float>(it->second);
		this->e_leak = new_param;
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

	it=param_map.find("tau_nmda");
	if (it!=param_map.end()){
		float new_param = boost::any_cast<float>(it->second);
		this->tau_nmda = new_param;
		this->inv_tau_nmda = 1.0/new_param;
		param_map.erase(it);
	}

	it=param_map.find("tau_ref");
	if (it!=param_map.end()){
		float new_param = boost::any_cast<float>(it->second);
		this->tau_ref = new_param;
		this->tau_ref_0_5 = new_param / 2;
		this->inv_tau_ref_0_5 = 1.0/this->tau_ref_0_5;
		param_map.erase(it);
	}


	// Search for the parameters in the dictionary
	TimeDrivenNeuronModel::SetParameters(param_map);

	//Set the new g_nmda_inf values based on the e_exc and e_inh parameters
	Generate_g_nmda_inf_values();

	return;
}


IntegrationMethod * TimeDrivenPurkinjeCell::CreateIntegrationMethod(ModelDescription imethodDescription) noexcept(false){
	return IntegrationMethodFactory<TimeDrivenPurkinjeCell>::CreateIntegrationMethod(imethodDescription, (TimeDrivenPurkinjeCell*) this);
}


std::map<std::string,boost::any> TimeDrivenPurkinjeCell::GetDefaultParameters() {
	// Return a dictionary with the parameters
	std::map<std::string, boost::any> newMap = TimeDrivenNeuronModel::GetDefaultParameters<TimeDrivenPurkinjeCell>();
	newMap["e_exc"] = boost::any(0.0f); // Excitatory reversal potential (mV)
	newMap["e_inh"] = boost::any(-80.0f); // Inhibitory reversal potential (mV)
	newMap["v_thr"] = boost::any(-35.0f); // Effective threshold potential (mV)
	newMap["e_leak"] = boost::any(-70.0f); // Effective leak potential (mV)
	newMap["tau_exc"] = boost::any(1.0f); // AMPA (excitatory) receptor time constant (ms)
	newMap["tau_inh"] = boost::any(2.0f); // GABA (inhibitory) receptor time constant (ms)
	newMap["tau_nmda"] = boost::any(20.0f); // NMDA (excitatory) receptor time constant (ms)
	newMap["tau_ref"] = boost::any(1.35f); // Refractory period (ms)
	return newMap;
}

NeuronModel* TimeDrivenPurkinjeCell::CreateNeuronModel(ModelDescription nmDescription){
	TimeDrivenPurkinjeCell * nmodel = new TimeDrivenPurkinjeCell();
	nmodel->SetParameters(nmDescription.param_map);
	return nmodel;
}

ModelDescription TimeDrivenPurkinjeCell::ParseNeuronModel(std::string FileName) noexcept(false){
	FILE *fh;
	ModelDescription nmodel;
	nmodel.model_name = TimeDrivenPurkinjeCell::GetName();
	long Currentline = 0L;
	fh=fopen(FileName.c_str(),"rt");
	if(!fh) {
		throw EDLUTFileException(TASK_TIME_DRIVEN_PURKINJE_CELL_LOAD, ERROR_NEURON_MODEL_OPEN, REPAIR_NEURON_MODEL_NAME, Currentline, FileName.c_str());
	}

	Currentline = 1L;
	float param;

	skip_comments(fh, Currentline);
	if (fscanf(fh, "%f", &param) != 1) {
		throw EDLUTFileException(TASK_TIME_DRIVEN_PURKINJE_CELL_LOAD, ERROR_TIME_DRIVEN_PURKINJE_CELL_E_EXC, REPAIR_NEURON_MODEL_VALUES, Currentline, FileName.c_str());
	}
	nmodel.param_map["e_exc"] = boost::any(param);

	skip_comments(fh, Currentline);
	if (fscanf(fh, "%f", &param) != 1) {
		throw EDLUTFileException(TASK_TIME_DRIVEN_PURKINJE_CELL_LOAD, ERROR_TIME_DRIVEN_PURKINJE_CELL_E_INH, REPAIR_NEURON_MODEL_VALUES, Currentline, FileName.c_str());
	}
	nmodel.param_map["e_inh"] = boost::any(param);

	skip_comments(fh, Currentline);
	if (fscanf(fh, "%f", &param) != 1) {
		throw EDLUTFileException(TASK_TIME_DRIVEN_PURKINJE_CELL_LOAD, ERROR_TIME_DRIVEN_PURKINJE_CELL_V_THR, REPAIR_NEURON_MODEL_VALUES, Currentline, FileName.c_str());
	}
	nmodel.param_map["v_thr"] = boost::any(param);

	skip_comments(fh, Currentline);
	if (fscanf(fh, "%f", &param) != 1) {
		throw EDLUTFileException(TASK_TIME_DRIVEN_PURKINJE_CELL_LOAD, ERROR_TIME_DRIVEN_PURKINJE_CELL_E_LEAK, REPAIR_NEURON_MODEL_VALUES, Currentline, FileName.c_str());
	}
	nmodel.param_map["e_leak"] = boost::any(param);

	skip_comments(fh, Currentline);
	if (fscanf(fh, "%f", &param) != 1 || param <= 0.0f) {
		throw EDLUTFileException(TASK_TIME_DRIVEN_PURKINJE_CELL_LOAD, ERROR_TIME_DRIVEN_PURKINJE_CELL_TAU_EXC, REPAIR_NEURON_MODEL_VALUES, Currentline, FileName.c_str());
	}
	nmodel.param_map["tau_exc"] = boost::any(param);

	skip_comments(fh, Currentline);
	if (fscanf(fh, "%f", &param) != 1 || param <= 0.0f) {
		throw EDLUTFileException(TASK_TIME_DRIVEN_PURKINJE_CELL_LOAD, ERROR_TIME_DRIVEN_PURKINJE_CELL_TAU_INH, REPAIR_NEURON_MODEL_VALUES, Currentline, FileName.c_str());
	}
	nmodel.param_map["tau_inh"] = boost::any(param);

	skip_comments(fh, Currentline);
	if (fscanf(fh, "%f", &param) != 1 || param <= 0.0f) {
		throw EDLUTFileException(TASK_TIME_DRIVEN_PURKINJE_CELL_LOAD, ERROR_TIME_DRIVEN_PURKINJE_CELL_TAU_NMDA, REPAIR_NEURON_MODEL_VALUES, Currentline, FileName.c_str());
	}
	nmodel.param_map["tau_nmda"] = boost::any(param);

	skip_comments(fh, Currentline);
	if (fscanf(fh, "%f", &param) != 1 || param <= 0.0f) {
		throw EDLUTFileException(TASK_TIME_DRIVEN_PURKINJE_CELL_LOAD, ERROR_TIME_DRIVEN_PURKINJE_CELL_TAU_REF, REPAIR_NEURON_MODEL_VALUES, Currentline, FileName.c_str());
	}
	nmodel.param_map["tau_ref"] = boost::any(param);


	skip_comments(fh, Currentline);
	try {
		ModelDescription intMethodDescription = TimeDrivenNeuronModel::ParseIntegrationMethod<TimeDrivenPurkinjeCell>(fh, Currentline);
		nmodel.param_map["int_meth"] = boost::any(intMethodDescription);
	} catch (EDLUTException exc) {
		throw EDLUTFileException(exc, Currentline, FileName.c_str());
	}

	nmodel.param_map["name"] = boost::any(TimeDrivenPurkinjeCell::GetName());

	fclose(fh);

	return nmodel;
}

std::string TimeDrivenPurkinjeCell::GetName(){
	return "TimeDrivenPurkinjeCell";
}

std::map<std::string, std::string> TimeDrivenPurkinjeCell::GetNeuronModelInfo() {
	// Return a dictionary with the parameters
	std::map<std::string, std::string> newMap;
	newMap["info"] = std::string("CPU Time-driven HH neuron model representing a Purkinje cell with three differential equations(membrane potential (v), calcium (ca) and Muscariny (M) channels) and four types of input synapses: AMPA (excitatory), GABA (inhibitory), NMDA (excitatory) and external input current (set on pA)");
	newMap["e_exc"] = std::string("Excitatory reversal potential (mV)");
	newMap["e_inh"] = std::string("Inhibitory reversal potential (mV)");
	newMap["v_thr"] = std::string("Effective threshold potential (mV)");
	newMap["e_leak"] = std::string("Effective leak potential (mV)");
	newMap["tau_exc"] = std::string("AMPA (excitatory) receptor time constant (ms)");
	newMap["tau_inh"] = std::string("GABA (inhibitory) receptor time constant (ms)");
	newMap["tau_nmda"] = std::string("NMDA (excitatory) receptor time constant (ms)");
	newMap["tau_ref"] = std::string("Refractory period (ms)");
	newMap["int_meth"] = std::string("Integraton method dictionary (from the list of available integration methods)");

	return newMap;
}
