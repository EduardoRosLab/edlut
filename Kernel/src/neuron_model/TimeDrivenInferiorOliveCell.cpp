/***************************************************************************
 *                           TimeDrivenInferiorOliveCell.cpp               *
 *                           -------------------                           *
 * copyright            : (C) 2019 by Niceto Luque and Francisco Naveros   *
 * email                : nluque@ugr.es and	fnaveros@ugr.es  			   *
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 3 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

#include "neuron_model/TimeDrivenInferiorOliveCell.h"
#include "neuron_model/VectorNeuronState.h"
#include "neuron_model/CurrentSynapseModel.h"

#include "spike/Neuron.h"
#include "spike/Interconnection.h"

#include "integration_method/IntegrationMethodFactory.h"

#include "../../include/simulation/ExponentialTable.h"

void TimeDrivenInferiorOliveCell::Generate_g_nmda_inf_values(){
	auxNMDA = (TableSizeNMDA - 1) / (e_exc - e_inh);
	for (int i = 0; i<TableSizeNMDA; i++){
		float V = e_inh + ((e_exc - e_inh)*i) / (TableSizeNMDA - 1);

		//g_nmda_inf
		g_nmda_inf_values[i] = 1.0f / (1.0f + exp(-62.0f*V)*(1.2f / 3.57f));
	}
}


float TimeDrivenInferiorOliveCell::Get_g_nmda_inf(float V_m){
	int position = int((V_m - e_inh)*auxNMDA);
		if(position<0){
			position=0;
		}
		else if (position>(TableSizeNMDA - 1)){
			position = TableSizeNMDA - 1;
		}
		return g_nmda_inf_values[position];
}


void TimeDrivenInferiorOliveCell::InitializeCurrentSynapsis(int N_neurons){
	this->CurrentSynapsis = new CurrentSynapseModel(N_neurons);
}


//this neuron model is implemented in a milisecond scale.
TimeDrivenInferiorOliveCell::TimeDrivenInferiorOliveCell(): TimeDrivenNeuronModel(MilisecondScale), EXC(false), INH(false),
	NMDA(false), EXT_I(false), I_COU(false), spk_peak(31.0f), N_coupling_synapses(0), index_coupling_synapses(0),
	Weight_coupling_synapses(0), Potential_coupling_synapses(0){
	std::map<std::string, boost::any> param_map = TimeDrivenInferiorOliveCell::GetDefaultParameters();
	param_map["name"] = TimeDrivenInferiorOliveCell::GetName();
	this->SetParameters(param_map);

	this->State = (VectorNeuronState *) new VectorNeuronState(N_NeuronStateVariables, true);
}

TimeDrivenInferiorOliveCell::~TimeDrivenInferiorOliveCell(void)
{
	if (N_coupling_synapses != 0){
		for (int i = 0; i < this->GetVectorNeuronState()->GetSizeState(); i++){
			if (N_coupling_synapses[i]>0){
				delete Weight_coupling_synapses[i];
				delete Potential_coupling_synapses[i];
			}
		}
		delete N_coupling_synapses;
		delete index_coupling_synapses;
		delete Weight_coupling_synapses;
		delete Potential_coupling_synapses;
	}
}

VectorNeuronState * TimeDrivenInferiorOliveCell::InitializeState(){
	return this->GetVectorNeuronState();
}


InternalSpike * TimeDrivenInferiorOliveCell::ProcessInputSpike(Interconnection * inter, double time){
	// Add the effect of the input spike
	this->GetVectorNeuronState()->IncrementStateVariableAtCPU(inter->GetTargetNeuronModelIndex(), N_DifferentialNeuronState + inter->GetType(), 1e-6f*inter->GetWeight());

	return 0;
}


void TimeDrivenInferiorOliveCell::ProcessInputCurrent(Interconnection * inter, Neuron * target, float current){
	//Update the external current in the corresponding input synapse of type EXT_I (defined in pA).
	this->CurrentSynapsis->SetInputCurrent(target->GetIndex_VectorNeuronState(), inter->GetSubindexType(), current);

	//Update the total external current that receive the neuron coming from all its EXT_I synapsis (defined in pA).
	float total_ext_I = this->CurrentSynapsis->GetTotalCurrent(target->GetIndex_VectorNeuronState());
	State->SetStateVariableAt(target->GetIndex_VectorNeuronState(), EXT_I_index, total_ext_I);
}


bool TimeDrivenInferiorOliveCell::UpdateState(int index, double CurrentTime){
	////ELECTRICAL COUPLING CURRENTS.

	#pragma omp barrier
	if (I_COU){
		for (int i = 0; i < this->GetVectorNeuronState()->GetSizeState(); i++){
			float current = 0;
			float * NeuronState = this->GetVectorNeuronState()->GetStateVariableAt(i);
			for (int j = 0; j < this->N_coupling_synapses[i]; j++){
				float source_V = Potential_coupling_synapses[i][j][V_m_index];
				float V = source_V - NeuronState[V_m_index];
				float weight = Weight_coupling_synapses[i][j];
				//			current += weight*V*(0.6f*exp(-V*V * 0.0004) + 0.4);                                     //current += weight*V*(0.6f*exp(-V*V / (50.0f*50.0f)) + 0.4);
				current += weight*V*(0.6f*ExponentialTable::GetResult(-V*V * 0.0004) + 0.4);      //current += weight*V*(0.6f*ExponentialTable::GetResult(-V*V / (50.0f*50.0f)) + 0.4);
			}
			NeuronState[I_COU_index] = 1e-6f*current;
		}
	}
	#pragma omp barrier
	////////////////////////////////////////

	//Reset the number of internal spikes in this update period
	this->State->NInternalSpikeIndexs = 0;

	this->integration_method->NextDifferentialEquationValues();

	this->CheckValidIntegration(CurrentTime, this->integration_method->GetValidIntegrationVariable());

	return false;
}


enum NeuronModelOutputActivityType TimeDrivenInferiorOliveCell::GetModelOutputActivityType(){
	return OUTPUT_SPIKE;
}


enum NeuronModelInputActivityType TimeDrivenInferiorOliveCell::GetModelInputActivityType(){
	return INPUT_SPIKE_AND_CURRENT;
}


ostream & TimeDrivenInferiorOliveCell::PrintInfo(ostream & out){
	out << "- Time Driven Inferior Olive Model: " << TimeDrivenInferiorOliveCell::GetName() << endl;
	out << "\tExcitatory reversal potential (e_exc): " << this->e_exc << "mV" << endl;
	out << "\tInhibitory reversal potential (e_inh): " << this->e_inh << "mV" << endl;
	out << "\tEffective leak potential (e_leak): " << this->e_leak << "mV" << endl;
	out << "\tEffective threshold potential (v_thr): " << this->v_thr << "mV" << endl;
	out << "\tMembrane capacitance (c_m): " << this->c_m << "uF/cm^2" << endl;
	out << "\tAMPA (excitatory) receptor time constant (tau_exc): " << this->tau_exc << "ms" << endl;
	out << "\tGABA (inhibitory) receptor time constant (tau_inh): " << this->tau_inh << "ms" << endl;
	out << "\tNMDA (excitatory) receptor time constant (tau_nmda): " << this->tau_nmda << "ms" << endl;
	out << "\tRefractory period (tau_ref): " << this->tau_ref << "ms" << endl;
	out << "\tLeak conductance (g_leak): " << this->g_leak << "mS/cm^2" << endl;
	out << "\tCell area (area): " << this->area << "cm^2" << endl;

	this->integration_method->PrintInfo(out);
	return out;
}



void TimeDrivenInferiorOliveCell::InitializeStates(int N_neurons, int OpenMPQueueIndex){
	//Initialize neural state variables.
	float initialization[] = {e_leak,0.0f,0.0f,0.0f,0.0f,0.0f};
	State->InitializeStates(N_neurons, initialization);

	//Initialize integration method state variables.
	this->integration_method->SetBifixedStepParameters((e_leak + v_thr) / 2, e_leak + 5.0, 0.0);
	this->integration_method->Calculate_conductance_exp_values();
	this->integration_method->InitializeStates(N_neurons, initialization);

	//Initilize the structure to compute the current derived of the electrical coupling between neurons.
	N_coupling_synapses=new int[N_neurons]();
	index_coupling_synapses = new int[N_neurons]();
    Weight_coupling_synapses=(float**)new float*[N_neurons];
    Potential_coupling_synapses=(float***)new float**[N_neurons];

	//Initialize the array that stores the number of input current synapses for each neuron in the model
	InitializeCurrentSynapsis(N_neurons);
}


void TimeDrivenInferiorOliveCell::GetBifixedStepParameters(float & startVoltageThreshold, float & endVoltageThreshold, float & timeAfterEndVoltageThreshold){
	startVoltageThreshold = (e_leak + v_thr) / 2;
	endVoltageThreshold = e_leak + 5.0;
	timeAfterEndVoltageThreshold = 0.0f;
	return;
}


void TimeDrivenInferiorOliveCell::EvaluateSpikeCondition(float previous_V, float * NeuronState, int index, float elapsedTimeInNeuronModelScale){
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


void TimeDrivenInferiorOliveCell::EvaluateDifferentialEquation(float * NeuronState, float * AuxNeuronState, int index, float elapsed_time){
	float V=NeuronState[V_m_index];
	float last_spike=this->time_scale*State->GetLastSpikeTime(index);

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
	current += NeuronState[EXT_I_index] + NeuronState[I_COU_index]; // (defined in pA).

	//V
	if(last_spike >= tau_ref){
		AuxNeuronState[V_m_index]=(g_leak*(this->e_leak - V) + current * inv_area)*inv_c_m;
	}else if(last_spike <= tau_ref_0_5){
		AuxNeuronState[V_m_index]=(spk_peak-v_thr)*inv_tau_ref_0_5;
	}else{
		AuxNeuronState[V_m_index]=(e_leak-spk_peak)*inv_tau_ref_0_5;
	}
}

void TimeDrivenInferiorOliveCell::EvaluateTimeDependentEquation(float * NeuronState, int index, int elapsed_time_index){
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

void TimeDrivenInferiorOliveCell::Calculate_conductance_exp_values(int index, float elapsed_time){
	//excitatory synapse.
	Set_conductance_exp_values(index, 0, expf(-elapsed_time*this->inv_tau_exc));
	//inhibitory synapse.
	Set_conductance_exp_values(index, 1, expf(-elapsed_time*this->inv_tau_inh));
	//nmda synapse.
	Set_conductance_exp_values(index, 2, expf(-elapsed_time*this->inv_tau_nmda));
}


bool TimeDrivenInferiorOliveCell::CheckSynapseType(Interconnection * connection){
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
		if (Type == 4){
			I_COU = true;
		}

		NeuronModel * model = connection->GetSource()->GetNeuronModel();
		//Synapse types that process input spikes
		if (Type < N_TimeDependentNeuronState - 2){
			if (model->GetModelOutputActivityType() == OUTPUT_SPIKE){
				return true;
			}
			else{
			cout << "Synapses type " << Type << " of neuron model " << TimeDrivenInferiorOliveCell::GetName() << " must receive spikes. The source model generates currents." << endl;
				return false;
			}
		}
		//Synapse types that process input current
		if (Type == N_TimeDependentNeuronState - 2){
			if (model->GetModelOutputActivityType() == OUTPUT_CURRENT){
				connection->SetSubindexType(this->CurrentSynapsis->GetNInputCurrentSynapsesPerNeuron(connection->GetTarget()->GetIndex_VectorNeuronState()));
				this->CurrentSynapsis->IncrementNInputCurrentSynapsesPerNeuron(connection->GetTarget()->GetIndex_VectorNeuronState());
				return true;
			}
			else{
				cout << "Synapses type " << Type << " of neuron model " << TimeDrivenInferiorOliveCell::GetName() << " must receive current. The source model generates spikes." << endl;
				return false;
			}
		}
		//Electrical coupling synapsis
		if (Type == N_TimeDependentNeuronState - 1){
			return true;
		}
	}
	cout << "Neuron model " << TimeDrivenInferiorOliveCell::GetName() << " does not support input synapses of type " << Type << ". Just defined " << N_TimeDependentNeuronState << " synapses types." << endl;
	return false;
}



void TimeDrivenInferiorOliveCell::InitializeElectricalCouplingSynapseDependencies(){
	for (int i = 0; i<this->GetVectorNeuronState()->GetSizeState(); i++){
		if (N_coupling_synapses[i] > 0){
			Weight_coupling_synapses[i] = new float[N_coupling_synapses[i]];
			Potential_coupling_synapses[i] = (float**)new float*[N_coupling_synapses[i]];
		}
		else{
			Weight_coupling_synapses[i] = 0;
			Potential_coupling_synapses[i] = 0;
		}
	}
}


void TimeDrivenInferiorOliveCell::CalculateElectricalCouplingSynapseNumber(Interconnection * inter){
	if (inter->GetType() == 4){
		int TargetIndex = inter->GetTarget()->GetIndex_VectorNeuronState();
		N_coupling_synapses[TargetIndex]++;
	}
}

void TimeDrivenInferiorOliveCell::CalculateElectricalCouplingSynapseDependencies(Interconnection * inter){
	if (inter->GetType() == 4){
		int TargetIndex = inter->GetTarget()->GetIndex_VectorNeuronState();
		Weight_coupling_synapses[TargetIndex][index_coupling_synapses[TargetIndex]] = inter->GetWeight();
		Potential_coupling_synapses[TargetIndex][index_coupling_synapses[TargetIndex]] = inter->GetSource()->GetNeuronModel()->GetVectorNeuronState()->GetStateVariableAt(inter->GetSource()->GetIndex_VectorNeuronState());
		index_coupling_synapses[TargetIndex]++;
	}
}


std::map<std::string,boost::any> TimeDrivenInferiorOliveCell::GetParameters() const {
	// Return a dictionary with the parameters
	std::map<std::string,boost::any> newMap = TimeDrivenNeuronModel::GetParameters();
	newMap["e_exc"] = boost::any(this->e_exc); // Excitatory reversal potential (mV)
	newMap["e_inh"] = boost::any(this->e_inh); // Inhibitory reversal potential (mV)
	newMap["e_leak"] = boost::any(this->e_leak); // Effective leak potential (mV)
	newMap["v_thr"] = boost::any(this->v_thr); // Effective threshold potential (mV)
	newMap["c_m"] = boost::any(this->c_m); // Membrane capacitance (uF/cm^2)
	newMap["tau_exc"] = boost::any(this->tau_exc); // AMPA (excitatory) receptor time constant (ms)
	newMap["tau_inh"] = boost::any(this->tau_inh); // GABA (inhibitory) receptor time constant (ms)
	newMap["tau_nmda"] = boost::any(this->tau_nmda); // NMDA (excitatory) receptor time constant (ms)
	newMap["tau_ref"] = boost::any(this->tau_ref); // Refractory period (ms)
	newMap["g_leak"] = boost::any(this->g_leak); // Leak conductance (mS/cm^2)
	newMap["area"] = boost::any(this->area); // Cell area (cm^2)
	return newMap;
}

std::map<std::string, boost::any> TimeDrivenInferiorOliveCell::GetSpecificNeuronParameters(int index) const noexcept(false){
	return GetParameters();
}

void TimeDrivenInferiorOliveCell::SetParameters(std::map<std::string, boost::any> param_map) noexcept(false){

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

	it = param_map.find("v_thr");
	if (it != param_map.end()){
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
		this->tau_ref_0_5 = new_param*0.5f;
		this->inv_tau_ref_0_5 = 1.0f/(new_param*0.5);
		param_map.erase(it);
	}

	it = param_map.find("g_leak");
	if (it != param_map.end()){
		float new_param = boost::any_cast<float>(it->second);
		this->g_leak = new_param;
		param_map.erase(it);
	}

	it = param_map.find("area");
	if (it != param_map.end()){
		float new_param = boost::any_cast<float>(it->second);
		this->area = new_param;
		this->inv_area = 1.0f/new_param;
		param_map.erase(it);
	}

	// Search for the parameters in the dictionary
	TimeDrivenNeuronModel::SetParameters(param_map);

	//Set the new g_nmda_inf values based on the e_exc and e_inh parameters
	Generate_g_nmda_inf_values();

	return;
}


IntegrationMethod * TimeDrivenInferiorOliveCell::CreateIntegrationMethod(ModelDescription imethodDescription) noexcept(false){
	return IntegrationMethodFactory<TimeDrivenInferiorOliveCell>::CreateIntegrationMethod(imethodDescription, (TimeDrivenInferiorOliveCell*) this);
}


std::map<std::string,boost::any> TimeDrivenInferiorOliveCell::GetDefaultParameters() {
	// Return a dictionary with the parameters
	std::map<std::string, boost::any> newMap = TimeDrivenNeuronModel::GetDefaultParameters<TimeDrivenInferiorOliveCell>();
	newMap["e_exc"] = boost::any(0.0f); // Excitatory reversal potential (mV)
	newMap["e_inh"] = boost::any(-80.0f); // Inhibitory reversal potential (mV)
	newMap["e_leak"] = boost::any(-70.0f); // Effective leak potential (mV)
	newMap["v_thr"] = boost::any(-50.0f); // Effective threshold potential (mV)
	newMap["c_m"] = boost::any(1.0f); // Membrane capacitance (uF/cm^2)
	newMap["tau_exc"] = boost::any(1.0f); // AMPA (excitatory) receptor time constant (ms)
	newMap["tau_inh"] = boost::any(2.0f); // GABA (inhibitory) receptor time constant (ms)
	newMap["tau_nmda"] = boost::any(20.0f); // NMDA (excitatory) receptor time constant (ms)
	newMap["tau_ref"] = boost::any(1.35f); // Refractory period (ms)
	newMap["g_leak"] = boost::any(0.015f); // Leak conductance (mS/cm^2)
	newMap["area"] = boost::any(0.00001f); // Cell area (cm^2)
	return newMap;
}

NeuronModel* TimeDrivenInferiorOliveCell::CreateNeuronModel(ModelDescription nmDescription){
	TimeDrivenInferiorOliveCell * nmodel = new TimeDrivenInferiorOliveCell();
	nmodel->SetParameters(nmDescription.param_map);
	return nmodel;
}

ModelDescription TimeDrivenInferiorOliveCell::ParseNeuronModel(std::string FileName) noexcept(false){
	FILE *fh;
	ModelDescription nmodel;
	nmodel.model_name = TimeDrivenInferiorOliveCell::GetName();
	long Currentline = 0L;
	fh=fopen(FileName.c_str(),"rt");
	if(!fh) {
		throw EDLUTFileException(TASK_TIME_DRIVEN_INFERIOR_OLIVE_CELL_LOAD, ERROR_NEURON_MODEL_OPEN, REPAIR_NEURON_MODEL_NAME, Currentline, FileName.c_str());
	}

	Currentline = 1L;
	skip_comments(fh, Currentline);

	float param;
	if (fscanf(fh, "%f", &param) != 1) {
		throw EDLUTFileException(TASK_TIME_DRIVEN_INFERIOR_OLIVE_CELL_LOAD, ERROR_TIME_DRIVEN_INFERIOR_OLIVE_CELL_E_EXC, REPAIR_NEURON_MODEL_VALUES, Currentline, FileName.c_str());
	}
	nmodel.param_map["e_exc"] = boost::any(param);

	skip_comments(fh, Currentline);
	if (fscanf(fh, "%f", &param) != 1) {
		throw EDLUTFileException(TASK_TIME_DRIVEN_INFERIOR_OLIVE_CELL_LOAD, ERROR_TIME_DRIVEN_INFERIOR_OLIVE_CELL_E_INH, REPAIR_NEURON_MODEL_VALUES, Currentline, FileName.c_str());
	}
	nmodel.param_map["e_inh"] = boost::any(param);

	skip_comments(fh, Currentline);
	if (fscanf(fh, "%f", &param) != 1) {
		throw EDLUTFileException(TASK_TIME_DRIVEN_INFERIOR_OLIVE_CELL_LOAD, ERROR_TIME_DRIVEN_INFERIOR_OLIVE_CELL_E_LEAK, REPAIR_NEURON_MODEL_VALUES, Currentline, FileName.c_str());
	}
	nmodel.param_map["e_leak"] = boost::any(param);

	skip_comments(fh, Currentline);
	if (fscanf(fh, "%f", &param) != 1) {
		throw EDLUTFileException(TASK_TIME_DRIVEN_INFERIOR_OLIVE_CELL_LOAD, ERROR_TIME_DRIVEN_INFERIOR_OLIVE_CELL_V_THR, REPAIR_NEURON_MODEL_VALUES, Currentline, FileName.c_str());
	}
	nmodel.param_map["v_thr"] = boost::any(param);

	skip_comments(fh, Currentline);
	if (fscanf(fh, "%f", &param) != 1 || param <= 0.0f) {
		throw EDLUTFileException(TASK_TIME_DRIVEN_INFERIOR_OLIVE_CELL_LOAD, ERROR_TIME_DRIVEN_INFERIOR_OLIVE_CELL_C_M, REPAIR_NEURON_MODEL_VALUES, Currentline, FileName.c_str());
	}
	nmodel.param_map["c_m"] = boost::any(param);

	skip_comments(fh, Currentline);
	if (fscanf(fh, "%f", &param) != 1 || param <= 0.0f) {
		throw EDLUTFileException(TASK_TIME_DRIVEN_INFERIOR_OLIVE_CELL_LOAD, ERROR_TIME_DRIVEN_INFERIOR_OLIVE_CELL_TAU_EXC, REPAIR_NEURON_MODEL_VALUES, Currentline, FileName.c_str());
	}
	nmodel.param_map["tau_exc"] = boost::any(param);

	skip_comments(fh, Currentline);
	if (fscanf(fh, "%f", &param) != 1 || param <= 0.0f) {
		throw EDLUTFileException(TASK_TIME_DRIVEN_INFERIOR_OLIVE_CELL_LOAD, ERROR_TIME_DRIVEN_INFERIOR_OLIVE_CELL_TAU_INH, REPAIR_NEURON_MODEL_VALUES, Currentline, FileName.c_str());
	}
	nmodel.param_map["tau_inh"] = boost::any(param);

	skip_comments(fh, Currentline);
	if (fscanf(fh, "%f", &param) != 1 || param <= 0.0f) {
		throw EDLUTFileException(TASK_TIME_DRIVEN_INFERIOR_OLIVE_CELL_LOAD, ERROR_TIME_DRIVEN_INFERIOR_OLIVE_CELL_TAU_NMDA, REPAIR_NEURON_MODEL_VALUES, Currentline, FileName.c_str());
	}
	nmodel.param_map["tau_nmda"] = boost::any(param);

	skip_comments(fh, Currentline);
	if (fscanf(fh, "%f", &param) != 1 || param <= 0.0f) {
		throw EDLUTFileException(TASK_TIME_DRIVEN_INFERIOR_OLIVE_CELL_LOAD, ERROR_TIME_DRIVEN_INFERIOR_OLIVE_CELL_TAU_REF, REPAIR_NEURON_MODEL_VALUES, Currentline, FileName.c_str());
	}
	nmodel.param_map["tau_ref"] = boost::any(param);

	skip_comments(fh, Currentline);
	if (fscanf(fh, "%f", &param) != 1 || param <= 0.0f) {
		throw EDLUTFileException(TASK_TIME_DRIVEN_INFERIOR_OLIVE_CELL_LOAD, ERROR_TIME_DRIVEN_INFERIOR_OLIVE_CELL_G_LEAK, REPAIR_NEURON_MODEL_VALUES, Currentline, FileName.c_str());
	}
	nmodel.param_map["g_leak"] = boost::any(param);

	skip_comments(fh, Currentline);
	if (fscanf(fh, "%f", &param) != 1 || param <= 0.0f) {
		throw EDLUTFileException(TASK_TIME_DRIVEN_INFERIOR_OLIVE_CELL_LOAD, ERROR_TIME_DRIVEN_INFERIOR_OLIVE_CELL_AREA, REPAIR_NEURON_MODEL_VALUES, Currentline, FileName.c_str());
	}
	nmodel.param_map["area"] = boost::any(param);

	skip_comments(fh, Currentline);
	try {
		ModelDescription intMethodDescription = TimeDrivenNeuronModel::ParseIntegrationMethod<TimeDrivenInferiorOliveCell>(fh, Currentline);
		nmodel.param_map["int_meth"] = boost::any(intMethodDescription);
	} catch (EDLUTException exc) {
		throw EDLUTFileException(exc, Currentline, FileName.c_str());
	}

    nmodel.param_map["name"] = boost::any(TimeDrivenInferiorOliveCell::GetName());

    fclose(fh);

	return nmodel;
}

std::string TimeDrivenInferiorOliveCell::GetName(){
	return "TimeDrivenInferiorOliveCell";
}

std::map<std::string, std::string> TimeDrivenInferiorOliveCell::GetNeuronModelInfo() {
	// Return a dictionary with the parameters
	std::map<std::string, std::string> newMap;
	newMap["info"] = std::string("CPU Time-driven LIF neuron model representing a Inferior Olive cell with one differential equations(membrane potential (v) with emulation of spike shapes for electrical coupling between cells) and five types of input synapses: AMPA (excitatory), GABA (inhibitory), NMDA (excitatory), external input current (set on pA) and electrical coupling");
	newMap["e_exc"] = std::string("Excitatory reversal potential (mV)");
	newMap["e_inh"] = std::string("Inhibitory reversal potential (mV)");
	newMap["e_leak"] = std::string("Effective leak potential (mV)");
	newMap["v_thr"] = std::string("Effective threshold potential (mV)");
	newMap["c_m"] = std::string("Membrane capacitance (uF/cm^2)");
	newMap["tau_exc"] = std::string("AMPA (excitatory) receptor time constant (ms)");
	newMap["tau_inh"] = std::string("GABA (inhibitory) receptor time constant (ms)");
	newMap["tau_nmda"] = std::string("NMDA (excitatory) receptor time constant (ms)");
	newMap["tau_ref"] = std::string("Refractory period (ms)");
	newMap["g_leak"] = std::string("Leak conductance (mS/cm^2)");
	newMap["area"] = std::string("Cell area (cm^2)");
	newMap["int_meth"] = std::string("Integraton method dictionary (from the list of available integration methods)");

	return newMap;
}
