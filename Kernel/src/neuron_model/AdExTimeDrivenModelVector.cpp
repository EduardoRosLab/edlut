/***************************************************************************
 *                           AdExTimeDrivenModelVector.cpp                 *
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

#include "neuron_model/AdExTimeDrivenModelVector.h"
#include "neuron_model/VectorNeuronState.h"
#include "neuron_model/CurrentSynapseModel.h"
#include "simulation/ExponentialTable.h"
#include "spike/Neuron.h"
#include "spike/Interconnection.h"

#include "integration_method/IntegrationMethodFactory.h"


void AdExTimeDrivenModelVector::Generate_g_nmda_inf_values(){
	auxNMDA = (TableSizeNMDA - 1) / (e_exc - e_inh);
	for (int i = 0; i<TableSizeNMDA; i++){
		float V = e_inh + ((e_exc - e_inh)*i) / (TableSizeNMDA - 1);

		//g_nmda_inf
		g_nmda_inf_values[i] = 1.0f / (1.0f + exp(-0.062f*V)*(1.2f / 3.57f));
	}
}


float AdExTimeDrivenModelVector::Get_g_nmda_inf(float V_m){
	int position = int((V_m - e_inh)*auxNMDA);
		if(position<0){
			position=0;
		}
		else if (position>(TableSizeNMDA - 1)){
			position = TableSizeNMDA - 1;
		}
		return g_nmda_inf_values[position];
}


void AdExTimeDrivenModelVector::InitializeCurrentSynapsis(int N_neurons){
	this->CurrentSynapsis = new CurrentSynapseModel(N_neurons);
}


//this neuron model is implemented in a milisecond scale.
AdExTimeDrivenModelVector::AdExTimeDrivenModelVector() : TimeDrivenNeuronModel(MilisecondScale), EXC(false), INH(false), NMDA(false), EXT_I(false),
		vectorParameters(0){
	std::map<std::string, boost::any> param_map = AdExTimeDrivenModelVector::GetDefaultParameters();
	param_map["name"] = AdExTimeDrivenModelVector::GetName();
	this->SetParameters(param_map);

	this->State = (VectorNeuronState *) new VectorNeuronState(N_NeuronStateVariables, true);
}


AdExTimeDrivenModelVector::~AdExTimeDrivenModelVector(void){
	if(vectorParameters!=0){
		delete vectorParameters;
	}
}


VectorNeuronState * AdExTimeDrivenModelVector::InitializeState(){
	return this->GetVectorNeuronState();
}


InternalSpike * AdExTimeDrivenModelVector::ProcessInputSpike(Interconnection * inter, double time){
	// Add the effect of the input spike
	this->GetVectorNeuronState()->IncrementStateVariableAtCPU(inter->GetTargetNeuronModelIndex(), N_DifferentialNeuronState + inter->GetType(), inter->GetWeight());

	return 0;
}


void AdExTimeDrivenModelVector::ProcessInputCurrent(Interconnection * inter, Neuron * target, float current){
	//Update the external current in the corresponding input synapse of type EXT_I (defined in pA).
	this->CurrentSynapsis->SetInputCurrent(target->GetIndex_VectorNeuronState(), inter->GetSubindexType(), current);

	//Update the total external current that receive the neuron coming from all its EXT_I synapsis (defined in pA).
	float total_ext_I = this->CurrentSynapsis->GetTotalCurrent(target->GetIndex_VectorNeuronState());
	State->SetStateVariableAt(target->GetIndex_VectorNeuronState(), EXT_I_index, total_ext_I);
}


bool AdExTimeDrivenModelVector::UpdateState(int index, double CurrentTime){
	//Reset the number of internal spikes in this update period
	this->State->NInternalSpikeIndexs = 0;

	this->integration_method->NextDifferentialEquationValues();

	this->CheckValidIntegration(CurrentTime, this->integration_method->GetValidIntegrationVariable());

	return false;
}


enum NeuronModelOutputActivityType AdExTimeDrivenModelVector::GetModelOutputActivityType(){
	return OUTPUT_SPIKE;
}


enum NeuronModelInputActivityType AdExTimeDrivenModelVector::GetModelInputActivityType(){
	return INPUT_SPIKE_AND_CURRENT;
}


ostream & AdExTimeDrivenModelVector::PrintInfo(ostream & out){
	out << "- AdEx Time-Driven Model Vector (initialization values): " << AdExTimeDrivenModelVector::GetName() << endl;
	out << "\tConductance (a): " << this->a << "nS (this parameter can take a different value for each neuron)" << endl;
	out << "\tSpike trigger adaptation (b): " << this->b << "pA (this parameter can take a different value for each neuron)" << endl;
	out << "\tThreshold slope factor (thr_slo_fac): " << this->thr_slo_fac << "mV (this parameter can take a different value for each neuron)" << endl;
	out << "\tEffective threshold potential (v_thr): " << this->v_thr << "mV (this parameter can take a different value for each neuron)" << endl;
	out << "\tAdaptation time constant (tau_w): " << this->tau_w << "ms (this parameter can take a different value for each neuron)" << endl;
	out << "\tExcitatory reversal potential (e_exc): " << this->e_exc << "mV (this parameter takes the same value foe each neuron)" << endl;
	out << "\tInhibitory reversal potential (e_inh): " << this->e_inh << "mV (this parameter takes the same value foe each neuron)" << endl;
	out << "\tReset potential (e_reset): " << this->e_reset << "mV (this parameter can take a different value for each neuron)" << endl;
	out << "\tEffective leak potential (e_leak): " << this->e_leak << "mV (this parameter can take a different value for each neuron)" << endl;
	out << "\tLeak conductance (g_leak): " << this->g_leak << "nS (this parameter can take a different value for each neuron)" << endl;
	out << "\tMembrane capacitance (c_m): " << this->c_m << "pF (this parameter can take a different value for each neuron)" << endl;
	out << "\tAMPA (excitatory) receptor time constant (tau_exc): " << this->tau_exc << "ms (this parameter takes the same value foe each neuron)" << endl;
	out << "\tGABA (inhibitory) receptor time constant (tau_inh): " << this->tau_inh << "ms (this parameter takes the same value foe each neuron)" << endl;
	out << "\tNMDA (excitatory) receptor time constant (tau_nmda): " << this->tau_nmda << "ms (this parameter takes the same value foe each neuron)" << endl;

	this->integration_method->PrintInfo(out);
	return out;
}


void AdExTimeDrivenModelVector::InitializeStates(int N_neurons, int OpenMPQueueIndex){
	//Initialize neural state variables.
	float initialization[] = {e_leak,0.0f,0.0f,0.0f,0.0f,0.0f};
	State->InitializeStates(N_neurons, initialization);

	//Initialize integration method state variables.
	this->integration_method->SetBifixedStepParameters(v_thr, v_thr, 2.0f);
	this->integration_method->Calculate_conductance_exp_values();
	this->integration_method->InitializeStates(N_neurons, initialization);

	//Initialize the array that stores the number of input current synapses for each neuron in the model
	InitializeCurrentSynapsis(N_neurons);

	vectorParameters = new VectorParameters[N_neurons];
	for(int i=0; i<N_neurons; i++){
		vectorParameters[i].a = this->a;
		vectorParameters[i].b = this->b;
		vectorParameters[i].thr_slo_fac = this->thr_slo_fac;
		vectorParameters[i].inv_thr_slo_fac = this->inv_thr_slo_fac;
		vectorParameters[i].v_thr = this->v_thr;
		//vectorParameters[i].tau_w = this->tau_w;
		vectorParameters[i].inv_tau_w = this->inv_tau_w;
		//vectorParameters[i].e_exc = this->e_exc;
		//vectorParameters[i].e_inh = this->e_inh;
		vectorParameters[i].e_reset = this->e_reset;
		vectorParameters[i].e_leak = this->e_leak;
		vectorParameters[i].g_leak = this->g_leak;
		//vectorParameters[i].c_m = this->c_m;
		vectorParameters[i].inv_c_m = this->inv_c_m;
		//vectorParameters[i].tau_exc = this->tau_exc;
		//vectorParameters[i].inv_tau_exc = this->inv_tau_exc;
		//vectorParameters[i].tau_inh = this->tau_inh;
		//vectorParameters[i].inv_tau_inh = this->inv_tau_inh;
		//vectorParameters[i].tau_nmda = this->tau_nmda;
		//vectorParameters[i].inv_tau_nmda = this->inv_tau_nmda;
	}
}


void AdExTimeDrivenModelVector::GetBifixedStepParameters(float & startVoltageThreshold, float & endVoltageThreshold, float & timeAfterEndVoltageThreshold){
	startVoltageThreshold = this->v_thr;
	endVoltageThreshold = this->v_thr;
	timeAfterEndVoltageThreshold = 2.0f;
	return;
}


void AdExTimeDrivenModelVector::EvaluateSpikeCondition(float previous_V, float * NeuronState, int index, float elapsedTimeInNeuronModelScale){
	if (NeuronState[V_m_index] > 0.0f){
		NeuronState[V_m_index] = this->vectorParameters[index].e_reset;
		NeuronState[w_index] += this->vectorParameters[index].b;
		State->NewFiredSpike(index);
		this->integration_method->resetState(index);
		this->State->InternalSpikeIndexs[this->State->NInternalSpikeIndexs] = index;
		this->State->NInternalSpikeIndexs++;
	}
}

void AdExTimeDrivenModelVector::EvaluateDifferentialEquation(float * NeuronState, float * AuxNeuronState, int index, float elapsed_time){
	if (NeuronState[V_m_index] <= -20.0f){
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
		current += NeuronState[EXT_I_index]; // (defined in pA).

		//V
		AuxNeuronState[V_m_index] = (this->vectorParameters[index].g_leak*(this->vectorParameters[index].e_leak - NeuronState[V_m_index]) + this->vectorParameters[index].g_leak*this->vectorParameters[index].thr_slo_fac*ExponentialTable::GetResult((NeuronState[V_m_index] - this->vectorParameters[index].v_thr)*this->vectorParameters[index].inv_thr_slo_fac) - NeuronState[w_index] + current)*this->vectorParameters[index].inv_c_m;
		//w
		AuxNeuronState[w_index] = (this->vectorParameters[index].a*(NeuronState[V_m_index] - this->vectorParameters[index].e_leak) - NeuronState[w_index])*this->vectorParameters[index].inv_tau_w;
	}
	else if (NeuronState[V_m_index] <= 0.0f){
		float Vm=-20.0f;
		float current = 0.0;
		if(EXC){
			current += NeuronState[EXC_index] * (this->e_exc - Vm);
		}
		if(INH){
			current += NeuronState[INH_index] * (this->e_inh - Vm);
		}
		if(NMDA){
			//float g_nmda_inf = 1.0f/(1.0f + ExponentialTable::GetResult(-0.062f*NeuronState[V_m_index])*(1.2f/3.57f));
			float g_nmda_inf = Get_g_nmda_inf(Vm);
			current += NeuronState[NMDA_index] * g_nmda_inf*(this->e_exc - Vm);
		}
		current += NeuronState[EXT_I_index]; // (defined in pA).

		//V
		AuxNeuronState[V_m_index] = (this->vectorParameters[index].g_leak*(this->vectorParameters[index].e_leak - Vm) + this->vectorParameters[index].g_leak*this->vectorParameters[index].thr_slo_fac*ExponentialTable::GetResult((Vm - this->vectorParameters[index].v_thr)*this->vectorParameters[index].inv_thr_slo_fac) - NeuronState[w_index] + current)*this->vectorParameters[index].inv_c_m;
		//w
		AuxNeuronState[w_index] = (this->vectorParameters[index].a*(NeuronState[V_m_index] - this->vectorParameters[index].e_leak) - NeuronState[w_index])*this->vectorParameters[index].inv_tau_w;
	}else{
		//V
		AuxNeuronState[V_m_index]=0;
		//w
		AuxNeuronState[w_index]=0;
	}
}

void AdExTimeDrivenModelVector::EvaluateTimeDependentEquation(float * NeuronState, int index, int elapsed_time_index){
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

void AdExTimeDrivenModelVector::Calculate_conductance_exp_values(int index, float elapsed_time){
	//excitatory synapse.
	Set_conductance_exp_values(index, 0, expf(-elapsed_time*this->inv_tau_exc));
	//inhibitory synapse.
	Set_conductance_exp_values(index, 1, expf(-elapsed_time*this->inv_tau_inh));
	//nmda synapse.
	Set_conductance_exp_values(index, 2, expf(-elapsed_time*this->inv_tau_nmda));
}


bool AdExTimeDrivenModelVector::CheckSynapseType(Interconnection * connection){
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
			cout << "Synapses type " << Type << " of neuron model " << AdExTimeDrivenModelVector::GetName() << " must receive spikes. The source model generates currents." << endl;
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
				cout << "Synapses type " << Type << " of neuron model " << AdExTimeDrivenModelVector::GetName() << " must receive current. The source model generates spikes." << endl;
				return false;
			}
		}
	}
	cout << "Neuron model " << AdExTimeDrivenModelVector::GetName() << " does not support input synapses of type " << Type << ". Just defined " << N_TimeDependentNeuronState << " synapses types." << endl;
	return false;
}

std::map<std::string,boost::any> AdExTimeDrivenModelVector::GetParameters() const {
	// Return a dictionary with the parameters
	std::map<std::string,boost::any> newMap = TimeDrivenNeuronModel::GetParameters();
	newMap["a"] = boost::any(this->a); //conductance (nS): VECTOR
	newMap["b"] = boost::any(this->b); //spike trigger adaptation (pA): VECTOR
	newMap["thr_slo_fac"] = boost::any(this->thr_slo_fac); //threshold slope factor (mV): VECTOR
	newMap["v_thr"] = boost::any(this->v_thr); //effective threshold potential (mV): VECTOR
	newMap["tau_w"] = boost::any(this->tau_w); //adaptation time constant (ms): VECTOR
	newMap["e_exc"] = boost::any(this->e_exc); //excitatory reversal potential (mV): FIXED
	newMap["e_inh"] = boost::any(this->e_inh); //inhibitory reversal potential (mV): FIXED
	newMap["e_reset"] = boost::any(this->e_reset); //reset potential (mV): VECTOR
	newMap["e_leak"] = boost::any(this->e_leak); //effective leak potential (mV): VECTOR
	newMap["g_leak"] = boost::any(this->g_leak); //leak conductance (nS): VECTOR
	newMap["c_m"] = boost::any(this->c_m); //membrane capacitance (pF): VECTOR
	newMap["tau_exc"] = boost::any(this->tau_exc); //AMPA (excitatory) receptor time constant (ms): FIXED
	newMap["tau_inh"] = boost::any(this->tau_inh); //GABA (inhibitory) receptor time constant (ms): FIXED
	newMap["tau_nmda"] = boost::any(this->tau_nmda); //NMDA (excitatory) receptor time constant (ms): FIXED
	return newMap;
}

std::map<std::string, boost::any> AdExTimeDrivenModelVector::GetSpecificNeuronParameters(int index) const noexcept(false){
	//This method must be executed after the initilization of the model.
	if (this->State->GetSizeState() == 0){
		//The neuron model has not been initialized
		throw EDLUTException(TASK_GET_NEURON_SPECIFIC_PARAMETERS, ERROR_NON_INITIALIZED_SIMULATION, REPAIR_EXECUTE_AFTER_INITIALIZE_SIMULATION);
	}

	// Return a dictionary with the parameters
	std::map<std::string, boost::any> newMap = TimeDrivenNeuronModel::GetParameters();
	newMap["a"] = boost::any(this->vectorParameters[index].a); //conductance (nS): VECTOR
	newMap["b"] = boost::any(this->vectorParameters[index].b); //spike trigger adaptation (pA): VECTOR
	newMap["thr_slo_fac"] = boost::any(this->vectorParameters[index].thr_slo_fac); //threshold slope factor (mV): VECTOR
	newMap["v_thr"] = boost::any(this->vectorParameters[index].v_thr); //effective threshold potential (mV): VECTOR
	newMap["tau_w"] = boost::any(1.0f/this->vectorParameters[index].inv_tau_w); //adaptation time constant (ms): VECTOR
	newMap["e_exc"] = boost::any(this->e_exc); //excitatory reversal potential (mV): FIXED
	newMap["e_inh"] = boost::any(this->e_inh); //inhibitory reversal potential (mV): FIXED
	newMap["e_reset"] = boost::any(this->vectorParameters[index].e_reset); //reset potential (mV): VECTOR
	newMap["e_leak"] = boost::any(this->vectorParameters[index].e_leak); //effective leak potential (mV): VECTOR
	newMap["g_leak"] = boost::any(this->vectorParameters[index].g_leak); //leak conductance (nS): VECTOR
	newMap["c_m"] = boost::any(1.0f/this->vectorParameters[index].inv_c_m); //membrane capacitance (pF): VECTOR
	newMap["tau_exc"] = boost::any(this->tau_exc); //AMPA (excitatory) receptor time constant (ms): FIXED
	newMap["tau_inh"] = boost::any(this->tau_inh); //GABA (inhibitory) receptor time constant (ms): FIXED
	newMap["tau_nmda"] = boost::any(this->tau_nmda); //NMDA (excitatory) receptor time constant (ms): FIXED
	return newMap;
}

void AdExTimeDrivenModelVector::SetParameters(std::map<std::string, boost::any> param_map) noexcept(false){

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

	it=param_map.find("thr_slo_fac");
	if (it!=param_map.end()){
		float new_param = boost::any_cast<float>(it->second);
		this->thr_slo_fac = new_param;
		this->inv_thr_slo_fac = 1./this->thr_slo_fac;
		param_map.erase(it);
	}

	it=param_map.find("v_thr");
	if (it!=param_map.end()){
		float new_param = boost::any_cast<float>(it->second);
		this->v_thr = new_param;
		param_map.erase(it);
	}

	it=param_map.find("tau_w");
	if (it!=param_map.end()){
		float new_param = boost::any_cast<float>(it->second);
		this->tau_w = new_param;
		this->inv_tau_w = 1./this->tau_w;
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

	it=param_map.find("e_reset");
	if (it!=param_map.end()){
		float new_param = boost::any_cast<float>(it->second);
		this->e_reset = new_param;
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


void AdExTimeDrivenModelVector::SetSpecificNeuronParameters(int index, std::map<std::string, boost::any> param_map) noexcept(false){

	//This method must be executed after the initilization of the model.
	if (this->State->GetSizeState() == 0){
		//The neuron model has not been initialized
		throw EDLUTException(TASK_SET_NEURON_SPECIFIC_PARAMETERS, ERROR_NON_INITIALIZED_SIMULATION, REPAIR_EXECUTE_AFTER_INITIALIZE_SIMULATION);
	}

	// Search for the parameters in the dictionary
	std::map<std::string, boost::any>::iterator it = param_map.find("a");
	if (it != param_map.end()){
		float new_param = boost::any_cast<float>(it->second);
		this->vectorParameters[index].a = new_param;
		param_map.erase(it);
	}

	it = param_map.find("b");
	if (it != param_map.end()){
		float new_param = boost::any_cast<float>(it->second);
		this->vectorParameters[index].b = new_param;
		param_map.erase(it);
	}

	it = param_map.find("thr_slo_fac");
	if (it != param_map.end()){
		float new_param = boost::any_cast<float>(it->second);
		this->vectorParameters[index].thr_slo_fac = new_param;
		this->vectorParameters[index].inv_thr_slo_fac = 1. / this->thr_slo_fac;
		param_map.erase(it);
	}

	it = param_map.find("v_thr");
	if (it != param_map.end()){
		float new_param = boost::any_cast<float>(it->second);
		this->vectorParameters[index].v_thr = new_param;
		param_map.erase(it);
	}

	it = param_map.find("tau_w");
	if (it != param_map.end()){
		float new_param = boost::any_cast<float>(it->second);
		this->vectorParameters[index].inv_tau_w = 1. / new_param;
		param_map.erase(it);
	}

	it = param_map.find("e_reset");
	if (it != param_map.end()){
		float new_param = boost::any_cast<float>(it->second);
		this->vectorParameters[index].e_reset = new_param;
		param_map.erase(it);
	}

	it = param_map.find("e_leak");
	if (it != param_map.end()){
		float new_param = boost::any_cast<float>(it->second);
		this->vectorParameters[index].e_leak = new_param;
		param_map.erase(it);
	}

	it = param_map.find("g_leak");
	if (it != param_map.end()){
		float new_param = boost::any_cast<float>(it->second);
		this->vectorParameters[index].g_leak = new_param;
		param_map.erase(it);
	}

	it = param_map.find("c_m");
	if (it != param_map.end()){
		float new_param = boost::any_cast<float>(it->second);
		this->vectorParameters[index].inv_c_m = 1. / new_param;
		param_map.erase(it);
	}

	NeuronModel::SetSpecificNeuronParameters(index, param_map);

	return;
}

std::map<std::string, std::string> AdExTimeDrivenModelVector::GetVectorizableParameters(){
	std::map<std::string, std::string> vectorizableParameters;
	vectorizableParameters["a"] = std::string("Conductance (nS): VECTOR");
	vectorizableParameters["b"] = std::string("Spike trigger adaptation (pA): VECTOR");
	vectorizableParameters["thr_slo_fac"] = std::string("Threshold slope factor (mV): VECTOR");
	vectorizableParameters["v_thr"] = std::string("Effective threshold potential (mV): VECTOR");
	vectorizableParameters["tau_w"] = std::string("Adaptation time constant (ms): VECTOR");
	vectorizableParameters["e_reset"] = std::string("Reset potential (mV): VECTOR");
	vectorizableParameters["e_leak"] = std::string("Effective leak potential (mV): VECTOR");
	vectorizableParameters["g_leak"] = std::string("Leak conductance (nS): VECTOR");
	vectorizableParameters["c_m"] = std::string("Membrane capacitance (pF): VECTOR");
	return vectorizableParameters;
}


IntegrationMethod * AdExTimeDrivenModelVector::CreateIntegrationMethod(ModelDescription imethodDescription) noexcept(false){
	return IntegrationMethodFactory<AdExTimeDrivenModelVector>::CreateIntegrationMethod(imethodDescription, (AdExTimeDrivenModelVector*) this);
}


std::map<std::string,boost::any> AdExTimeDrivenModelVector::GetDefaultParameters() {
	// Return a dictionary with the parameters
	std::map<std::string,boost::any> newMap = TimeDrivenNeuronModel::GetDefaultParameters<AdExTimeDrivenModelVector>();
	newMap["a"] = boost::any(1.0f); //conductance (nS): VECTOR
	newMap["b"] = boost::any(9.0f); //spike trigger adaptation (pA): VECTOR
	newMap["thr_slo_fac"] = boost::any(2.0f); //threshold slope factor (mV): VECTOR
	newMap["v_thr"] = boost::any(-50.0f); //effective threshold potential (mV): VECTOR
	newMap["tau_w"] = boost::any(50.0f); //adaptation time constant (ms): VECTOR
	newMap["e_exc"] = boost::any(0.0f); //excitatory reversal potential (mV): FIXED
	newMap["e_inh"] = boost::any(-80.0f); //inhibitory reversal potential (mV): FIXED
	newMap["e_reset"] = boost::any(-80.0f); //reset potential (mV): VECTOR
	newMap["e_leak"] = boost::any(-65.0f); //effective leak potential (mV): VECTOR
	newMap["g_leak"] = boost::any(10.0f); //leak conductance (nS): VECTOR
	newMap["c_m"] = boost::any(110.0f); //membrane capacitance (pF): VECTOR
	newMap["tau_exc"] = boost::any(5.0f); //AMPA (excitatory) receptor time constant (ms): FIXED
	newMap["tau_inh"] = boost::any(10.0f); //GABA (inhibitory) receptor time constant (ms): FIXED
	newMap["tau_nmda"] = boost::any(20.0f); //NMDA (excitatory) receptor time constant (ms): FIXED
	return newMap;
}

NeuronModel* AdExTimeDrivenModelVector::CreateNeuronModel(ModelDescription nmDescription){
	AdExTimeDrivenModelVector * nmodel = new AdExTimeDrivenModelVector();
	nmodel->SetParameters(nmDescription.param_map);
	return nmodel;
}

ModelDescription AdExTimeDrivenModelVector::ParseNeuronModel(std::string FileName) noexcept(false){
	FILE *fh;
	ModelDescription nmodel;
	nmodel.model_name = AdExTimeDrivenModelVector::GetName();
	long Currentline = 0L;
	fh=fopen(FileName.c_str(),"rt");
	if(!fh) {
		throw EDLUTFileException(TASK_ADEX_TIME_DRIVEN_MODEL_LOAD, ERROR_NEURON_MODEL_OPEN, REPAIR_NEURON_MODEL_NAME, Currentline, FileName.c_str());
	}

	Currentline = 1L;
	skip_comments(fh, Currentline);

	float param;
	if (fscanf(fh, "%f", &param) != 1) {
		throw EDLUTFileException(TASK_ADEX_TIME_DRIVEN_MODEL_LOAD, ERROR_ADEX_TIME_DRIVEN_MODEL_A, REPAIR_NEURON_MODEL_VALUES, Currentline, FileName.c_str());
	}
	nmodel.param_map["a"] = boost::any(param);

	skip_comments(fh, Currentline);
	if (fscanf(fh, "%f", &param) != 1) {
		throw EDLUTFileException(TASK_ADEX_TIME_DRIVEN_MODEL_LOAD, ERROR_ADEX_TIME_DRIVEN_MODEL_B, REPAIR_NEURON_MODEL_VALUES, Currentline, FileName.c_str());
	}
	nmodel.param_map["b"] = boost::any(param);

	skip_comments(fh, Currentline);
	if (fscanf(fh, "%f", &param) != 1 || param <= 0.0f) {
		throw EDLUTFileException(TASK_ADEX_TIME_DRIVEN_MODEL_LOAD, ERROR_ADEX_TIME_DRIVEN_MODEL_THR_SLO_FAC, REPAIR_NEURON_MODEL_VALUES, Currentline, FileName.c_str());
	}
	nmodel.param_map["thr_slo_fac"] = boost::any(param);

	skip_comments(fh, Currentline);
	if (fscanf(fh, "%f", &param) != 1) {
		throw EDLUTFileException(TASK_ADEX_TIME_DRIVEN_MODEL_LOAD, ERROR_ADEX_TIME_DRIVEN_MODEL_V_THR, REPAIR_NEURON_MODEL_VALUES, Currentline, FileName.c_str());
	}
	nmodel.param_map["v_thr"] = boost::any(param);

	skip_comments(fh, Currentline);
	if (fscanf(fh, "%f", &param) != 1 || param <= 0.0f) {
		throw EDLUTFileException(TASK_ADEX_TIME_DRIVEN_MODEL_LOAD, ERROR_ADEX_TIME_DRIVEN_MODEL_TAU_W, REPAIR_NEURON_MODEL_VALUES, Currentline, FileName.c_str());
	}
	nmodel.param_map["tau_w"] = boost::any(param);

	skip_comments(fh, Currentline);
	if (fscanf(fh, "%f", &param) != 1) {
		throw EDLUTFileException(TASK_ADEX_TIME_DRIVEN_MODEL_LOAD, ERROR_ADEX_TIME_DRIVEN_MODEL_E_EXC, REPAIR_NEURON_MODEL_VALUES, Currentline, FileName.c_str());
	}
	nmodel.param_map["e_exc"] = boost::any(param);

	skip_comments(fh, Currentline);
	if (fscanf(fh, "%f", &param) != 1) {
		throw EDLUTFileException(TASK_ADEX_TIME_DRIVEN_MODEL_LOAD, ERROR_ADEX_TIME_DRIVEN_MODEL_E_INH, REPAIR_NEURON_MODEL_VALUES, Currentline, FileName.c_str());
	}
	nmodel.param_map["e_inh"] = boost::any(param);

	skip_comments(fh, Currentline);
	if (fscanf(fh, "%f", &param) != 1) {
		throw EDLUTFileException(TASK_ADEX_TIME_DRIVEN_MODEL_LOAD, ERROR_ADEX_TIME_DRIVEN_MODEL_E_RESET, REPAIR_NEURON_MODEL_VALUES, Currentline, FileName.c_str());
	}
	nmodel.param_map["e_reset"] = boost::any(param);

	skip_comments(fh, Currentline);
	if (fscanf(fh, "%f", &param) != 1) {
		throw EDLUTFileException(TASK_ADEX_TIME_DRIVEN_MODEL_LOAD, ERROR_ADEX_TIME_DRIVEN_MODEL_E_LEAK, REPAIR_NEURON_MODEL_VALUES, Currentline, FileName.c_str());
	}
	nmodel.param_map["e_leak"] = boost::any(param);

	skip_comments(fh, Currentline);
	if (fscanf(fh, "%f", &param) != 1 || param <= 0.0f) {
		throw EDLUTFileException(TASK_ADEX_TIME_DRIVEN_MODEL_LOAD, ERROR_ADEX_TIME_DRIVEN_MODEL_G_LEAK, REPAIR_NEURON_MODEL_VALUES, Currentline, FileName.c_str());
	}
	nmodel.param_map["g_leak"] = boost::any(param);

	skip_comments(fh, Currentline);
	if (fscanf(fh, "%f", &param) != 1 || param <= 0.0f) {
		throw EDLUTFileException(TASK_ADEX_TIME_DRIVEN_MODEL_LOAD, ERROR_ADEX_TIME_DRIVEN_MODEL_C_M, REPAIR_NEURON_MODEL_VALUES, Currentline, FileName.c_str());
	}
	nmodel.param_map["c_m"] = boost::any(param);

	skip_comments(fh, Currentline);
	if (fscanf(fh, "%f", &param) != 1 || param <= 0.0f) {
		throw EDLUTFileException(TASK_ADEX_TIME_DRIVEN_MODEL_LOAD, ERROR_ADEX_TIME_DRIVEN_MODEL_TAU_EXC, REPAIR_NEURON_MODEL_VALUES, Currentline, FileName.c_str());
	}
	nmodel.param_map["tau_exc"] = boost::any(param);

	skip_comments(fh, Currentline);
	if (fscanf(fh, "%f", &param) != 1 || param <= 0.0f) {
		throw EDLUTFileException(TASK_ADEX_TIME_DRIVEN_MODEL_LOAD, ERROR_ADEX_TIME_DRIVEN_MODEL_TAU_INH, REPAIR_NEURON_MODEL_VALUES, Currentline, FileName.c_str());
	}
	nmodel.param_map["tau_inh"] = boost::any(param);

	skip_comments(fh, Currentline);
	if (fscanf(fh, "%f", &param) != 1 || param <= 0.0f) {
		throw EDLUTFileException(TASK_ADEX_TIME_DRIVEN_MODEL_LOAD, ERROR_ADEX_TIME_DRIVEN_MODEL_TAU_NMDA, REPAIR_NEURON_MODEL_VALUES, Currentline, FileName.c_str());
	}
	nmodel.param_map["tau_nmda"] = boost::any(param);

	skip_comments(fh, Currentline);
	try {
		ModelDescription intMethodDescription = TimeDrivenNeuronModel::ParseIntegrationMethod<AdExTimeDrivenModelVector>(fh, Currentline);
		nmodel.param_map["int_meth"] = boost::any(intMethodDescription);
	} catch (EDLUTException exc) {
		throw EDLUTFileException(exc, Currentline, FileName.c_str());
	}

	fclose(fh);

	nmodel.param_map["name"] = boost::any(AdExTimeDrivenModelVector::GetName());

	return nmodel;
}

std::string AdExTimeDrivenModelVector::GetName(){
	return "AdExTimeDrivenModelVector";
}


std::map<std::string, std::string> AdExTimeDrivenModelVector::GetNeuronModelInfo() {
	// Return a dictionary with the parameters
	std::map<std::string, std::string> newMap;
	newMap["info"] = std::string("CPU Time-driven Adaptative Exponential Integrate and Fire (AdEx) neuron model with VECTORIZED PARAMETERS and two differential equations(membrane potential (v) and membrane recovery (w)) and four types of input synapses: AMPA (excitatory), GABA (inhibitory), NMDA (excitatory) and external input current (set on pA)");
	newMap["a"] = std::string("Conductance (nS): VECTOR");
	newMap["b"] = std::string("Spike trigger adaptation (pA): VECTOR");
	newMap["thr_slo_fac"] = std::string("Threshold slope factor (mV): VECTOR");
	newMap["v_thr"] = std::string("Effective threshold potential (mV): VECTOR");
	newMap["tau_w"] = std::string("Adaptation time constant (ms): VECTOR");
	newMap["e_exc"] = std::string("Excitatory reversal potential (mV): FIXED");
	newMap["e_inh"] = std::string("Inhibitory reversal potential (mV): FIXED");
	newMap["e_reset"] = std::string("Reset potential (mV): VECTOR");
	newMap["e_leak"] = std::string("Effective leak potential (mV): VECTOR");
	newMap["g_leak"] = std::string("Leak conductance (nS): VECTOR");
	newMap["c_m"] = std::string("Membrane capacitance (pF): VECTOR");
	newMap["tau_exc"] = std::string("AMPA (excitatory) receptor time constant (ms): FIXED");
	newMap["tau_inh"] = std::string("GABA (inhibitory) receptor time constant (ms): FIXED");
	newMap["tau_nmda"] = std::string("NMDA (excitatory) receptor time constant (ms): FIXED");
	newMap["int_meth"] = std::string("Integraton method dictionary (from the list of available integration methods): FIXED");

	return newMap;
}
