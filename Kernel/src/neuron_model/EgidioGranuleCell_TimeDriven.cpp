/***************************************************************************
 *                           EdidioGranuleCell_TimeDriven.cpp              *
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

#include "neuron_model/EgidioGranuleCell_TimeDriven.h"
#include "neuron_model/VectorNeuronState.h"
#include "neuron_model/CurrentSynapseModel.h"
#include "simulation/ExponentialTable.h"
#include "spike/Neuron.h"
#include "spike/Interconnection.h"

#include "integration_method/IntegrationMethodFactory.h"

	const float EgidioGranuleCell_TimeDriven::gMAXNa_f=0.013f;
	const float EgidioGranuleCell_TimeDriven::gMAXNa_r=0.0005f;
	const float EgidioGranuleCell_TimeDriven::gMAXNa_p=0.0002f;
	const float EgidioGranuleCell_TimeDriven::gMAXK_V=0.003f;
	const float EgidioGranuleCell_TimeDriven::gMAXK_A=0.004f;
	const float EgidioGranuleCell_TimeDriven::gMAXK_IR=0.0009f;
	const float EgidioGranuleCell_TimeDriven::gMAXK_Ca=0.004f;
	const float EgidioGranuleCell_TimeDriven::gMAXCa=0.00046f;
	const float EgidioGranuleCell_TimeDriven::gMAXK_sl=0.00035f;
	const float EgidioGranuleCell_TimeDriven::gLkg1=5.68e-5f;
	const float EgidioGranuleCell_TimeDriven::gLkg2=2.17e-5f;
	const float EgidioGranuleCell_TimeDriven::VNa=87.39f;
	const float EgidioGranuleCell_TimeDriven::VK=-84.69f;
	const float EgidioGranuleCell_TimeDriven::VLkg1=-58.0f;
	const float EgidioGranuleCell_TimeDriven::VLkg2=-65.0f;
	const float EgidioGranuleCell_TimeDriven::V0_xK_Ai=-46.7f;
	const float EgidioGranuleCell_TimeDriven::K_xK_Ai=-19.8f;
	const float EgidioGranuleCell_TimeDriven::V0_yK_Ai=-78.8f;
	const float EgidioGranuleCell_TimeDriven::K_yK_Ai=8.4f;
	const float EgidioGranuleCell_TimeDriven::V0_xK_sli=-30.0f;
	const float EgidioGranuleCell_TimeDriven::B_xK_sli=6.0f;
	const float EgidioGranuleCell_TimeDriven::F=96485.309f;
	const float EgidioGranuleCell_TimeDriven::A=1e-04f;
	const float EgidioGranuleCell_TimeDriven::d=0.2f;
	const float EgidioGranuleCell_TimeDriven::betaCa=1.5f;
	const float EgidioGranuleCell_TimeDriven::Ca0=1e-04f;
	const float EgidioGranuleCell_TimeDriven::R=8.3134f;
	const float EgidioGranuleCell_TimeDriven::cao=2.0f;
	const float EgidioGranuleCell_TimeDriven::Cm = 1.0e-3f;
	const float EgidioGranuleCell_TimeDriven::inv_Cm = 1.0f/1.0e-3f;
	const float EgidioGranuleCell_TimeDriven::temper=30.0f;
	const float EgidioGranuleCell_TimeDriven::Q10_20 = pow(3.0f,((temper-20.0f)/10.0f));
	const float EgidioGranuleCell_TimeDriven::Q10_22 = pow(3.0f,((temper-22.0f)/10.0f));
	const float EgidioGranuleCell_TimeDriven::Q10_30 = pow(3.0f,((temper-30.0f)/10.0f));
	const float EgidioGranuleCell_TimeDriven::Q10_6_3 = pow(3.0f,((temper-6.3f)/10.0f));

	const float EgidioGranuleCell_TimeDriven::Max_V=50.0f;
	const float EgidioGranuleCell_TimeDriven::Min_V=-100.0f;

	const float EgidioGranuleCell_TimeDriven::aux=(EgidioGranuleCell_TimeDriven::TableSize-1)/( EgidioGranuleCell_TimeDriven::Max_V - EgidioGranuleCell_TimeDriven::Min_V);

	float * EgidioGranuleCell_TimeDriven::channel_values=Generate_channel_values();


void EgidioGranuleCell_TimeDriven::Generate_g_nmda_inf_values(){
	auxNMDA = (TableSizeNMDA - 1) / (e_exc - e_inh);
	for (int i = 0; i<TableSizeNMDA; i++){
		float V = e_inh + ((e_exc - e_inh)*i) / (TableSizeNMDA - 1);

		//g_nmda_inf
		g_nmda_inf_values[i] = 1.0f / (1.0f + exp(-0.062f*V)*(1.2f / 3.57f));
	}
}


float EgidioGranuleCell_TimeDriven::Get_g_nmda_inf(float V_m){
	int position = int((V_m - e_inh)*auxNMDA);
		if(position<0){
			position=0;
		}
		else if (position>(TableSizeNMDA - 1)){
			position = TableSizeNMDA - 1;
		}
		return g_nmda_inf_values[position];
}


void EgidioGranuleCell_TimeDriven::InitializeCurrentSynapsis(int N_neurons){
	this->CurrentSynapsis = new CurrentSynapseModel(N_neurons);
}




//This neuron model is implemented in a milisecond scale.
EgidioGranuleCell_TimeDriven::EgidioGranuleCell_TimeDriven(): TimeDrivenNeuronModel(MilisecondScale),
	e_exc(0.0), e_inh(-80), tau_exc(0.5), tau_inh(10.0), tau_nmda(15.0), v_thr(0.0), EXC(false), INH(false), NMDA(false), EXT_I(false)
{
	std::map<std::string, boost::any> param_map = EgidioGranuleCell_TimeDriven::GetDefaultParameters();
	param_map["name"] = EgidioGranuleCell_TimeDriven::GetName();
	this->SetParameters(param_map);

	this->State = (VectorNeuronState *) new VectorNeuronState(N_NeuronStateVariables, true);
}


EgidioGranuleCell_TimeDriven::~EgidioGranuleCell_TimeDriven(void)
{
	// We cannot remove channel_values unless we are sure that there not exist (now or in the future) any object instance
	/*if(this->channel_values){
		delete this->channel_values;
		this->channel_values=0;
	}*/
}

VectorNeuronState * EgidioGranuleCell_TimeDriven::InitializeState(){
	return this->GetVectorNeuronState();
}


InternalSpike * EgidioGranuleCell_TimeDriven::ProcessInputSpike(Interconnection * inter, double time){

	// Add the effect of the input spike
	this->GetVectorNeuronState()->IncrementStateVariableAtCPU(inter->GetTargetNeuronModelIndex(), N_DifferentialNeuronState + inter->GetType(), inter->GetWeight());

	return 0;
}


void EgidioGranuleCell_TimeDriven::ProcessInputCurrent(Interconnection * inter, Neuron * target, float current){
	//Update the external current in the corresponding input synapse of type EXT_I (defined in pA).
	this->CurrentSynapsis->SetInputCurrent(target->GetIndex_VectorNeuronState(), inter->GetSubindexType(), current);

	//Update the total external current that receive the neuron coming from all its EXT_I synapsis (defined in pA).
	float total_ext_I = this->CurrentSynapsis->GetTotalCurrent(target->GetIndex_VectorNeuronState());
	State->SetStateVariableAt(target->GetIndex_VectorNeuronState(), EXT_I_index, total_ext_I);
}


bool EgidioGranuleCell_TimeDriven::UpdateState(int index, double CurrentTime){
	//Reset the number of internal spikes in this update period
	this->State->NInternalSpikeIndexs = 0;

	this->integration_method->NextDifferentialEquationValues();

	this->CheckValidIntegration(CurrentTime, this->integration_method->GetValidIntegrationVariable());

	return false;
}


enum NeuronModelOutputActivityType EgidioGranuleCell_TimeDriven::GetModelOutputActivityType(){
	return OUTPUT_SPIKE;
}


enum NeuronModelInputActivityType EgidioGranuleCell_TimeDriven::GetModelInputActivityType(){
	return INPUT_SPIKE_AND_CURRENT;
}

ostream & EgidioGranuleCell_TimeDriven::PrintInfo(ostream & out){
	out << "- EgidioGranuleCell Time-Driven Model: " << EgidioGranuleCell_TimeDriven::GetName() << endl;
	out << "\tExcitatory reversal potential (e_exc): " << this->e_exc << "mV" << endl;
	out << "\tInhibitory reversal potential (e_inh): " << this->e_inh << "mV" << endl;
	out << "\tAMPA (excitatory) receptor time constant (tau_exc): " << this->tau_exc << "ms" << endl;
	out << "\tGABA (inhibitory) receptor time constant (tau_inh): " << this->tau_inh << "ms" << endl;
	out << "\tNMDA (excitatory) receptor time constant (tau_nmda): " << this->tau_nmda << "ms" << endl;
	out << "\tEffective threshold potential (v_thr): " << this->v_thr << "mV" << endl;

	this->integration_method->PrintInfo(out);
	return out;
}

void EgidioGranuleCell_TimeDriven::InitializeStates(int N_neurons, int OpenMPQueueIndex){
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
	float initialization[] = { V, xNa_f, yNa_f, xNa_r, yNa_r, xNa_p, xK_V, xK_A, yK_A, xK_IR, xK_Ca, xCa, yCa, xK_sl, Ca, gexc, ginh, gnmda, External_current};
	State->InitializeStates(N_neurons, initialization);

	//Initialize integration method state variables.
	this->integration_method->SetBifixedStepParameters(-50.0f, -50.0f, 2.0f);
	this->integration_method->Calculate_conductance_exp_values();
	this->integration_method->InitializeStates(N_neurons, initialization);

	//Initialize the array that stores the number of input current synapses for each neuron in the model
	InitializeCurrentSynapsis(N_neurons);
}

void EgidioGranuleCell_TimeDriven::GetBifixedStepParameters(float & startVoltageThreshold, float & endVoltageThreshold, float & timeAfterEndVoltageThreshold){
	startVoltageThreshold = -50.0f;
	endVoltageThreshold = -50.0f;
	timeAfterEndVoltageThreshold = 2.0f;
	return;
}


void EgidioGranuleCell_TimeDriven::EvaluateSpikeCondition(float previous_V, float * NeuronState, int index, float elapsedTimeInNeuronModelScale){
	if (NeuronState[V_m_index]>v_thr && previous_V<v_thr){
		State->NewFiredSpike(index);
		this->State->InternalSpikeIndexs[this->State->NInternalSpikeIndexs] = index;
		this->State->NInternalSpikeIndexs++;
	}
}


void EgidioGranuleCell_TimeDriven::EvaluateDifferentialEquation(float * NeuronState, float * AuxNeuronState, int index, float elapsed_time){
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

	//We normalize the current.
	//current *= 1e-9f / 299.26058e-8f;
	current *= 3.34156941e-4;

	float previous_V=NeuronState[V_m_index];

	float VCa=nernst(NeuronState[Ca_index],cao,2,temper);

	float * values=Get_channel_values(previous_V);

	//////////////////////xNa_f//////////////////////////
	float xNa_f_inf = values[0];
	float inv_tau_xNa_f = values[1];

	//////////////////////yNa_f//////////////////////////
	float yNa_f_inf = values[2];
	float inv_tau_yNa_f = values[3];

	//////////////////////xNa_r//////////////////////////
	float xNa_r_inf = values[4];
	float inv_tau_xNa_r = values[5];

	//////////////////////yNa_r//////////////////////////
	float yNa_r_inf = values[6];
	float inv_tau_yNa_r = values[7];

	//////////////////////xNa_p//////////////////////////
	float xNa_p_inf = values[8];
	float inv_tau_xNa_p = values[9];

	//////////////////////xK_V//////////////////////////
	float xK_V_inf = values[10];
	float inv_tau_xK_V = values[11];

	//////////////////////xK_A//////////////////////////
	float xK_A_inf = values[12];
	float inv_tau_xK_A = values[13];

	//////////////////////yK_A//////////////////////////
	float yK_A_inf = values[14];
	float inv_tau_yK_A = values[15];

	//////////////////////xK_IR//////////////////////////
	float xK_IR_inf = values[16];
	float inv_tau_xK_IR = values[17];

	//////////////////////xK_Ca//////////////////////////
	float aux_xK_Ca = values[18];
	float inv_aux_xK_Ca = values[19];
	float alpha_xK_Ca = (Q10_30*2.5f)/(1.0f + aux_xK_Ca/NeuronState[14]);	//NOOOOOOOOOOOO
	float beta_xK_Ca = (Q10_30*1.5f)/(1.0f + NeuronState[14]*inv_aux_xK_Ca);	//NOOOOOOOOOOOO
	float xK_Ca_inf = alpha_xK_Ca / (alpha_xK_Ca + beta_xK_Ca);
	float inv_tau_xK_Ca = (alpha_xK_Ca + beta_xK_Ca);

	//////////////////////xCa//////////////////////////
	float xCa_inf = values[20];
	float inv_tau_xCa = values[21];

	//////////////////////yCa//////////////////////////
	float yCa_inf = values[22];
	float inv_tau_yCa = values[23];

	//////////////////////xK_sl//////////////////////////
	float xK_sl_inf = values[24];
	float inv_tau_xK_sl = values[25];


	float gNa_f = gMAXNa_f * NeuronState[xNa_f_index]*NeuronState[xNa_f_index]*NeuronState[xNa_f_index] * NeuronState[yNa_f_index];
	float gNa_r = gMAXNa_r * NeuronState[xNa_r_index] * NeuronState[yNa_r_index];
	float gNa_p= gMAXNa_p * NeuronState[xNa_p_index];
	float gK_V  = gMAXK_V * NeuronState[xK_V_index]*NeuronState[xK_V_index]*NeuronState[xK_V_index]*NeuronState[xK_V_index];
	float gK_A  = gMAXK_A * NeuronState[xK_A_index]*NeuronState[xK_A_index]*NeuronState[xK_A_index] * NeuronState[yK_A_index];
	float gK_IR = gMAXK_IR * NeuronState[xK_IR_index];
	float gK_Ca = gMAXK_Ca * NeuronState[xK_Ca_index];
	float gCa    = gMAXCa * NeuronState[xCa_index]*NeuronState[xCa_index] * NeuronState[yCa_index];
	float gK_sl  = gMAXK_sl * NeuronState[xK_sl_index];

	AuxNeuronState[xNa_f_index] = (xNa_f_inf - NeuronState[xNa_f_index]) * inv_tau_xNa_f;
	AuxNeuronState[yNa_f_index] = (yNa_f_inf - NeuronState[yNa_f_index]) * inv_tau_yNa_f;
	AuxNeuronState[xNa_r_index] = (xNa_r_inf - NeuronState[xNa_r_index]) * inv_tau_xNa_r;
	AuxNeuronState[yNa_r_index] = (yNa_r_inf - NeuronState[yNa_r_index]) * inv_tau_yNa_r;
	 AuxNeuronState[xNa_p_index]=(xNa_p_inf - NeuronState[xNa_p_index]) * inv_tau_xNa_p;
	 AuxNeuronState[xK_V_index] = (xK_V_inf - NeuronState[xK_V_index]) * inv_tau_xK_V;
	 AuxNeuronState[xK_A_index]=(xK_A_inf  - NeuronState[xK_A_index]) * inv_tau_xK_A;
	 AuxNeuronState[yK_A_index]=(yK_A_inf  - NeuronState[yK_A_index]) * inv_tau_yK_A;
	 AuxNeuronState[xK_IR_index] = (xK_IR_inf - NeuronState[xK_IR_index]) * inv_tau_xK_IR;
	 AuxNeuronState[xK_Ca_index] = (xK_Ca_inf - NeuronState[xK_Ca_index]) * inv_tau_xK_Ca;
	 AuxNeuronState[xCa_index] = (xCa_inf - NeuronState[xCa_index]) * inv_tau_xCa;
	 AuxNeuronState[yCa_index] = (yCa_inf - NeuronState[yCa_index]) * inv_tau_yCa;
	 AuxNeuronState[xK_sl_index]=(xK_sl_inf - NeuronState[xK_sl_index]) * inv_tau_xK_sl;
	 AuxNeuronState[Ca_index]=(-gCa*(previous_V-VCa)/(2*F*A*d) - (betaCa*(NeuronState[Ca_index] - Ca0)));
	 AuxNeuronState[V_m_index]=(current+
		 gNa_f*(VNa - previous_V) + gNa_r*(VNa - previous_V) +
		 gNa_p*(VNa - previous_V) + gK_V*(VK - previous_V) +
		 gK_A*(VK - previous_V) + gK_IR*(VK - previous_V) +
		 gK_Ca*(VK - previous_V) + gCa*(VCa - previous_V) +
		 gK_sl*(VK - previous_V) + gLkg1*(VLkg1 - previous_V) +
		 gLkg2*(VLkg2 - previous_V))*inv_Cm;
}



void EgidioGranuleCell_TimeDriven::EvaluateTimeDependentEquation(float * NeuronState, int index, int elapsed_time_index){
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

void EgidioGranuleCell_TimeDriven::Calculate_conductance_exp_values(int index, float elapsed_time){
	//excitatory synapse.
	Set_conductance_exp_values(index, 0, exp(-elapsed_time/this->tau_exc));
	//inhibitory synapse.
	Set_conductance_exp_values(index, 1, exp(-elapsed_time/this->tau_inh));
	//nmda synapse.
	Set_conductance_exp_values(index, 2, expf(-elapsed_time/this->tau_nmda));
}


bool EgidioGranuleCell_TimeDriven::CheckSynapseType(Interconnection * connection){
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
			cout << "Synapses type " << Type << " of neuron model " << EgidioGranuleCell_TimeDriven::GetName() << " must receive spikes. The source model generates currents." << endl;
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
				cout << "Synapses type " << Type << " of neuron model " << EgidioGranuleCell_TimeDriven::GetName() << " must receive current. The source model generates spikes." << endl;
				return false;
			}
		}
	}
	cout << "Neuron model " << EgidioGranuleCell_TimeDriven::GetName() << " does not support input synapses of type " << Type << ". Just defined " << N_TimeDependentNeuronState << " synapses types." << endl;
	return false;
}

std::map<std::string,boost::any> EgidioGranuleCell_TimeDriven::GetParameters() const {
	// Return a dictionary with the parameters
	std::map<std::string,boost::any> newMap = TimeDrivenNeuronModel::GetParameters();
	return newMap;
}

std::map<std::string, boost::any> EgidioGranuleCell_TimeDriven::GetSpecificNeuronParameters(int index) const noexcept(false){
	return GetParameters();
}

void EgidioGranuleCell_TimeDriven::SetParameters(std::map<std::string, boost::any> param_map) noexcept(false){

	// Search for the parameters in the dictionary
	TimeDrivenNeuronModel::SetParameters(param_map);

	//Set the new g_nmda_inf values based on the e_exc and e_inh parameters
	Generate_g_nmda_inf_values();

	return;
}

IntegrationMethod * EgidioGranuleCell_TimeDriven::CreateIntegrationMethod(ModelDescription imethodDescription) noexcept(false){
	return IntegrationMethodFactory<EgidioGranuleCell_TimeDriven>::CreateIntegrationMethod(imethodDescription, (EgidioGranuleCell_TimeDriven*) this);
}

std::map<std::string,boost::any> EgidioGranuleCell_TimeDriven::GetDefaultParameters() {
	// Return a dictionary with the parameters
	std::map<std::string,boost::any> newMap = TimeDrivenNeuronModel::GetDefaultParameters<EgidioGranuleCell_TimeDriven>();
	return newMap;
}

NeuronModel* EgidioGranuleCell_TimeDriven::CreateNeuronModel(ModelDescription nmDescription){
	EgidioGranuleCell_TimeDriven * nmodel = new EgidioGranuleCell_TimeDriven();
	nmodel->SetParameters(nmDescription.param_map);
	return nmodel;
}

ModelDescription EgidioGranuleCell_TimeDriven::ParseNeuronModel(std::string FileName) noexcept(false){
	FILE *fh;
	ModelDescription nmodel;
	nmodel.model_name = EgidioGranuleCell_TimeDriven::GetName();
	long Currentline = 0L;
	fh=fopen(FileName.c_str(),"rt");
	if(!fh) {
		throw EDLUTFileException(TASK_EGIDIO_GRANULE_CELL_TIME_DRIVEN_LOAD, ERROR_NEURON_MODEL_OPEN, REPAIR_NEURON_MODEL_NAME, Currentline, FileName.c_str());
	}

	Currentline = 1L;
	skip_comments(fh, Currentline);
	try {
		ModelDescription intMethodDescription = TimeDrivenNeuronModel::ParseIntegrationMethod<EgidioGranuleCell_TimeDriven>(fh, Currentline);
		nmodel.param_map["int_meth"] = boost::any(intMethodDescription);
	} catch (EDLUTException exc) {
		throw EDLUTFileException(exc, Currentline, FileName.c_str());
	}

	nmodel.param_map["name"] = boost::any(EgidioGranuleCell_TimeDriven::GetName());

	fclose(fh);

	return nmodel;
}

std::string EgidioGranuleCell_TimeDriven::GetName(){
	return "EgidioGranuleCell_TimeDriven";
}

std::map<std::string, std::string> EgidioGranuleCell_TimeDriven::GetNeuronModelInfo() {
	// Return a dictionary with the parameters
	std::map<std::string, std::string> newMap;
	newMap["info"] = std::string("CPU Time-driven complex neuron model representing a cerebellar granular cell with fifteen differential equations(membrane potential (v) and several ionic-channel variables) and four types of input synapses: AMPA (excitatory), GABA (inhibitory), NMDA (excitatory) and external input current (set on pA)");
	newMap["int_meth"] = std::string("Integraton method dictionary (from the list of available integration methods)");
	return newMap;
}
