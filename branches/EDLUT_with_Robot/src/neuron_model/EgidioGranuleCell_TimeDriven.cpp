/***************************************************************************
 *                           EdidioGranuleCell_TimeDriven.cpp              *
 *                           -------------------                           *
 * copyright            : (C) 2013 by Francisco Naveros                    *
 * email                : fnaveros@atc.ugr.es                              *
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 3 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

#include "../../include/neuron_model/EgidioGranuleCell_TimeDriven.h"
#include "../../include/neuron_model/VectorNeuronState.h"

#include <iostream>
#include <cmath>
#include <string>

#include "../../include/openmp/openmp.h"

//This neuron model is implemented in milisecond. EDLUT is implemented in second and it is necesary to
//use this constant in order to adapt this model to EDLUT.
#define ms_to_s 1000.0f

#include "../../include/spike/EDLUTFileException.h"
#include "../../include/spike/Neuron.h"
#include "../../include/spike/InternalSpike.h"
#include "../../include/spike/PropagatedSpike.h"
#include "../../include/spike/Interconnection.h"

#include "../../include/simulation/Utils.h"


	const float EgidioGranuleCell_TimeDriven::gMAXNa_f=0.013f;
	const float EgidioGranuleCell_TimeDriven::gMAXNa_r=0.0005f;
	const float EgidioGranuleCell_TimeDriven::gMAXNa_p=2.0e-5f;
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
	const float EgidioGranuleCell_TimeDriven::Cm=1.0e-3f;
	const float EgidioGranuleCell_TimeDriven::temper=30.0f;
	const float EgidioGranuleCell_TimeDriven::Q10_20 = pow(3.0f,((temper-20.0f)/10.0f));
	const float EgidioGranuleCell_TimeDriven::Q10_22 = pow(3.0f,((temper-22.0f)/10.0f));
	const float EgidioGranuleCell_TimeDriven::Q10_30 = pow(3.0f,((temper-30.0f)/10.0f));
	const float EgidioGranuleCell_TimeDriven::Q10_6_3 = pow(3.0f,((temper-6.3f)/10.0f));

	const float EgidioGranuleCell_TimeDriven::Max_V=50.0f;
	const float EgidioGranuleCell_TimeDriven::Min_V=-100.0f;

	const float EgidioGranuleCell_TimeDriven::aux=(EgidioGranuleCell_TimeDriven::TableSize-1)/( EgidioGranuleCell_TimeDriven::Max_V - EgidioGranuleCell_TimeDriven::Min_V);

	float * EgidioGranuleCell_TimeDriven::channel_values=Generate_channel_values();



void EgidioGranuleCell_TimeDriven::LoadNeuronModel(string ConfigFile) throw (EDLUTFileException){
	FILE *fh;
	long Currentline = 0L;
	fh=fopen(ConfigFile.c_str(),"rt");
	if(fh){
		Currentline=1L;
		this->State = (VectorNeuronState *) new VectorNeuronState(N_NeuronStateVariables, true);

		//INTEGRATION METHOD
		this->integrationMethod = LoadIntegrationMethod::loadIntegrationMethod((TimeDrivenNeuronModel *)this, fh, &Currentline, N_NeuronStateVariables, N_DifferentialNeuronState, N_TimeDependentNeuronState);
	}else{
		cout<<"EgidioGranuleCell_TimeDriven definition file not defined"<<endl;
	}
}

void EgidioGranuleCell_TimeDriven::SynapsisEffect(int index, Interconnection * InputConnection){
	this->GetVectorNeuronState()->IncrementStateVariableAtCPU(index,N_DifferentialNeuronState+InputConnection->GetType(),1e-9f*InputConnection->GetWeight());
}

EgidioGranuleCell_TimeDriven::EgidioGranuleCell_TimeDriven(string NeuronTypeID, string NeuronModelID): TimeDrivenNeuronModel(NeuronTypeID, NeuronModelID), 
	//This is a constant current which can be externally injected to the cell.
	I_inj_abs(11e-12)/*I_inj_abs(0)*/,
	I_inj(-I_inj_abs*1000/299.26058e-8), eexc(0.0), einh(-80), texc(0.5), tinh(10), vthr(-0.25)
{
}

EgidioGranuleCell_TimeDriven::~EgidioGranuleCell_TimeDriven(void)
{
}

void EgidioGranuleCell_TimeDriven::LoadNeuronModel() throw (EDLUTFileException){
	this->LoadNeuronModel(this->GetModelID()+".cfg");
}


VectorNeuronState * EgidioGranuleCell_TimeDriven::InitializeState(){
	return this->GetVectorNeuronState();
}


InternalSpike * EgidioGranuleCell_TimeDriven::ProcessInputSpike(Interconnection * inter, Neuron * target, double time){

	// Add the effect of the input spike
	this->SynapsisEffect(target->GetIndex_VectorNeuronState(),inter);

	return 0;
}



bool EgidioGranuleCell_TimeDriven::UpdateState(int index, double CurrentTime){
	
	bool * internalSpike=State->getInternalSpike();
	int Size=State->GetSizeState();
	double last_update;
	double elapsed_time;
	float elapsed_time_f;
	double last_spike;
	bool spike;
	float vm_cou;
	int i;
	float previous_V;


	float * NeuronState;
	for (int i=0; i< Size; i++){

		last_update = State->GetLastUpdateTime(i);
		elapsed_time = CurrentTime - last_update;
		elapsed_time_f=elapsed_time;
		State->AddElapsedTime(i,elapsed_time);
		last_spike = State->GetLastSpikeTime(i);

		NeuronState=State->GetStateVariableAt(i);

		
		spike = false;


		previous_V=NeuronState[14];
		this->integrationMethod->NextDifferentialEcuationValue(i, NeuronState, elapsed_time_f);
		if(NeuronState[14]>vthr && previous_V<vthr){
			State->NewFiredSpike(i);
			spike = true;
		}


		internalSpike[i]=spike;

		State->SetLastUpdateTime(i,CurrentTime);
	}

	return false;
}





ostream & EgidioGranuleCell_TimeDriven::PrintInfo(ostream & out){
	return out;
}	


void EgidioGranuleCell_TimeDriven::InitializeStates(int N_neurons, int OpenMPQueueIndex){
	//Initial State
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
	float V=-80.0f;
	float gexc=0.0f;
	float ginh=0.0f;

	//Initialize neural state variables.
	float initialization[] = {xNa_f,yNa_f,xNa_r,yNa_r,xNa_p,xK_V,xK_A,yK_A,xK_IR,xK_Ca,xCa,yCa,xK_sl,Ca,V,gexc,ginh};
	State->InitializeStates(N_neurons, initialization);

	//Initialize integration method state variables.
	this->integrationMethod->InitializeStates(N_neurons, initialization);
}


void EgidioGranuleCell_TimeDriven::EvaluateDifferentialEcuation(float * NeuronState, float * AuxNeuronState, int index){
	float previous_V=NeuronState[14];

	float VCa=nernst(NeuronState[13],cao,2,temper);

	float * values=Get_channel_values(previous_V);
	
	//////////////////////xNa_f//////////////////////////
	float alpha_xNa_f = values[0];
	float inv_tau_xNa_f = values[1];

	//////////////////////yNa_f//////////////////////////
	float alpha_yNa_f = values[2];				
	float inv_tau_yNa_f = values[3];				

	//////////////////////xNa_r//////////////////////////
	float alpha_xNa_r = values[4];
	float inv_tau_xNa_r = values[5];

	//////////////////////yNa_r//////////////////////////
	float alpha_yNa_r = values[6];
	float inv_tau_yNa_r = values[7];

	//////////////////////xNa_p//////////////////////////
	float xNa_p_inf = values[8];						
	float inv_tau_xNa_p = values[9];

	//////////////////////xK_V//////////////////////////
	float alpha_xK_V = values[10];										
	float inv_tau_xK_V = values[11];

	//////////////////////xK_A//////////////////////////
	float xK_A_inf = values[12];
	float inv_tau_xK_A = values[13];

	//////////////////////yK_A//////////////////////////
	float yK_A_inf = values[14];
	float inv_tau_yK_A = values[15];

	//////////////////////xK_IR//////////////////////////
	float alpha_xK_IR = values[16];
	float inv_tau_xK_IR = values[17];

	//////////////////////xK_Ca//////////////////////////
	float aux_xK_Ca = values[18];
	float inv_aux_xK_Ca = values[19];
	float alpha_xK_Ca = (Q10_30*2.5f)/(1.0f + aux_xK_Ca/NeuronState[13]);	//NOOOOOOOOOOOO
	float beta_xK_Ca = (Q10_30*1.5f)/(1.0f + NeuronState[13]*inv_aux_xK_Ca);	//NOOOOOOOOOOOO
	float inv_tau_xK_Ca = (alpha_xK_Ca + beta_xK_Ca);

	//////////////////////xCa//////////////////////////
	float alpha_xCa = values[20];
	float inv_tau_xCa = values[21];

	//////////////////////yCa//////////////////////////
	float alpha_yCa = values[22];
	float inv_tau_yCa = values[23];

	//////////////////////xK_sl//////////////////////////
	float xK_sl_inf = values[24];
	float inv_tau_xK_sl = values[25];


	float gNa_f = gMAXNa_f * NeuronState[0]*NeuronState[0]*NeuronState[0] * NeuronState[1];
	float gNa_r = gMAXNa_r * NeuronState[2] * NeuronState[3];
	float gNa_p= gMAXNa_p * NeuronState[4];
	float gK_V  = gMAXK_V * NeuronState[5]*NeuronState[5]*NeuronState[5]*NeuronState[5];
	float gK_A  = gMAXK_A * NeuronState[6]*NeuronState[6]*NeuronState[6] * NeuronState[7];
	float gK_IR = gMAXK_IR * NeuronState[8];
	float gK_Ca = gMAXK_Ca * NeuronState[9];
	float gCa    = gMAXCa * NeuronState[10]*NeuronState[10] * NeuronState[11];
	float gK_sl  = gMAXK_sl * NeuronState[12];

	 AuxNeuronState[0]=ms_to_s*(alpha_xNa_f - NeuronState[0]*inv_tau_xNa_f);
	 AuxNeuronState[1]=ms_to_s*(alpha_yNa_f - NeuronState[1]*inv_tau_yNa_f);
	 AuxNeuronState[2]=ms_to_s*(alpha_xNa_r - NeuronState[2]*inv_tau_xNa_r);
	 AuxNeuronState[3]=ms_to_s*(alpha_yNa_r - NeuronState[3]*inv_tau_yNa_r);
	 AuxNeuronState[4]=ms_to_s*(xNa_p_inf - NeuronState[4])*inv_tau_xNa_p;
	 AuxNeuronState[5]=ms_to_s*(alpha_xK_V - NeuronState[5]*inv_tau_xK_V);
	 AuxNeuronState[6]=ms_to_s*(xK_A_inf  - NeuronState[6])*inv_tau_xK_A;
	 AuxNeuronState[7]=ms_to_s*(yK_A_inf  - NeuronState[7])*inv_tau_yK_A;
	 AuxNeuronState[8]=ms_to_s*(alpha_xK_IR - NeuronState[8]*inv_tau_xK_IR);
	 AuxNeuronState[9]=ms_to_s*(alpha_xK_Ca - NeuronState[9]*inv_tau_xK_Ca);
	 AuxNeuronState[10]=ms_to_s*(alpha_xCa - NeuronState[10]*inv_tau_xCa);
	 AuxNeuronState[11]=ms_to_s*(alpha_yCa - NeuronState[11]*inv_tau_yCa);
	 AuxNeuronState[12]=ms_to_s*(xK_sl_inf-NeuronState[12])*inv_tau_xK_sl;
	 AuxNeuronState[13]=ms_to_s*(-gCa*(previous_V-VCa)/(2*F*A*d) - (betaCa*(NeuronState[13] - Ca0)));
	 AuxNeuronState[14]=ms_to_s*(-1/Cm)*((NeuronState[15]/299.26058e-8f) * (previous_V - eexc) + (NeuronState[16]/299.26058e-8f) * (previous_V - einh)+gNa_f*(previous_V-VNa)+gNa_r*(previous_V-VNa)+gNa_p*(previous_V-VNa)+gK_V*(previous_V-VK)+gK_A*(previous_V-VK)+gK_IR*(previous_V-VK)+gK_Ca*(previous_V-VK)+gCa*(previous_V-VCa)+gK_sl*(previous_V-VK)+gLkg1*(previous_V-VLkg1)+gLkg2*(previous_V-VLkg2)+I_inj);
}



void EgidioGranuleCell_TimeDriven::EvaluateTimeDependentEcuation(float * NeuronState, float elapsed_time){
	//NeuronState[15]*= ExponentialTable::GetResult(-(ms_to_s*elapsed_time/this->texc));
	//NeuronState[16]*= ExponentialTable::GetResult(-(ms_to_s*elapsed_time/this->tinh));

	if(NeuronState[15]<1e-30){
		NeuronState[15]=0.0f;
	}else{
		NeuronState[15]*= ExponentialTable::GetResult(-(ms_to_s*elapsed_time/this->texc));
	}
	if(NeuronState[16]<1e-30){
		NeuronState[16]=0.0f;
	}else{
		NeuronState[16]*= ExponentialTable::GetResult(-(ms_to_s*elapsed_time/this->tinh));
	}
}


int EgidioGranuleCell_TimeDriven::CheckSynapseTypeNumber(int Type){
	if(Type<N_TimeDependentNeuronState && Type>=0){
		return Type;
	}else{
		cout<<"Neuron model "<<this->GetTypeID()<<", "<<this->GetModelID()<<" does not support input synapses of type "<<Type<<endl;
		return 0;
	}
}
