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

#ifdef _OPENMP
	#include <omp.h>
#else
	#define omp_get_thread_num() 0
	#define omp_get_num_thread() 1
#endif

//This neuron model is implemented in milisecond. EDLUT is implemented in second and it is necesary to
//use this constant in order to adapt this model to EDLUT.
#define ms_to_s 1000

#include "../../include/spike/EDLUTFileException.h"
#include "../../include/spike/Neuron.h"
#include "../../include/spike/InternalSpike.h"
#include "../../include/spike/PropagatedSpike.h"
#include "../../include/spike/Interconnection.h"

#include "../../include/simulation/Utils.h"

void EgidioGranuleCell_TimeDriven::LoadNeuronModel(string ConfigFile) throw (EDLUTFileException){
	FILE *fh;
	long Currentline = 0L;
	fh=fopen(ConfigFile.c_str(),"rt");
	if(fh){
		Currentline=1L;
		skip_comments(fh,Currentline);
		if(fscanf(fh,"%f",&this->gMAXNa_f)==1){
			skip_comments(fh,Currentline);

			if (fscanf(fh,"%f",&this->gMAXNa_r)==1){
				skip_comments(fh,Currentline);

				if(fscanf(fh,"%f",&this->gMAXNa_p)==1){
					skip_comments(fh,Currentline);

					if(fscanf(fh,"%f",&this->gMAXK_V)==1){
						skip_comments(fh,Currentline);

						if(fscanf(fh,"%f",&this->gMAXK_A)==1){
							skip_comments(fh,Currentline);

							if(fscanf(fh,"%f",&this->gMAXK_IR)==1){
								skip_comments(fh,Currentline);

								if(fscanf(fh,"%f",&this->gMAXK_Ca)==1){
									skip_comments(fh,Currentline);

									if(fscanf(fh,"%f",&this->gMAXCa)==1){
										skip_comments(fh,Currentline);

										if(fscanf(fh,"%f",&this->gMAXK_sl)==1){
											skip_comments(fh,Currentline);

											this->InitialState = (VectorNeuronState *) new VectorNeuronState(17, true);
										}
//TODOS LOS CODIGOS DE ERROR HAY QUE MODIFICARLOS, PORQUE AHORA LOS PARAMETROS QUE SE CARGAN SON OTROS.										
										else {
											throw EDLUTFileException(13,60,3,1,Currentline);
										}
									} else {
										throw EDLUTFileException(13,61,3,1,Currentline);
									}
								} else {
									throw EDLUTFileException(13,62,3,1,Currentline);
								}
							} else {
								throw EDLUTFileException(13,63,3,1,Currentline);
							}
						} else {
							throw EDLUTFileException(13,64,3,1,Currentline);
						}
					} else {
						throw EDLUTFileException(13,65,3,1,Currentline);
					}
				} else {
					throw EDLUTFileException(13,66,3,1,Currentline);
				}
			} else {
				throw EDLUTFileException(13,67,3,1,Currentline);
			}
		} else {
			throw EDLUTFileException(13,68,3,1,Currentline);
		}

		//INTEGRATION METHOD
		this->integrationMethod = LoadIntegrationMethod::loadIntegrationMethod(fh, &Currentline, N_NeuronStateVariables, N_DifferentialNeuronState, N_TimeDependentNeuronState, N_CPU_thread);
	}
}

void EgidioGranuleCell_TimeDriven::SynapsisEffect(int index, VectorNeuronState * State, Interconnection * InputConnection){

	switch (InputConnection->GetType()){
		case 0: {
			float gexc = State->GetStateVariableAt(index,N_DifferentialNeuronState);
			float inc=1e-9*InputConnection->GetWeight();
			State->SetStateVariableAt(index,N_DifferentialNeuronState,gexc+inc);
			break;
		}case 1:{
			float ginh = State->GetStateVariableAt(index,N_DifferentialNeuronState+1);
			float inc=1e-9*InputConnection->GetWeight();
			State->SetStateVariableAt(index,N_DifferentialNeuronState+1,ginh+inc);
			break;
		}
	}
}



EgidioGranuleCell_TimeDriven::EgidioGranuleCell_TimeDriven(string NeuronTypeID, string NeuronModelID): TimeDrivenNeuronModel(NeuronTypeID, NeuronModelID), gMAXNa_f(0), gMAXNa_r(0), gMAXNa_p(0), gMAXK_V(0), gMAXK_A(0), gMAXK_IR(0), gMAXK_Ca(0),
		gMAXCa(0), gMAXK_sl(0),
gLkg1(5.68e-5),
gLkg2(2.17e-5),
VNa(87.39),
VK(-84.69),
VLkg1(-58),
VLkg2(-65),
V0_xK_Ai(-46.7),
K_xK_Ai(-19.8),
V0_yK_Ai(-78.8),
K_yK_Ai(8.4),
V0_xK_sli(-30),
B_xK_sli(6),
F(96485.309),
A(1e-04),
d(0.2),
betaCa(1.5),
Ca0(1e-04),
R(8.3134),
cao(2),
Cm(1.0e-3),
temper(30),
Q10_20 ( pow(3,((temper-20)/10))),
Q10_22 ( pow(3,((temper-22)/10))),
Q10_30 ( pow(3,((temper-30)/10))),
Q10_6_3 ( pow(3,((temper-6.3)/10))),

//This is a constant current which can be externally injected to the cell.
/*I_inj_abs(11e-12)*/I_inj_abs(0),
I_inj(-I_inj_abs*1000/299.26058e-8),
eexc(0.0),
einh(-80),

texc(0.5),
tinh(10),

vthr(-0.25)
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


InternalSpike * EgidioGranuleCell_TimeDriven::ProcessInputSpike(PropagatedSpike *  InputSpike){
	Interconnection * inter = InputSpike->GetSource()->GetOutputConnectionAt(InputSpike->GetTarget());

	Neuron * TargetCell = inter->GetTarget();

	VectorNeuronState * CurrentState = TargetCell->GetVectorNeuronState();

	// Add the effect of the input spike
	this->SynapsisEffect(inter->GetTarget()->GetIndex_VectorNeuronState(),(VectorNeuronState *)CurrentState,inter);

	return 0;
}


float EgidioGranuleCell_TimeDriven::nernst(float ci, float co, float z, float temper){
	return (1000*(R*(temper + 273.15)/F)/z*log(co/ci));
}

float EgidioGranuleCell_TimeDriven::linoid(float x, float y){
	float f=0.0;
	if (abs(x/y)<1e-06){
		f=y*(1-x/y/2);
	}else{
		f=x/(exp(x/y)-1);
	}
	return f;
}



bool EgidioGranuleCell_TimeDriven::UpdateState(int index, VectorNeuronState * State, double CurrentTime){
	
	bool * internalSpike=State->getInternalSpike();
	int Size=State->GetSizeState();
	double last_update;
	double elapsed_time;
	double last_spike;
	bool spike;
	float vm_cou;
	int i;
	float previous_V;
	int CPU_thread_index;

	float * NeuronState;
	if(index==-1){
		#pragma omp parallel for default(none) shared(Size, State, internalSpike, CurrentTime, elapsed_time) private(i, last_update, last_spike, spike, vm_cou, NeuronState, previous_V, CPU_thread_index)
		for (int i=0; i< Size; i++){

			last_update = State->GetLastUpdateTime(i);
			elapsed_time = CurrentTime - last_update;
			State->AddElapsedTime(i,elapsed_time);
			last_spike = State->GetLastSpikeTime(i);

			NeuronState=State->GetStateVariableAt(i);

			
			spike = false;


			previous_V=NeuronState[14];
			CPU_thread_index=omp_get_thread_num();
			this->integrationMethod->NextDifferentialEcuationValue(i, this, NeuronState, elapsed_time, CPU_thread_index);
			if(NeuronState[14]>vthr && previous_V<vthr){
				State->NewFiredSpike(i);
				spike = true;
			}


			internalSpike[i]=spike;

			State->SetLastUpdateTime(i,CurrentTime);
		}

		return false;
	}

	else{
		last_update = State->GetLastUpdateTime(index);
		elapsed_time = CurrentTime - last_update;
		State->AddElapsedTime(index,elapsed_time);
		last_spike = State->GetLastSpikeTime(index);

		NeuronState=State->GetStateVariableAt(index);

		
		spike = false;


		previous_V=NeuronState[14];
		this->integrationMethod->NextDifferentialEcuationValue(i, this, NeuronState, elapsed_time, 0);
		if(NeuronState[14]>vthr && previous_V<vthr){
			State->NewFiredSpike(index);
			spike = true;
		}


		internalSpike[index]=spike;

		State->SetLastUpdateTime(index,CurrentTime);
	}

	return false;
}


ostream & EgidioGranuleCell_TimeDriven::PrintInfo(ostream & out){
	return out;
}	


void EgidioGranuleCell_TimeDriven::InitializeStates(int N_neurons){
	//Initial State
	float xNa_f=0.00047309535;
	float yNa_f=1.0;
	float xNa_r=0.00013423511;
	float yNa_r=0.96227829;
	float xNa_p=0.00050020111;
	float xK_V=0.010183001;
	float xK_A=0.15685486;
	float yK_A=0.53565367;
	float xK_IR=0.37337035;
	float xK_Ca=0.00012384122;
	float xCa=0.0021951104;
	float yCa=0.89509747;
	float xK_sl=0.00024031171;
	float Ca=Ca0;
	float V=-80.0;
	float gexc=0.0;
	float ginh=0.0;

	//Initialize neural state variables.
	float initialization[] = {xNa_f,yNa_f,xNa_r,yNa_r,xNa_p,xK_V,xK_A,yK_A,xK_IR,xK_Ca,xCa,yCa,xK_sl,Ca,V,gexc,ginh};
	InitialState->InitializeStates(N_neurons, initialization);

	//Initialize integration method state variables.
	this->integrationMethod->InitializeStates(N_neurons, initialization);
}


void EgidioGranuleCell_TimeDriven::EvaluateDifferentialEcuation(float * NeuronState, float * AuxNeuronState){
	float previous_V=NeuronState[14];

	float VCa=nernst(NeuronState[13],cao,2,temper);
	float alphaxNa_f = Q10_20*(-0.3)*linoid(previous_V+19, -10);
	float betaxNa_f  = Q10_20*12*exp(-(previous_V+44)/18.182);
	float xNa_f_inf    = alphaxNa_f/(alphaxNa_f + betaxNa_f);
	float inv_tauxNa_f     = (alphaxNa_f + betaxNa_f);
	float alphayNa_f = Q10_20*0.105*exp(-(previous_V+44)/3.333);
	float betayNa_f   = Q10_20*1.5/(1+exp(-(previous_V+11)/5));
	float yNa_f_inf    = alphayNa_f/(alphayNa_f + betayNa_f);
	float inv_tauyNa_f     = (alphayNa_f + betayNa_f);
	float alphaxNa_r = Q10_20*(0.00008-0.00493*linoid(previous_V-4.48754,-6.81881));
	float betaxNa_r   = Q10_20*(0.04752+0.01558*linoid(previous_V+43.97494,0.10818));
	float xNa_r_inf    = alphaxNa_r/(alphaxNa_r + betaxNa_r);
	float inv_tauxNa_r     = (alphaxNa_r + betaxNa_r);
	float alphayNa_r = Q10_20*0.31836*exp(-(previous_V+80)/62.52621);
	float betayNa_r   = Q10_20*0.01014*exp((previous_V+83.3332)/16.05379);
	float yNa_r_inf     = alphayNa_r/(alphayNa_r + betayNa_r);
	float inv_tauyNa_r      = (alphayNa_r + betayNa_r);
	float alphaxNa_p = Q10_30*(-0.091)*linoid(previous_V+42,-5);
	float betaxNa_p   = Q10_30*0.062*linoid(previous_V+42,5);
	float xNa_p_inf    = 1/(1+exp(-(previous_V+42)/5));
	float inv_tauxNa_p     = (alphaxNa_p + betaxNa_p)*0.2;
	float alphaxK_V = Q10_6_3*(-0.01)*linoid(previous_V+25,-10);
	float betaxK_V   = Q10_6_3*0.125*exp(-0.0125*(previous_V+35));
	float xK_V_inf    = alphaxK_V/(alphaxK_V + betaxK_V);
	float inv_tauxK_V     = (alphaxK_V + betaxK_V);
	float alphaxK_A = (Q10_20*4.88826)/(1+exp(-(previous_V+9.17203)/23.32708));
	float betaxK_A  = (Q10_20*0.99285)/exp((previous_V+18.27914)/19.47175);

	float xK_A_inf    = 1/(1+exp((previous_V-V0_xK_Ai)/K_xK_Ai));
	float inv_tauxK_A     = (alphaxK_A + betaxK_A);
	float alphayK_A = (Q10_20*0.11042)/(1+exp((previous_V+111.33209)/12.8433));
	float betayK_A   = (Q10_20*0.10353)/(1+exp(-(previous_V+49.9537)/8.90123));
	float yK_A_inf    = 1/(1+exp((previous_V-V0_yK_Ai)/K_yK_Ai));
	float inv_tauyK_A     = (alphayK_A + betayK_A);
	float alphaxK_IR = Q10_20*0.13289*exp(-(previous_V+83.94)/24.3902);
	float betaxK_IR  = Q10_20*0.16994*exp((previous_V+83.94)/35.714);
	float xK_IR_inf    = alphaxK_IR/(alphaxK_IR + betaxK_IR);
	float inv_tauxK_IR     = (alphaxK_IR + betaxK_IR);
	float alphaxK_Ca = (Q10_30*2.5)/(1+(0.0015*exp(-previous_V/11.765))/NeuronState[13]);
	float betaxK_Ca   = (Q10_30*1.5)/(1+NeuronState[13]/(0.00015*exp(-previous_V/11.765)));
	float xK_Ca_inf    = alphaxK_Ca/(alphaxK_Ca + betaxK_Ca);
	float inv_tauxK_Ca     = (alphaxK_Ca + betaxK_Ca);
	float alphaxCa  = Q10_20*0.04944*exp((previous_V+29.06)/15.87301587302);
	float betaxCa   = Q10_20*0.08298*exp(-(previous_V+18.66)/25.641);
	float xCa_inf    = alphaxCa/(alphaxCa + betaxCa);
	float inv_tauxCa     = (alphaxCa + betaxCa);
	float alphayCa = Q10_20*0.0013*exp(-(previous_V+48)/18.183);
	float betayCa   = Q10_20*0.0013*exp((previous_V+48)/83.33);
	float yCa_inf    = alphayCa/(alphayCa + betayCa);
	float inv_tauyCa     = (alphayCa + betayCa);
	float alphaxK_sl = Q10_22*0.0033*exp((previous_V+30)/40);
	float betaxK_sl   = Q10_22*0.0033*exp(-(previous_V+30)/20);
	float xK_sl_inf    = 1/(1+exp(-(previous_V-V0_xK_sli)/B_xK_sli));
	float inv_tauxK_sl     = (alphaxK_sl + betaxK_sl);
	float gNa_f = gMAXNa_f * NeuronState[0]*NeuronState[0]*NeuronState[0] * NeuronState[1];
	float gNa_r = gMAXNa_r * NeuronState[2] * NeuronState[3];
	float gNa_p= gMAXNa_p * NeuronState[4];
	float gK_V  = gMAXK_V * NeuronState[5]*NeuronState[5]*NeuronState[5]*NeuronState[5];
	float gK_A  = gMAXK_A * NeuronState[6]*NeuronState[6]*NeuronState[6] * NeuronState[7];
	float gK_IR = gMAXK_IR * NeuronState[8];
	float gK_Ca=gMAXK_Ca * NeuronState[9];
	float gCa    = gMAXCa * NeuronState[10]*NeuronState[10] * NeuronState[11];
	float gK_sl  = gMAXK_sl * NeuronState[12];

	 AuxNeuronState[0]=ms_to_s*(xNa_f_inf  - NeuronState[0])*inv_tauxNa_f;
	 AuxNeuronState[1]=ms_to_s*(yNa_f_inf  - NeuronState[1])*inv_tauyNa_f;
	 AuxNeuronState[2]=ms_to_s*(xNa_r_inf  - NeuronState[2])*inv_tauxNa_r;
	 AuxNeuronState[3]=ms_to_s*(yNa_r_inf  - NeuronState[3])*inv_tauyNa_r;
	 AuxNeuronState[4]=ms_to_s*(xNa_p_inf - NeuronState[4])*inv_tauxNa_p;
	 AuxNeuronState[5]=ms_to_s*(xK_V_inf  - NeuronState[5])*inv_tauxK_V;
	 AuxNeuronState[6]=ms_to_s*(xK_A_inf  - NeuronState[6])*inv_tauxK_A;
	 AuxNeuronState[7]=ms_to_s*(yK_A_inf  - NeuronState[7])*inv_tauyK_A;
	 AuxNeuronState[8]=ms_to_s*(xK_IR_inf - NeuronState[8])*inv_tauxK_IR;
	 AuxNeuronState[9]=ms_to_s*(xK_Ca_inf - NeuronState[9])*inv_tauxK_Ca;
	 AuxNeuronState[10]=ms_to_s*(xCa_inf    - NeuronState[10])*inv_tauxCa;
	 AuxNeuronState[11]=ms_to_s*(yCa_inf    - NeuronState[11])*inv_tauyCa;
	 AuxNeuronState[12]=ms_to_s*(xK_sl_inf-NeuronState[12])*inv_tauxK_sl;
	 AuxNeuronState[13]=ms_to_s*(-gCa*(previous_V-VCa)/(2*F*A*d) - (betaCa*(NeuronState[13] - Ca0)));
	 AuxNeuronState[14]=ms_to_s*(-1/Cm)*((NeuronState[15]/299.26058e-8) * (previous_V - eexc) + (NeuronState[16]/299.26058e-8) * (previous_V - einh)+gNa_f*(previous_V-VNa)+gNa_r*(previous_V-VNa)+gNa_p*(previous_V-VNa)+gK_V*(previous_V-VK)+gK_A*(previous_V-VK)+gK_IR*(previous_V-VK)+gK_Ca*(previous_V-VK)+gCa*(previous_V-VCa)+gK_sl*(previous_V-VK)+gLkg1*(previous_V-VLkg1)+gLkg2*(previous_V-VLkg2)+I_inj);
}


void EgidioGranuleCell_TimeDriven::EvaluateTimeDependentEcuation(float * NeuronState, double elapsed_time){
	//NeuronState[15]*= exp(-(ms_to_s*elapsed_time/this->texc));
	//NeuronState[16]*= exp(-(ms_to_s*elapsed_time/this->tinh));

	if(NeuronState[15]<1e-30){
		NeuronState[15]=0;
	}else{
		NeuronState[15]*= exp(-(ms_to_s*elapsed_time/this->texc));
	}
	if(NeuronState[16]<1e-30){
		NeuronState[16]=0;
	}else{
		NeuronState[16]*= exp(-(ms_to_s*elapsed_time/this->tinh));
	}
}