/***************************************************************************
 *                           EgidioGranuleCell_TimeDriven_GPU2.cuh          *
 *                           -------------------                           *
 * copyright            : (C) 2013 by Francisco Naveros                    *
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

#ifndef EGIDIOGRANULECELL_TIMEDRIVEN_GPU2_H_
#define EGIDIOGRANULECELL_TIMEDRIVEN_GPU2_H_


/*!
 * \file EgidioGranuleCell_TimeDriven_GPU.cuh
 *
 * \author Francisco Naveros 
 * \date May 2013
 *
 * This file declares a class which abstracts a Leaky Integrate-And-Fire neuron model for a cerebellar 
 * granule cell. This neuron model has 15 differential equations, 2 time dependent equations (conductances).
 * This model is implemented in GPU.
 */

#include "./TimeDrivenNeuronModel_GPU2.cuh"
#include "integration_method/IntegrationMethod_GPU2.cuh"

#include "integration_method/IntegrationMethodFactory_GPU2.cuh"

//Library for CUDA
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

/*!
 * \class EgidioGranuleCell_TimeDriven_GPU2
 *
 * \brief Leaky Integrate-And-Fire Time-Driven neuron model with fifteen differential equations, three
 * conductances (AMPA, GABA, NMDA) and one external current. This model is implemented in GPU.
 *
 * This class abstracts the behavior of a neuron in a time-driven spiking neural network.
 * It includes internal model functions which define the behavior of the model
 * (initialization, update of the state, synapses effect, next firing prediction...).
 * This is only a virtual function (an interface) which defines the functions of the
 * inherited classes.
 *
 * \author Francisco Naveros
 * \date May 2013
 */
class EgidioGranuleCell_TimeDriven_GPU2 : public TimeDrivenNeuronModel_GPU2 {
	public:

		float gMAXNa_f;
		float gMAXNa_r;
		float gMAXNa_p;
		float gMAXK_V;
		float gMAXK_A;
		float gMAXK_IR;
		float gMAXK_Ca;
		float gMAXCa;
		float gMAXK_sl;

		const float gLkg1;
		const float gLkg2;
		const float VNa;
		const float VK;
		const float VLkg1;
		const float VLkg2;
		const float V0_xK_Ai;
		const float K_xK_Ai;
		const float V0_yK_Ai;
		const float K_yK_Ai;
		const float V0_xK_sli;
		const float B_xK_sli;
		const float F;
		const float A;
		const float d;
		const float betaCa;
		const float Ca0;
		const float R;
		const float cao;
		//Membrane Capacitance = 1uF/cm^2 = 10^-3mF/cm^2;
		const float Cm;
		const float inv_Cm;

		const float temper;

		// gating-kinetic correction for experiment temperature=20ºC;
		const float Q10_20;
		// gating-kinetic correction for experiment temperature=22ºC;
		const float Q10_22;
		// experiment temperature=30ºC;
		const float Q10_30;
		// experiment temperature=6.3ºC;
		const float Q10_6_3;

//		const float I_inj_abs;

		// Injected Current in absolute terms = -10pA;
		// Cell membrane area = pi*2*r*L = pi*9.76*9.76 = 299.26058um^2 = 299.26058*10^-8cm^2;
		// Injected current per area unit = -I_inj_abs/ 299.26058*10^-8cm^2 = I_inj;
//		const float I_inj;

		const float e_exc;
		const float e_inh;
		const float tau_exc;
		const float inv_tau_exc;
		const float tau_inh;
		const float inv_tau_inh;
		const float tau_nmda;
		const float inv_tau_nmda;
		const float v_thr;

		/*!
		 * \brief Number of state variables for each cell.
		*/
		const int N_NeuronStateVariables=19;

		/*!
		 * \brief Number of state variables which are calculate with a differential equation for each cell.
		*/
		const int N_DifferentialNeuronState=15;

		/*!
		 * \brief Number of state variables which are calculate with a time dependent equation for each cell.
		*/
		const int N_TimeDependentNeuronState=4;
		
		/*!
		 * \brief Boolean variable setting in runtime if the neuron model receive each one of the supported input synpase types (N_TimeDependentNeuronState) 
		 */
		bool EXC, INH, NMDA, EXT_I;

		/*!
		* \brief It initilizes the index corresponding to each neural state variable.
		*
		* It initilizes the index corresponding to each neural state variable.
		*/
		__device__ virtual void SetEnabledSynapsis(bool new_EXC, bool new_INH, bool new_NMDA, bool new_EXT_I){
			EXC = new_EXC;
			INH = new_INH;
			NMDA = new_NMDA;
			EXT_I = new_EXT_I;
		}


		/*!
		 * \brief 
		 *
		 * 
		 *
		 * \param ci.
		 * \param co.
		 * \param z.
		 * \param temper temperature.
		 */
		__device__ float nernst(float ci, float co, float z, float temper){
			return (1000*(R*(temper + 273.15f)/F)/z*log(co/ci));
		}


		/*!
		 * \brief 
		 *
		 * 
		 *
		 * \param x.
		 * \param y.
		 */
		__device__ float linoid(float x, float y){
			float f=0.0;
			if (abs(x/y)<1e-06f){
				f=y*(1-x/y/2);
			}else{
				f=x/(exp(x/y)-1);
			}
			return f;
		}


		/*!
		 * \brief constructor with parameters.
		 *
		 * It generates a new neuron model object.
		 *
		 * \param GMAXNa_f gMAXNa_f.
		 * \param GMAXNa_r gMAXNa_r.
		 * \param GMAXK_V gMAXK_V.
		 * \param GMAXK_A gMAXK_A.
		 * \param GMAXK_IR gMAXK_IR.
		 * \param GMAXK_Ca gMAXK_Ca.
		 * \param GMAXCa gMAXCa.
		 * \param GMAXK_sl gMAXK_sl.
		 * \param integrationName integration method type.
		 * \param N_neurons number of neurons.
		 * \param Buffer_GPU Gpu auxiliar memory.
		 *
		 */
		__device__ EgidioGranuleCell_TimeDriven_GPU2(float GMAXNa_f, float GMAXNa_r, float GMAXNa_p, float GMAXK_V,
			float GMAXK_A,float GMAXK_IR,float GMAXK_Ca,float GMAXCa,float GMAXK_sl, char const* integrationName, 
			int N_neurons, void ** Buffer_GPU):TimeDrivenNeuronModel_GPU2(MilisecondScale_GPU), gMAXNa_f(GMAXNa_f),
			gMAXNa_r(GMAXNa_r), gMAXNa_p(GMAXNa_p), gMAXK_V(GMAXK_V), gMAXK_A(GMAXK_A), gMAXK_IR(GMAXK_IR), 
			gMAXK_Ca(GMAXK_Ca), gMAXCa(GMAXCa), gMAXK_sl(GMAXK_sl), gLkg1(5.68e-5f), gLkg2(2.17e-5f), VNa(87.39f),
			VK(-84.69f), VLkg1(-58.0f), VLkg2(-65.0f), V0_xK_Ai(-46.7f), K_xK_Ai(-19.8f), V0_yK_Ai(-78.8f), K_yK_Ai(8.4f),
			V0_xK_sli(-30.0f), B_xK_sli(6.0f), F(96485.309f), A(1e-04f), d(0.2f), betaCa(1.5f), Ca0(1e-04f), R(8.3134f),
			cao(2.0f), Cm(1.0e-3f), inv_Cm(1.0f/1.0e-3f), temper(30.0f), Q10_20(pow(3.0f, ((temper - 20.0f) / 10.0f))), Q10_22(pow(3.0f, ((temper - 22.0f) / 10.0f))),
			Q10_30 ( pow(3.0f,((temper-30.0f)/10.0f))), Q10_6_3 ( pow(3.0f,((temper-6.3f)/10.0f))), e_exc(0.0f), e_inh(-80.0f), tau_exc(0.5f),
			inv_tau_exc(1.0f/0.5f), tau_inh(10.0f), inv_tau_inh(1.0f/10.0f), tau_nmda(15.0f), inv_tau_nmda(1.0f/15.0f), v_thr(0.0f){
					
			this->integration_method_GPU2 = IntegrationMethodFactory_GPU2<EgidioGranuleCell_TimeDriven_GPU2>::loadIntegrationMethod_GPU2(integrationName, Buffer_GPU, this);

			integration_method_GPU2->Calculate_conductance_exp_values();
		}

		/*!
		 * \brief Class destructor.
		 *
		 * It destroys an object of this class.
		 */
		__device__ virtual ~EgidioGranuleCell_TimeDriven_GPU2(){
			delete integration_method_GPU2;
		}


		/*!
		 * \brief Update the neuron state variables.
		 *
		 * It updates the neuron state variables.
		 *
		 * \param index The cell index inside the StateGPU. 
		 * \param AuxStateGPU Auxiliary incremental conductance vector.
		 * \param StateGPU Neural state variables.
		 * \param LastUpdateGPU Last update time
		 * \param LastSpikeTimeGPU Last spike time
		 * \param InternalSpikeGPU In this vector is stored if a neuron must generate an output spike.
		 * \param SizeStates Number of neurons
		 * \param CurrentTime Current time.
		 *
		 * \return True if an output spike have been fired. False in other case.
		 */	
		__device__ void UpdateState(double CurrentTime)
		{
			int index = blockIdx.x * blockDim.x + threadIdx.x;
			while (index<vectorNeuronState_GPU2->SizeStates){
				//EXCITATORY
				if (EXC){
					vectorNeuronState_GPU2->VectorNeuronStates_GPU[(this->N_DifferentialNeuronState) * vectorNeuronState_GPU2->SizeStates + index] += vectorNeuronState_GPU2->AuxStateGPU[index];
				}
				//INHIBITORY
				if (INH){
					vectorNeuronState_GPU2->VectorNeuronStates_GPU[(this->N_DifferentialNeuronState + 1) * vectorNeuronState_GPU2->SizeStates + index] += vectorNeuronState_GPU2->AuxStateGPU[vectorNeuronState_GPU2->SizeStates + index];
				}
				//NMDA
				if (NMDA){
					vectorNeuronState_GPU2->VectorNeuronStates_GPU[(this->N_DifferentialNeuronState + 2) * vectorNeuronState_GPU2->SizeStates + index] += vectorNeuronState_GPU2->AuxStateGPU[2 * vectorNeuronState_GPU2->SizeStates + index];
				}
				//EXTERNAL CURRENT (defined in pA).
				if (EXT_I){
					vectorNeuronState_GPU2->VectorNeuronStates_GPU[(this->N_DifferentialNeuronState + 3) * vectorNeuronState_GPU2->SizeStates + index] = vectorNeuronState_GPU2->AuxStateGPU[3 * vectorNeuronState_GPU2->SizeStates + index];
				}

				index += blockDim.x*gridDim.x;
			}
			
			this->integration_method_GPU2->NextDifferentialEquationValues(vectorNeuronState_GPU2->SizeStates, vectorNeuronState_GPU2->VectorNeuronStates_GPU);
		}


		/*!
		 * \brief It evaluates if a neuron must spike.
		 *
		 * It evaluates if a neuron must spike.
		 *
		 * \param previous_V previous membrane potential
		 * \param NeuronState neuron state variables.
		 * \param index Neuron index inside the neuron model.
 		 * \param elapsedTimeInNeuronModelScale integration method step.
		 * \return It returns if a neuron must spike.
		 */
		__device__ void EvaluateSpikeCondition(float previous_V, float * NeuronState, int index, float elapsedTimeInNeuronModelScale){
			if (previous_V<v_thr && NeuronState[index] >= v_thr){
				vectorNeuronState_GPU2->LastSpikeTimeGPU[index]=0.0;
				this->integration_method_GPU2->resetState(index);
				vectorNeuronState_GPU2->InternalSpikeGPU[index] = true;
			}
		}

		/*!
		 * \brief It evaluates the differential equation in NeuronState and it stores the results in AuxNeuronState.
		 *
		 * It evaluates the differential equation in NeuronState and it stores the results in AuxNeuronState.
		 *
		 * \param index index inside the NeuronState vector.
		 * \param SizeStates number of element in NeuronState vector.
		 * \param NeuronState value of the neuron state variables where differential equations are evaluated.
		 * \param AuxNeuronState results of the differential equations evaluation.
		 */
		__device__ void EvaluateDifferentialEquation(int index, int SizeStates, float * NeuronState, float * AuxNeuronState, float elapsed_time){
			float previous_V=NeuronState[index];
			float g_exc = NeuronState[15 * SizeStates + index];
			float g_inh = NeuronState[16 * SizeStates + index];
			float g_nmda = NeuronState[17 * SizeStates + index];
			float ext_I = NeuronState[18 * SizeStates + index]; // (defined in pA).

			float VCa=nernst(NeuronState[14*SizeStates + index],cao,2,temper);
			float alphaxNa_f = Q10_20*(-0.3f)*linoid(previous_V+19, -10);
			float betaxNa_f  = Q10_20*12*exp(-(previous_V+44)/18.182f);
			float xNa_f_inf    = alphaxNa_f/(alphaxNa_f + betaxNa_f);
			float inv_tauxNa_f     = (alphaxNa_f + betaxNa_f);
			float alphayNa_f = Q10_20*0.105f*exp(-(previous_V+44)/3.333f);
			float betayNa_f   = Q10_20*1.5f/(1+exp(-(previous_V+11)/5));
			float yNa_f_inf    = alphayNa_f/(alphayNa_f + betayNa_f);
			float inv_tauyNa_f     = (alphayNa_f + betayNa_f);
			float alphaxNa_r = Q10_20*(0.00008f-0.00493f*linoid(previous_V-4.48754f,-6.81881f));
			float betaxNa_r   = Q10_20*(0.04752f+0.01558f*linoid(previous_V+43.97494f,0.10818f));
			float xNa_r_inf    = alphaxNa_r/(alphaxNa_r + betaxNa_r);
			float inv_tauxNa_r     = (alphaxNa_r + betaxNa_r);
			float alphayNa_r = Q10_20*0.31836f*exp(-(previous_V+80)/62.52621f);
			float betayNa_r   = Q10_20*0.01014f*exp((previous_V+83.3332f)/16.05379f);
			float yNa_r_inf     = alphayNa_r/(alphayNa_r + betayNa_r);
			float inv_tauyNa_r      = (alphayNa_r + betayNa_r);
			float alphaxNa_p = Q10_30*(-0.091f)*linoid(previous_V+42,-5);
			float betaxNa_p   = Q10_30*0.062f*linoid(previous_V+42,5);
			float xNa_p_inf    = 1.0f/(1.0f+exp(-(previous_V+42)/5));
			float inv_tauxNa_p     = (alphaxNa_p + betaxNa_p)*0.2;
			float alphaxK_V = Q10_6_3*(-0.01f)*linoid(previous_V+25,-10);
			float betaxK_V   = Q10_6_3*0.125f*exp(-0.0125f*(previous_V+35));
			float xK_V_inf    = alphaxK_V/(alphaxK_V + betaxK_V);
			float inv_tauxK_V     = (alphaxK_V + betaxK_V);
			float alphaxK_A = (Q10_20*4.88826f)/(1+exp(-(previous_V+9.17203f)/23.32708f));
			float betaxK_A  = (Q10_20*0.99285f)/exp((previous_V+18.27914f)/19.47175f);
			float xK_A_inf    = 1.0f/(1.0f+exp((previous_V-V0_xK_Ai)/K_xK_Ai));
			float inv_tauxK_A     = (alphaxK_A + betaxK_A);
			float alphayK_A = (Q10_20*0.11042f)/(1.0f+exp((previous_V+111.33209f)/12.8433f));
			float betayK_A   = (Q10_20*0.10353f)/(1.0f+exp(-(previous_V+49.9537f)/8.90123f));
			float yK_A_inf    = 1.0f/(1.0f+exp((previous_V-V0_yK_Ai)/K_yK_Ai));
			float inv_tauyK_A     = (alphayK_A + betayK_A);
			float alphaxK_IR = Q10_20*0.13289f*exp(-(previous_V+83.94f)/24.3902f);
			float betaxK_IR  = Q10_20*0.16994f*exp((previous_V+83.94f)/35.714f);
			float xK_IR_inf    = alphaxK_IR/(alphaxK_IR + betaxK_IR);
			float inv_tauxK_IR     = (alphaxK_IR + betaxK_IR);
			float alphaxK_Ca = (Q10_30*2.5f)/(1.0f+(0.0015f*exp(-previous_V/11.765f))/NeuronState[14*SizeStates + index]);
			float betaxK_Ca   = (Q10_30*1.5f)/(1.0f+NeuronState[14*SizeStates + index]/(0.00015f*exp(-previous_V/11.765f)));
			float xK_Ca_inf    = alphaxK_Ca/(alphaxK_Ca + betaxK_Ca);
			float inv_tauxK_Ca     = (alphaxK_Ca + betaxK_Ca);
			float alphaxCa  = Q10_20*0.04944f*exp((previous_V+29.06f)/15.87301587302f);
			float betaxCa   = Q10_20*0.08298f*exp(-(previous_V+18.66f)/25.641f);
			float xCa_inf    = alphaxCa/(alphaxCa + betaxCa);
			float inv_tauxCa     = (alphaxCa + betaxCa);
			float alphayCa = Q10_20*0.0013f*exp(-(previous_V+48)/18.183f);
			float betayCa   = Q10_20*0.0013f*exp((previous_V+48)/83.33f);
			float yCa_inf    = alphayCa/(alphayCa + betayCa);
			float inv_tauyCa     = (alphayCa + betayCa);
			float alphaxK_sl = Q10_22*0.0033f*exp((previous_V+30)/40);
			float betaxK_sl   = Q10_22*0.0033f*exp(-(previous_V+30)/20);
			float xK_sl_inf    = 1.0f/(1.0f+exp(-(previous_V-V0_xK_sli)/B_xK_sli));
			float inv_tauxK_sl     = (alphaxK_sl + betaxK_sl);
			float gNa_f = gMAXNa_f * NeuronState[1*SizeStates + index]*NeuronState[1*SizeStates + index]*NeuronState[1*SizeStates + index] * NeuronState[2*SizeStates + index];
			float gNa_r = gMAXNa_r * NeuronState[3*SizeStates + index] * NeuronState[4*SizeStates + index];
			float gNa_p= gMAXNa_p * NeuronState[5*SizeStates + index];
			float gK_V  = gMAXK_V * NeuronState[6*SizeStates + index]*NeuronState[6*SizeStates + index]*NeuronState[6*SizeStates + index]*NeuronState[6*SizeStates + index];
			float gK_A  = gMAXK_A * NeuronState[7*SizeStates + index]*NeuronState[7*SizeStates + index]*NeuronState[7*SizeStates + index] * NeuronState[8*SizeStates + index];
			float gK_IR = gMAXK_IR * NeuronState[9*SizeStates + index];
			float gK_Ca=gMAXK_Ca * NeuronState[10*SizeStates + index];
			float gCa    = gMAXCa * NeuronState[11*SizeStates + index]*NeuronState[11*SizeStates + index] * NeuronState[12*SizeStates + index];
			float gK_sl  = gMAXK_sl * NeuronState[13*SizeStates + index];

			int offset1=gridDim.x * blockDim.x;
			int offset2=blockDim.x*blockIdx.x + threadIdx.x;
			
			float current = 0;
			if (EXC){
				current += g_exc * (this->e_exc - previous_V);
			}
			if (INH){
				current += g_inh * (this->e_inh - previous_V);
			}
			if (NMDA){
				float g_nmda_inf = 1.0f / (1.0f + expf(-0.062f*previous_V)*(1.2f / 3.57f));
				current += g_nmda * g_nmda_inf*(this->e_exc - previous_V);
			}
			current += ext_I;
			//We normalize the current.
			//current *= 1e-9f / 299.26058e-8f;
			current *= 3.34156941e-4;

			AuxNeuronState[1*offset1 + offset2]=(xNa_f_inf  - NeuronState[1*SizeStates + index])*inv_tauxNa_f;
			AuxNeuronState[2*offset1 + offset2]=(yNa_f_inf  - NeuronState[2*SizeStates + index])*inv_tauyNa_f;
			AuxNeuronState[3*offset1 + offset2]=(xNa_r_inf  - NeuronState[3*SizeStates + index])*inv_tauxNa_r;
			AuxNeuronState[4*offset1 + offset2]=(yNa_r_inf  - NeuronState[4*SizeStates + index])*inv_tauyNa_r;
			AuxNeuronState[5*offset1 + offset2]=(xNa_p_inf - NeuronState[5*SizeStates + index])*inv_tauxNa_p;
			AuxNeuronState[6*offset1 + offset2]=(xK_V_inf  - NeuronState[6*SizeStates + index])*inv_tauxK_V;
			AuxNeuronState[7*offset1 + offset2]=(xK_A_inf  - NeuronState[7*SizeStates + index])*inv_tauxK_A;
			AuxNeuronState[8*offset1 + offset2]=(yK_A_inf  - NeuronState[8*SizeStates + index])*inv_tauyK_A;
			AuxNeuronState[9*offset1 + offset2]=(xK_IR_inf - NeuronState[9*SizeStates + index])*inv_tauxK_IR;
			AuxNeuronState[10*offset1 + offset2]=(xK_Ca_inf - NeuronState[10*SizeStates + index])*inv_tauxK_Ca;
			AuxNeuronState[11*offset1 + offset2]=(xCa_inf    - NeuronState[11*SizeStates + index])*inv_tauxCa;
			AuxNeuronState[12*offset1 + offset2]=(yCa_inf    - NeuronState[12*SizeStates + index])*inv_tauyCa;
			AuxNeuronState[13*offset1 + offset2]=(xK_sl_inf-NeuronState[13*SizeStates + index])*inv_tauxK_sl;
			AuxNeuronState[14*offset1 + offset2]=(-gCa*(previous_V-VCa)/(2*F*A*d) - (betaCa*(NeuronState[14*SizeStates + index] - Ca0)));
			AuxNeuronState[0 * offset1 + offset2] = (current +
				gNa_f*(VNa - previous_V) + gNa_r*(VNa - previous_V) +
				gNa_p*(VNa - previous_V) + gK_V*(VK - previous_V) +
				gK_A*(VK - previous_V) + gK_IR*(VK - previous_V) +
				gK_Ca*(VK - previous_V) + gCa*(VCa - previous_V) +
				gK_sl*(VK - previous_V) + gLkg1*(VLkg1 - previous_V) +
				gLkg2*(VLkg2 - previous_V))*inv_Cm;
//				(-gNa_f*(previous_V - VNa) - gNa_r*(previous_V - VNa) - gNa_p*(previous_V - VNa) - gK_V*(previous_V - VK) - gK_A*(previous_V - VK) - gK_IR*(previous_V - VK) - gK_Ca*(previous_V - VK) - gCa*(previous_V - VCa) - gK_sl*(previous_V - VK) - gLkg1*(previous_V - VLkg1) - gLkg2*(previous_V - VLkg2) + current) * inv_Cm;
		}

		
		/*!
		 * \brief It evaluates the time depedendent Equation in NeuronState for elapsed_time and it stores the results in NeuronState.
		 *
		 * It evaluates the time depedendent Equation in NeuronState for elapsed_time and it stores the results in NeuronState.
		 *
		 * \param index index inside the NeuronState vector.
		 * \param SizeStates number of element in NeuronState vector.
		 * \param NeuronState value of the neuron state variables where time dependent equations are evaluated.
		 * \param elapsed_time integration time step.
		 * \param elapsed_time_index index inside the conductance_exp_values array.
		 */
		__device__ void EvaluateTimeDependentEquation(int index, int SizeStates, float * NeuronState, float elapsed_time, int elapsed_time_index){
			float limit=1e-9;

			float * Conductance_values=this->Get_conductance_exponential_values(elapsed_time_index);

			if(NeuronState[this->N_DifferentialNeuronState*SizeStates + index]<limit){
				NeuronState[this->N_DifferentialNeuronState*SizeStates + index]=0.0f;
			}else{
				NeuronState[this->N_DifferentialNeuronState*SizeStates + index]*=  Conductance_values[0];
			}
			if(NeuronState[(this->N_DifferentialNeuronState+1)*SizeStates + index]<limit){
				NeuronState[(this->N_DifferentialNeuronState+1)*SizeStates + index]=0.0f;
			}else{
				NeuronState[(this->N_DifferentialNeuronState+1)*SizeStates + index]*=  Conductance_values[1];
			}
			if (NeuronState[(this->N_DifferentialNeuronState+2)*SizeStates + index]<limit){
				NeuronState[(this->N_DifferentialNeuronState+2)*SizeStates + index] = 0.0f;
			}else{
				NeuronState[(this->N_DifferentialNeuronState+2)*SizeStates + index] *= Conductance_values[2];
			}
		}

		/*!
		 * \brief It calculates the conductace exponential value for an elapsed time.
		 *
		 * It calculates the conductace exponential value for an elapsed time.
		 *
		 * \param index elapsed time index .
		 * \param elapses_time elapsed time.
		 */
		__device__ void Calculate_conductance_exp_values(int index, float elapsed_time){
			//excitatory synapse.
			Set_conductance_exp_values(index, 0, expf(-elapsed_time*this->inv_tau_exc));
			//inhibitory synapse.
			Set_conductance_exp_values(index, 1, expf(-elapsed_time*this->inv_tau_inh));
			//nmda synapse.
			Set_conductance_exp_values(index, 2, expf(-elapsed_time*this->inv_tau_nmda));
		}

};

#endif /* EGIDIOGRANULECELL_TIMEDRIVEN_GPU2_H_ */
