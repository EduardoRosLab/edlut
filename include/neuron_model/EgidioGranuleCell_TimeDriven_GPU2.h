/***************************************************************************
 *                           EgidioGranuleCell_TimeDriven_GPU.h            *
 *                           -------------------                           *
 * copyright            : (C) 2011 by Francisco Naveros                    *
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

#ifndef EGIDIOGRANULECELL_TIMEDRIVEN_GPU2_H_
#define EGIDIOGRANULECELL_TIMEDRIVEN_GPU2_H_


/*!
 * \file EgidioGranuleCell_TimeDriven_GPU.h
 *
 * \author Francisco Naveros 
 * \date May 2013
 *
 * This file declares a class which abstracts a Leaky Integrate-And-Fire neuron model for a cerebellar 
 * granule cell. This neuron model has 15 differential equations and 2 time dependent equations (conductances).
 * This model is implemented in GPU.
 */

#include "./TimeDrivenNeuronModel_GPU2.h"
#include "../../include/integration_method/IntegrationMethod_GPU2.h"
#include "../../include/integration_method/LoadIntegrationMethod_GPU2.h"

//Library for CUDA
#include <cutil_inline.h>

//This neuron model is implemented in milisecond. EDLUT is implemented in second and it is necesary to
//use this constant in order to adapt this model to EDLUT.
#define ms_to_s 1000

/*!
 * \class LIFTimeDrivenModel
 *
 * \brief Leaky Integrate-And-Fire Time-Driven neuron model
 *
 * This class abstracts the behavior of a neuron in a time-driven spiking neural network.
 * It includes internal model functions which define the behavior of the model
 * (initialization, update of the state, synapses effect, next firing prediction...).
 * This is only a virtual function (an interface) which defines the functions of the
 * inherited classes.
 *
 * \author Jesus Garrido
 * \date January 2011
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

		const float temper;

		// gating-kinetic correction for experiment temperature=20ºC;
		const float Q10_20;
		// gating-kinetic correction for experiment temperature=22ºC;
		const float Q10_22;
		// experiment temperature=30ºC;
		const float Q10_30;
		// experiment temperature=6.3ºC;
		const float Q10_6_3;

		const float I_inj_abs;

		// Injected Current in absolute terms = -10pA;
		// Cell membrane area = pi*2*r*L = pi*9.76*9.76 = 299.26058um^2 = 299.26058*10^-8cm^2;
		// Injected current per area unit = -I_inj_abs/ 299.26058*10^-8cm^2 = I_inj;
		const float I_inj;

		const float eexc;
		const float einh;
		const float texc;
		const float tinh;
		const float vthr;

		/*!
		 * \brief Number of state variables for each cell.
		*/
		static const int N_NeuronStateVariables=17;

		/*!
		 * \brief Number of state variables witch are calculate with a differential equation for each cell.
		*/
		static const int N_DifferentialNeuronState=15;

		/*!
		 * \brief Number of state variables witch are calculate with a time dependent equation for each cell.
		*/
		static const int N_TimeDependentNeuronState=2;


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
			return (1000*(R*(temper + 273.15)/F)/z*log(co/ci));
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
			if (abs(x/y)<1e-06){
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
		 * \param Total_N_thread total number of CUDA thread.
		 * \param Buffer_GPU Gpu auxiliar memory.
		 *
		 */
		__device__ EgidioGranuleCell_TimeDriven_GPU2(float GMAXNa_f, float GMAXNa_r, float GMAXNa_p, float GMAXK_V,
			float GMAXK_A,float GMAXK_IR,float GMAXK_Ca,float GMAXCa,float GMAXK_sl, char const* integrationName, 
			int N_neurons, int Total_N_thread, void ** Buffer_GPU):TimeDrivenNeuronModel_GPU2(), gMAXNa_f(GMAXNa_f),
			gMAXNa_r(GMAXNa_r), gMAXNa_p(GMAXNa_p), gMAXK_V(GMAXK_V), gMAXK_A(GMAXK_A), gMAXK_IR(GMAXK_IR), 
			gMAXK_Ca(GMAXK_Ca), gMAXCa(GMAXCa), gMAXK_sl(GMAXK_sl), gLkg1(5.68e-5), gLkg2(2.17e-5), VNa(87.39),
			VK(-84.69), VLkg1(-58), VLkg2(-65), V0_xK_Ai(-46.7), K_xK_Ai(-19.8), V0_yK_Ai(-78.8), K_yK_Ai(8.4),
			V0_xK_sli(-30), B_xK_sli(6), F(96485.309), A(1e-04), d(0.2), betaCa(1.5), Ca0(1e-04), R(8.3134),
			cao(2), Cm(1.0e-3), temper(30), Q10_20 ( pow(3,((temper-20)/10))), Q10_22 ( pow(3,((temper-22)/10))),
			Q10_30 ( pow(3,((temper-30)/10))), Q10_6_3 ( pow(3,((temper-6.3)/10))), I_inj_abs(11e-12)/*I_inj_abs(0)*/,
			I_inj(-I_inj_abs*1000/299.26058e-8), eexc(0.0), einh(-80), texc(0.5), tinh(10), vthr(-0.25){
					
			integrationMethod_GPU2=LoadIntegrationMethod_GPU2::loadIntegrationMethod_GPU2(integrationName, N_NeuronStateVariables, N_DifferentialNeuronState, N_TimeDependentNeuronState, Total_N_thread, Buffer_GPU);
		}

		/*!
		 * \brief Class destructor.
		 *
		 * It destroys an object of this class.
		 */
		__device__ ~EgidioGranuleCell_TimeDriven_GPU2(){
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
		__device__ void UpdateState(int index, float * AuxStateGPU, float * StateGPU, double * LastUpdateGPU, double * LastSpikeTimeGPU, bool * InternalSpikeGPU, int SizeStates, double CurrentTime)
		{
			double elapsed_time =CurrentTime - LastUpdateGPU[index];

			LastSpikeTimeGPU[index]+=elapsed_time;
			double last_spike=LastSpikeTimeGPU[index];

			StateGPU[15*SizeStates + index]+=AuxStateGPU[0*SizeStates +index];
			StateGPU[16*SizeStates + index]+=AuxStateGPU[1*SizeStates + index];

			bool spike = false;

			float previous_V=StateGPU[14*SizeStates + index];
			integrationMethod_GPU2->NextDifferentialEcuationValue(index, SizeStates, this, StateGPU, elapsed_time);
			if(StateGPU[14*SizeStates + index]>vthr && previous_V<vthr){
				LastSpikeTimeGPU[index]=0;
				spike = true;
			}

			InternalSpikeGPU[index]=spike;
			LastUpdateGPU[index]=CurrentTime;
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
		__device__ void EvaluateDifferentialEcuation(int index, int SizeStates, float * NeuronState, float * AuxNeuronState){
			float previous_V=NeuronState[14*SizeStates + index];

			float VCa=nernst(NeuronState[13*SizeStates + index],cao,2,temper);
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
			float alphaxK_Ca = (Q10_30*2.5)/(1+(0.0015*exp(-previous_V/11.765))/NeuronState[13*SizeStates + index]);
			float betaxK_Ca   = (Q10_30*1.5)/(1+NeuronState[13*SizeStates + index]/(0.00015*exp(-previous_V/11.765)));
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
			float gNa_f = gMAXNa_f * NeuronState[0*SizeStates + index]*NeuronState[0*SizeStates + index]*NeuronState[0*SizeStates + index] * NeuronState[1*SizeStates + index];
			float gNa_r = gMAXNa_r * NeuronState[2*SizeStates + index] * NeuronState[3*SizeStates + index];
			float gNa_p= gMAXNa_p * NeuronState[4*SizeStates + index];
			float gK_V  = gMAXK_V * NeuronState[5*SizeStates + index]*NeuronState[5*SizeStates + index]*NeuronState[5*SizeStates + index]*NeuronState[5*SizeStates + index];
			float gK_A  = gMAXK_A * NeuronState[6*SizeStates + index]*NeuronState[6*SizeStates + index]*NeuronState[6*SizeStates + index] * NeuronState[7*SizeStates + index];
			float gK_IR = gMAXK_IR * NeuronState[8*SizeStates + index];
			float gK_Ca=gMAXK_Ca * NeuronState[9*SizeStates + index];
			float gCa    = gMAXCa * NeuronState[10*SizeStates + index]*NeuronState[10*SizeStates + index] * NeuronState[11*SizeStates + index];
			float gK_sl  = gMAXK_sl * NeuronState[12*SizeStates + index];

			int offset1=gridDim.x * blockDim.x;
			int offset2=blockDim.x*blockIdx.x + threadIdx.x;

			 AuxNeuronState[0*offset1 + offset2]=ms_to_s*(xNa_f_inf  - NeuronState[0*SizeStates + index])*inv_tauxNa_f;
			 AuxNeuronState[1*offset1 + offset2]=ms_to_s*(yNa_f_inf  - NeuronState[1*SizeStates + index])*inv_tauyNa_f;
			 AuxNeuronState[2*offset1 + offset2]=ms_to_s*(xNa_r_inf  - NeuronState[2*SizeStates + index])*inv_tauxNa_r;
			 AuxNeuronState[3*offset1 + offset2]=ms_to_s*(yNa_r_inf  - NeuronState[3*SizeStates + index])*inv_tauyNa_r;
			 AuxNeuronState[4*offset1 + offset2]=ms_to_s*(xNa_p_inf - NeuronState[4*SizeStates + index])*inv_tauxNa_p;
			 AuxNeuronState[5*offset1 + offset2]=ms_to_s*(xK_V_inf  - NeuronState[5*SizeStates + index])*inv_tauxK_V;
			 AuxNeuronState[6*offset1 + offset2]=ms_to_s*(xK_A_inf  - NeuronState[6*SizeStates + index])*inv_tauxK_A;
			 AuxNeuronState[7*offset1 + offset2]=ms_to_s*(yK_A_inf  - NeuronState[7*SizeStates + index])*inv_tauyK_A;
			 AuxNeuronState[8*offset1 + offset2]=ms_to_s*(xK_IR_inf - NeuronState[8*SizeStates + index])*inv_tauxK_IR;
			 AuxNeuronState[9*offset1 + offset2]=ms_to_s*(xK_Ca_inf - NeuronState[9*SizeStates + index])*inv_tauxK_Ca;
			 AuxNeuronState[10*offset1 + offset2]=ms_to_s*(xCa_inf    - NeuronState[10*SizeStates + index])*inv_tauxCa;
			 AuxNeuronState[11*offset1 + offset2]=ms_to_s*(yCa_inf    - NeuronState[11*SizeStates + index])*inv_tauyCa;
			 AuxNeuronState[12*offset1 + offset2]=ms_to_s*(xK_sl_inf-NeuronState[12*SizeStates + index])*inv_tauxK_sl;
			 AuxNeuronState[13*offset1 + offset2]=ms_to_s*(-gCa*(previous_V-VCa)/(2*F*A*d) - (betaCa*(NeuronState[13*SizeStates + index] - Ca0)));
			 AuxNeuronState[14*offset1 + offset2]=ms_to_s*((-1/Cm)*((NeuronState[15*SizeStates + index]/299.26058e-8) * (previous_V - eexc) + (NeuronState[16*SizeStates + index]/299.26058e-8) * (previous_V - einh)+gNa_f*(previous_V-VNa)+gNa_r*(previous_V-VNa)+gNa_p*(previous_V-VNa)+gK_V*(previous_V-VK)+gK_A*(previous_V-VK)+gK_IR*(previous_V-VK)+gK_Ca*(previous_V-VK)+gCa*(previous_V-VCa)+gK_sl*(previous_V-VK)+gLkg1*(previous_V-VLkg1)+gLkg2*(previous_V-VLkg2)+I_inj));
		}

		
		/*!
		 * \brief It evaluates the time depedendent ecuation in NeuronState for elapsed_time and it stores the results in NeuronState.
		 *
		 * It evaluates the time depedendent ecuation in NeuronState for elapsed_time and it stores the results in NeuronState.
		 *
		 * \param index index inside the NeuronState vector.
		 * \param SizeStates number of element in NeuronState vector.
		 * \param NeuronState value of the neuron state variables where time dependent equations are evaluated.
		 * \param elapsed_time integration time step.
		 */
		__device__ void EvaluateTimeDependentEcuation(int index, int SizeStates,float * NeuronState, double elapsed_time){
			NeuronState[15*SizeStates + index]*= exp(-(ms_to_s*elapsed_time/this->texc));
			NeuronState[16*SizeStates + index]*= exp(-(ms_to_s*elapsed_time/this->tinh));
		}



};

#endif /* EGIDIOGRANULECELL_TIMEDRIVEN_GPU2_H_ */
