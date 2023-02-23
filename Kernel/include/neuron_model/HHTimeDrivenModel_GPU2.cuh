/***************************************************************************
 *                          HHTimeDrivenModel_GPU2.h                       *
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

#ifndef HHTIMEDRIVENMODEL_GPU2_H_
#define HHTIMEDRIVENMODEL_GPU2_H_

/*!
 * \file HHTimeDrivenModel_GPU2.cuh
 *
 * \author Francisco Naveros
 * \date December 2015
 *
 * This file declares a class which abstracts a Hodgkin and Huxley (HH) neuron model with four 
 * differential equation, trhree time dependent equations (conductances) and one external current. This model is
 * implemented in GPU.
 */

#include "./TimeDrivenNeuronModel_GPU2.cuh"
#include "integration_method/IntegrationMethod_GPU2.cuh"

#include "integration_method/IntegrationMethodFactory_GPU2.cuh"

//Library for CUDA
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

/*!
 * \class HHTimeDrivenModel_GPU2
 *
 *
 * \brief Hodgkin and Huxley (HH) Time-Driven neuron model with a membrane potential, three ion-channel conductances, three
 * conductances (AMPA, GABA, NMDA) and one external current. This model is implemented in GPU.
 *
 * This class abstracts the behavior of a neuron in a time-driven spiking neural network.
 * It includes internal model functions which define the behavior of the model
 * (initialization, update of the state, synapses effect, next firing prediction...).
 * This is only a virtual function (an interface) which defines the functions of the
 * inherited classes.
 *
 * \author Francisco Naveros
 * \date December 2015
 */

class HHTimeDrivenModel_GPU2 : public TimeDrivenNeuronModel_GPU2 {
	public:
		/*!
		 * \brief Excitatory reversal potential in mV units
		 */
		const float e_exc;

		/*!
		 * \brief Inhibitory reversal potential in mV units
		 */
		const float e_inh;

		/*!
		 * \brief Resting potential in mV units
		 */
		const float e_leak;

		/*!
		 * \brief Resting conductance in nS units
		 */
		const float g_leak;

		/*!
		 * \brief Membrane capacitance in pF units
		 */
		const float c_m;
		const float inv_c_m;

		/*!
		* \brief Efective threshold potential in mV units
		*/
		const float v_thr;

		/*!
		 * \brief AMPA receptor time constant in ms units
		 */
		const float tau_exc;
		const float inv_tau_exc;

		/*!
		 * \brief GABA receptor time constant in ms units
		 */
		const float tau_inh;
		const float inv_tau_inh;
		
		/*!
		 * \brief NMDA receptor time constant in ms units
		 */
		const float tau_nmda;
		const float inv_tau_nmda;

		/*!
		* \brief Maximum value of sodium conductance in nS units
		*/
		const float g_na;
		
		/*!
		* \brief Maximum value of potassium conductance in nS units
		*/
		const float g_kd;
		
		/*!
		* \brief Sodium potential in mV units
		*/
		const float e_na;
		
		/*!
		* \brief Potassium potential in mV units
		*/
		const float e_k;


		/*!
		 * \brief Number of state variables for each cell.
		*/
		const int N_NeuronStateVariables=8;

		/*!
		 * \brief Number of state variables which are calculate with a differential equation for each cell.
		*/
		const int N_DifferentialNeuronState=4;

		/*!
		 * \brief Number of state variables which are calculate with a time dependent equation for each cell (EXC, INH, NMDA, EXT_I).
		 */
		const int N_TimeDependentNeuronState = 4;

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
		 * \brief constructor with parameters.
		 *
		 * It generates a new neuron model object.
		 *
		 * \param e_exc e_exc.
		 * \param e_inh e_inh.
		 * \param e_leak e_leak.
		 * \param c_m c_m.
		 * \param tau_exc tau_exc.
		 * \param tau_inh tau_inh.
		 * \param tau_inh tau_nmda.
		 * \param Tref tref.
		 * \param g_leak g_leak.
		 * \param integrationName integration method type.
		 * \param N_neurons number of neurons.
		 * \param Total_N_thread total number of CUDA thread.
		 * \param Buffer_GPU Gpu auxiliar memory.
		 *
		 */
		__device__ HHTimeDrivenModel_GPU2(float new_e_exc, float new_e_inh, float new_e_leak, float new_g_leak, float new_c_m, float new_tau_exc, 
				float new_tau_inh, float new_tau_nmda, float new_g_na, float new_g_kd, float new_e_na, float new_e_k, float new_v_thr,
				char const* integrationName, int N_neurons, void ** Buffer_GPU):TimeDrivenNeuronModel_GPU2(MilisecondScale_GPU),
				e_exc(new_e_exc), e_inh(new_e_inh), e_leak(new_e_leak), g_leak(new_g_leak), c_m(new_c_m), inv_c_m(1.0f/new_c_m), tau_exc(new_tau_exc), 
				inv_tau_exc(1.0f/new_tau_exc), tau_inh(new_tau_inh), inv_tau_inh(1.0f/new_tau_inh), tau_nmda(new_tau_nmda), inv_tau_nmda(1.0f/new_tau_nmda),
				g_na(new_g_na), g_kd(new_g_kd), e_na(new_e_na), e_k(new_e_k),	v_thr(new_v_thr){
			this->integration_method_GPU2 = IntegrationMethodFactory_GPU2<HHTimeDrivenModel_GPU2>::loadIntegrationMethod_GPU2(integrationName, Buffer_GPU, this);
		
			integration_method_GPU2->Calculate_conductance_exp_values();
		}

		/*!
		 * \brief Class destructor.
		 *
		 * It destroys an object of this class.
		 */
		__device__ virtual ~HHTimeDrivenModel_GPU2(){
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
			if (NeuronState[index] > -30.0f && previous_V < -30.0f){
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
			//V
			if (NeuronState[index]<e_k + 1.0f){
				NeuronState[index] = e_k + 1.0f;
			}
			if (NeuronState[index]>e_na - 1.0f){
				NeuronState[index] = e_na - 1.0f;
			}
			//m
			if (NeuronState[SizeStates + index]<0){
				NeuronState[SizeStates + index] = 0.0f;
			}
			if (NeuronState[SizeStates + index]>1){
				NeuronState[SizeStates + index] = 1.0f;
			}
			//h
			if (NeuronState[2 * SizeStates + index]<0){
				NeuronState[2 * SizeStates + index] = 0.0f;
			}
			if (NeuronState[2 * SizeStates + index]>1){
				NeuronState[2 * SizeStates + index] = 1.0f;
			}
			//n
			if (NeuronState[3 * SizeStates + index]<0){
				NeuronState[3 * SizeStates + index] = 0.0f;
			}
			if (NeuronState[3 * SizeStates + index]>0.72){
				NeuronState[3 * SizeStates + index] = 0.72f;
			}
			
			float V=NeuronState[index];
			float m=NeuronState[SizeStates + index];
			float h=NeuronState[2*SizeStates + index];
			float n=NeuronState[3*SizeStates + index];
			float g_exc=NeuronState[4*SizeStates + index];
			float g_inh=NeuronState[5*SizeStates + index];
			float g_nmda = NeuronState[6*SizeStates + index];
			float ext_I = NeuronState[7*SizeStates + index];


			
			float current = 0;
			if (EXC){
				current += g_exc * (this->e_exc - V);
			}
			if (INH){
				current += g_inh * (this->e_inh - V);
			}
			if (NMDA){
				float g_nmda_inf = 1.0f / (1.0f + expf(-0.062f*V)*(1.2f / 3.57f));
				current += g_nmda * g_nmda_inf*(this->e_exc - V);
			}
			current += ext_I; // (defined in pA).
			

			//v			
			AuxNeuronState[blockDim.x*blockIdx.x + threadIdx.x]=(current + g_leak * (this->e_leak - V) + g_na*m*m*m*h*(e_na - V) + g_kd*n*n*n*n*(e_k - V))*this->inv_c_m;
			
			//m
			float alpha_m, beta_m;
			if((13.0f-V+v_thr)!=0.0f){
				alpha_m=0.32f*(13.0f-V+v_thr)/(__expf((13.0f-V+v_thr)*0.25f)-1.0f);
			}else{
				alpha_m=1.279999894092334f;
			}

			if((V-v_thr-40.0f)!=0.0f){
				beta_m=0.28f*(V-v_thr-40.0f)/(__expf((V-v_thr-40.0f)*0.2f)-1.0f);
			}else{
				beta_m=1.400030971093304f;
			}
			AuxNeuronState[blockDim.x*gridDim.x + blockDim.x*blockIdx.x + threadIdx.x]=(alpha_m*(1.0f-m)-beta_m*m);
			
			//h
			float alpha_h=0.128f*__expf((17.0f-V+v_thr)*0.0555555555555555555f);
			float beta_h=4.0f/(1.0f+__expf((40.0f-V+v_thr)*0.2f));
			AuxNeuronState[2*blockDim.x*gridDim.x + blockDim.x*blockIdx.x + threadIdx.x]=(alpha_h*(1-h)-beta_h*h);
			
			//n
			float alpha_n;
			if((15.0f-V+v_thr)!=0.0f){
				alpha_n=0.032f*(15.0f-V+v_thr)/(__expf((15.0f-V+v_thr)*0.2f)-1.0f);
			}else{
				alpha_n=0.1599999867615418f;
			}
			float beta_n=0.5f*__expf((10.0f-V+v_thr)*0.025f);
			AuxNeuronState[3*blockDim.x*gridDim.x + blockDim.x*blockIdx.x + threadIdx.x]=(alpha_n*(1-n)-beta_n*n);
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
				NeuronState[this->N_DifferentialNeuronState*SizeStates + index]*= Conductance_values[0];
			}
			if(NeuronState[(this->N_DifferentialNeuronState+1)*SizeStates + index]<limit){
				NeuronState[(this->N_DifferentialNeuronState+1)*SizeStates + index]=0.0f;
			}else{
				NeuronState[(this->N_DifferentialNeuronState+1)*SizeStates + index]*= Conductance_values[1];
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


#endif /* HHTIMEDRIVENMODEL_GPU2_H_ */
