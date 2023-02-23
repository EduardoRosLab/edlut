/***************************************************************************
 *                           LIFTimeDrivenModel_IS_GPU2.cuh                *
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

#ifndef LIFTIMEDRIVENMODEL_IS_GPU2_H_
#define LIFTIMEDRIVENMODEL_IS_GPU2_H_

/*!
 * \file LIFTimeDrivenModel_IS_GPU.cuh
 *
 * \author Francisco Naveros
 * \date December 2015
 *
 * This file declares a class which abstracts a Leaky Integrate-And-Fire neuron model with one 
 * differential equation, three time dependent equations (conductances) and one external current. This model is
 * implemented in GPU.
 */

#include "./TimeDrivenNeuronModel_GPU2.cuh"
#include "integration_method/IntegrationMethod_GPU2.cuh"

#include "integration_method/IntegrationMethodFactory_GPU2.cuh"

//Library for CUDA
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

/*!
 * \class LIFTimeDrivenModel_IS_GPU2
 *
 *
 * \brief Leaky Integrate-And-Fire Time-Driven neuron model with a membrane potential, three
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

class LIFTimeDrivenModel_IS_GPU2 : public TimeDrivenNeuronModel_GPU2 {
	public:
		/*!
		 * \brief Excitatory reversal potential in V units
		 */
		const float e_exc;

		/*!
		 * \brief Inhibitory reversal potential in V units
		 */
		const float e_inh;

		/*!
		 * \brief Resting potential in V units
		 */
		const float e_leak;

		/*!
		 * \brief Membrane capacitance in F units
		 */
		const float c_m;
		const float inv_c_m_nF; // Auxiliar inverse membrane capacitance in 1/nF units

		/*!
		 * \brief Firing threshold in V units
		 */
		const float v_thr;
		
		/*!
		 * \brief AMPA receptor time constant in s units
		 */
		const float tau_exc;
		const float inv_tau_exc;

		/*!
		 * \brief GABA receptor time constant in s units
		 */
		const float tau_inh;
		const float inv_tau_inh;

		/*!
		 * \brief Refractory period in s units
		 */
		const float tau_ref;

		/*!
		* \brief Resting conductance in S units
		*/
		const float g_leak;
		const float g_leak_nS; //Auxiliar resting conductance in nS units
		
		/*!
		 * \brief NMDA receptor time constant in s units
		 */
		const float tau_nmda;
		const float inv_tau_nmda;
		
		/*!
		 * \brief Number of state variables for each cell.
		*/
		const int N_NeuronStateVariables=5;

		/*!
		 * \brief Number of state variables which are calculate with a differential equation for each cell.
		*/
		const int N_DifferentialNeuronState=1;

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
		 * \brief constructor with parameters.
		 *
		 * It generates a new neuron model object.
		 *
		 * \param e_exc e_exc.
		 * \param e_inh e_inh.
		 * \param e_leak e_leak.
		 * \param v_thr v_thr.
		 * \param c_m c_m.
		 * \param tau_exc tau_exc.
		 * \param tau_inh tau_inh.
		 * \param tau_ref tau_ref.
		 * \param g_leak g_leak.
		 * \param tau_nmda.
		 * \param integrationName integration method type.
		 * \param N_neurons number of neurons.
		 * \param Buffer_GPU Gpu auxiliar memory.
		 *
		 */
		__device__ LIFTimeDrivenModel_IS_GPU2(float e_exc, float e_inh, float e_leak, float v_thr, float c_m, float tau_exc, float tau_inh,
				float tau_ref, float g_leak, float tau_nmda, char const* integrationName, int N_neurons, void ** Buffer_GPU):TimeDrivenNeuronModel_GPU2(SecondScale_GPU),
				e_exc(e_exc), e_inh(e_inh), e_leak(e_leak), v_thr(v_thr), c_m(c_m), inv_c_m_nF(1.0f/(c_m*1.e9)), tau_exc(tau_exc), inv_tau_exc(1.0f/tau_exc), 
				tau_inh(tau_inh), inv_tau_inh(1.0f/tau_inh), tau_ref(tau_ref), g_leak(g_leak), g_leak_nS(g_leak*1.e9), tau_nmda(tau_nmda), inv_tau_nmda(1.0f/tau_nmda){

			this->integration_method_GPU2 = IntegrationMethodFactory_GPU2<LIFTimeDrivenModel_IS_GPU2>::loadIntegrationMethod_GPU2(integrationName, Buffer_GPU, this);

			integration_method_GPU2->Calculate_conductance_exp_values();
		}

		/*!
		 * \brief Class destructor.
		 *
		 * It destroys an object of this class.
		 */
		__device__ virtual ~LIFTimeDrivenModel_IS_GPU2(){
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
			if (NeuronState[index] > this->v_thr){
				NeuronState[index] = this->e_leak;
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
			float V=NeuronState[index];
			float g_exc=NeuronState[SizeStates + index];
			float g_inh=NeuronState[2*SizeStates + index];
			float g_nmda = NeuronState[3*SizeStates + index];
			float ext_I = NeuronState[4*SizeStates + index]; // (defined in pA).
			
			if(vectorNeuronState_GPU2->LastSpikeTimeGPU[index]>this->tau_ref){
				float current = 0;
				if (EXC){
					current += g_exc * (this->e_exc - V);
				}
				if (INH){
					current += g_inh * (this->e_inh - V);
				}
				if (NMDA){
					float g_nmda_inf = 1.0f / (1.0f + expf(-62.0f*V)*(1.2f / 3.57f));
					current += g_nmda * g_nmda_inf*(this->e_exc - V);
				}
				float external_current_nA = 0.001 * ext_I; 

				current += external_current_nA; // (defined in nA).

				AuxNeuronState[blockDim.x*blockIdx.x + threadIdx.x]=(current + g_leak_nS * (this->e_leak - V))*this->inv_c_m_nF;
			}else if((vectorNeuronState_GPU2->LastSpikeTimeGPU[index] + elapsed_time)>this->tau_ref){
				float current = 0;
				if (EXC){
					current += g_exc * (this->e_exc - V);
				}
				if (INH){
					current += g_inh * (this->e_inh - V);
				}
				if (NMDA){
					float g_nmda_inf = 1.0f / (1.0f + expf(-62.0f*V)*(1.2f / 3.57f));
					current += g_nmda * g_nmda_inf*(this->e_exc - V);
				}
				float external_current_nA = 0.001 * ext_I;

				current += external_current_nA; // (defined in nA).
				
				float fraction=(this->vectorNeuronState_GPU2->LastSpikeTimeGPU[index]+elapsed_time -this->tau_ref)/elapsed_time;
				AuxNeuronState[blockDim.x*blockIdx.x + threadIdx.x]=fraction*(current + g_leak_nS * (this->e_leak - V))*this->inv_c_m_nF;
			}else{
				AuxNeuronState[blockDim.x*blockIdx.x + threadIdx.x]=0.0f;
			}
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


#endif /* LIFTIMEDRIVENMODEL_IS_GPU2_H_ */
