/***************************************************************************
 *                           TimeDrivenPurkinjeCell_GPU2.h                 *
 *                           -------------------                           *
 * copyright            : (C) 2015 by Richard Carrill, Niceto Luque and    *
						  Francisco Naveros	                               *
 * email                : rcarrillo@ugr.es, nluque@ugr.es and		       *
						  fnaveros@ugr.es    	                           *
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 3 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

#ifndef TIMEDRIVENPURKINJECELL_GPU2_H_
#define TIMEDRIVENPURKINJECELL_GPU2_H_

/*!
 * \file TimeDrivenPurkinjeCell_GPU.h
 *
 * \author Richard Carrillo
 * \author Niceto Luque
 * \author Francisco Naveros
 * \date December 2015
 *
 * This file declares a class which implements a Purkinje cell model. This model is
 * implemented in GPU.
 */

#include "./TimeDrivenNeuronModel_GPU2.cuh"
#include "integration_method/IntegrationMethod_GPU2.cuh"

#include "integration_method/IntegrationMethodFactory_GPU2.cuh"

//Library for CUDA
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

/*!
 * \class TimeDrivenPurkinjeCell_GPU2.h
 *
 * \brief Time driven neuron model with a membrane potential, two current channels, three conductances
 * (AMPA, GABA, NMDA) and one external current. This model is implemented in GPU.
 *
 * \author Richard Carrillo
 * \author Niceto Luque
 * \author Francisco Naveros
 * \date December 2015
 */

class TimeDrivenPurkinjeCell_GPU2 : public TimeDrivenNeuronModel_GPU2 {
	public:

		/*!
		* \brief leak current in mS/cm^2 units
		*/
		const float g_leak;

		/*!
		* \brief high-threshold noninactivating calcium conducance in mS/cm^2 units
		*/
		const float g_Ca;

		/*!
		* \brief muscarinic receptor suppressed potassium conductance (or M conductance) in mS/cm^2 units
		*/
		const float g_M;

		/*!
		* \brief Cylinder length of the soma in cm units
		*/
		const float cylinder_length_of_the_soma;


		/*!
		* \brief Radius of the soma in cm units
		*/
		const float radius_of_the_soma;

		/*!
		* \brief Cell area in cm^2 units
		*/
		const float area;
		const float inv_area;

		/*!
		* \brief Membrane capacitance in uF/cm^2 units
		*/
		const float c_m;
		const float inv_c_m;

		/*!
		* \brief Peak amplitude in mV units
		*/
		const float spk_peak;

		/*!
		* \brief Excitatory reversal potential in mV units
		*/
		const float e_exc;

		/*!
		* \brief Inhibitory reversal potential in mV units
		*/
		const float e_inh;

		/*!
		* \brief Firing threshold in mV units
		*/
		const float v_thr;

		/*!
		* \brief Resting potential in mV units
		*/
		const float e_leak;

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
		* \brief Refractory period in ms units
		*/
		const float tau_ref;
		const float tau_ref_0_5;
		const float inv_tau_ref_0_5;

		/*!
		 * \brief Number of state variables for each cell.
		*/
		const int N_NeuronStateVariables=7;

		/*!
		 * \brief Number of state variables which are calculate with a differential equation for each cell.
		*/
		const int N_DifferentialNeuronState=3;

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
		 * \param g_leak;
		 * \param g_Ca;
		 * \param g_M;
		 * \param cylinder_length_of_the_soma;
		 * \param radius_of_the_soma;
		 * \param area;
		 * \param c_m;
		 * \param spk_peak;
		 * \param e_exc;
		 * \param e_inh;
		 * \param v_thr;
		 * \param e_leak;
		 * \param tau_exc;
		 * \param tau_inh;
		 * \param tau_nmda;
		 * \param tau_ref;
		 * \param integrationName integration method type.
		 * \param N_neurons number of neurons.
		 * \param Total_N_thread total number of CUDA thread.
		 * \param Buffer_GPU Gpu auxiliar memory.
		 *
		 */
		__device__ TimeDrivenPurkinjeCell_GPU2(float new_g_leak, float new_g_Ca, float new_g_M, float new_cylinder_length_of_the_soma,
		    float new_radius_of_the_soma, float new_area,float new_c_m, float new_spk_peak, float new_e_exc, float new_e_inh,
			float new_v_thr, float new_e_leak, float new_tau_exc, float new_tau_inh, float new_tau_nmda, float new_tau_ref, 
			char const* integrationName,	int N_neurons, void ** Buffer_GPU):TimeDrivenNeuronModel_GPU2(MilisecondScale_GPU),
			g_leak(new_g_leak),g_Ca(new_g_Ca), g_M(new_g_M), cylinder_length_of_the_soma(new_cylinder_length_of_the_soma), 
			radius_of_the_soma(new_radius_of_the_soma),	area(new_area), inv_area(1.0f/new_area), 
			c_m(new_c_m), inv_c_m(1.0f / new_c_m), spk_peak(new_spk_peak), e_exc(new_e_exc), e_inh(new_e_inh), v_thr(new_v_thr), 
			e_leak(new_e_leak), tau_exc(new_tau_exc), inv_tau_exc(1.0f/new_tau_exc), tau_inh(new_tau_inh),	inv_tau_inh(1.0f/new_tau_inh), 
			tau_nmda(new_tau_nmda),	inv_tau_nmda(1.0f/new_tau_nmda), tau_ref(new_tau_ref), tau_ref_0_5(new_tau_ref*0.5f), 
			inv_tau_ref_0_5(2.0f/new_tau_ref)
		{
			this->integration_method_GPU2 = IntegrationMethodFactory_GPU2<TimeDrivenPurkinjeCell_GPU2>::loadIntegrationMethod_GPU2(integrationName, Buffer_GPU, this);
		
			integration_method_GPU2->Calculate_conductance_exp_values();
		}

		/*!
		 * \brief Class destructor.
		 *
		 * It destroys an object of this class.
		 */
		__device__ virtual ~TimeDrivenPurkinjeCell_GPU2(){
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
				vectorNeuronState_GPU2->InternalSpikeGPU[index] = true;
			}

			float last_spike=time_scale*vectorNeuronState_GPU2->LastSpikeTimeGPU[index];
			if(last_spike < tau_ref){
				if(last_spike <= tau_ref_0_5){
					vectorNeuronState_GPU2->VectorNeuronStates_GPU[index]=v_thr+(spk_peak-v_thr)*(last_spike*inv_tau_ref_0_5);
				}else{
					vectorNeuronState_GPU2->VectorNeuronStates_GPU[index]=spk_peak-(spk_peak-e_leak)*((last_spike-tau_ref_0_5)*inv_tau_ref_0_5);
				}
			}else if((last_spike - tau_ref)<elapsedTimeInNeuronModelScale){
				vectorNeuronState_GPU2->VectorNeuronStates_GPU[index]=e_leak;
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
			float ca=NeuronState[1*SizeStates + index];
			float M=NeuronState[2*SizeStates + index];
			float g_exc=NeuronState[3*SizeStates + index];
			float g_inh=NeuronState[4*SizeStates + index];
			float g_nmda = NeuronState[5*SizeStates + index];
			float ext_I = NeuronState[6*SizeStates + index];
			
			float last_spike=time_scale*vectorNeuronState_GPU2->LastSpikeTimeGPU[index];

			int offset1=gridDim.x * blockDim.x;
			int offset2=blockDim.x*blockIdx.x + threadIdx.x;

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

			//We must transform the external current defined in pA to uA/cm^2
			current = current * (1e-6 * inv_area);


		
			//V
			if(last_spike >= tau_ref){
				AuxNeuronState[0 * offset1 + offset2] = (-g_leak*(V + 70.0f) - g_Ca*ca*ca*(V - 125.0f) - g_M*M*(V + 95.0f) + current) * inv_c_m;
			}else if(last_spike <= tau_ref_0_5){
				AuxNeuronState[0*offset1 + offset2]=(spk_peak-v_thr)*inv_tau_ref_0_5;
			}else{
				AuxNeuronState[0*offset1 + offset2]=(e_leak-spk_peak)*inv_tau_ref_0_5;
			}


			//ca
			float alpha_ca=1.6f/(1+__expf(-0.072f*(V-5.0f)));
			float beta_ca=(0.02f*(V+8.9f))/(__expf((V+8.9f)*0.2f)-1.0f);
			float inv_tau_ca=alpha_ca+beta_ca;

			AuxNeuronState[1*offset1 + offset2]=alpha_ca - ca*inv_tau_ca;


			//M
			float alpha_M=0.3f/(1+__expf((-V-2.0f)*0.2f));
			float beta_M=0.001f*__expf((-V-60.0f)*0.055555555555555f);
			float inv_tau_M=alpha_M+beta_M;

			AuxNeuronState[2*offset1 + offset2]=alpha_M - M*inv_tau_M;
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


#endif /* TIMEDRIVENPURKINJECELL_GPU2_H_ */
