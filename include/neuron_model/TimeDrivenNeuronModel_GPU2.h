/***************************************************************************
 *                           TimeDrivenNeuronModel_GPU2.h                  *
 *                           -------------------                           *
 * copyright            : (C) 2012 by Francisco Naveros                    *
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

#ifndef TIMEDRIVENNEURONMODEL_GPU2_H_
#define TIMEDRIVENNEURONMODEL_GPU2_H_

/*!
 * \file TimeDrivenNeuronModel_GPU.h
 *
 * \author Francisco Naveros
 * \date November 2012
 *
 * This file declares a class which abstracts an time-driven neuron model in a GPU.
 */

class IntegrationMethod_GPU2;

#include "./VectorNeuronState_GPU2.h"



//Library for CUDA
#include <helper_cuda.h>


/*!
 * \class TimeDrivenNeuronModel_GPU
 *
 * \brief Time-Driven Spiking neuron model in a GPU
 *
 * This class abstracts the behavior of a neuron in a time-driven spiking neural network.
 * It includes internal model functions which define the behavior of the model
 * (initialization, update of the state, synapses effect, next firing prediction...).
 * This is only a virtual function (an interface) which defines the functions of the
 * inherited classes.
 *
 * \author Francisco Naveros
 * \date November 2012
 */

class TimeDrivenNeuronModel_GPU2{
	public:

		/*!
		 * \brief integration time step.
		*/
		double TimeDrivenStep_GPU;

		/*!
		 * \brief integration time step.
		*/
		double TimeDrivenStep_GPU_f;

		/*!
		 * \brief integration method.
		*/
		IntegrationMethod_GPU2 * integrationMethod_GPU2;


		/*!
		 * \brief Vector neuron state in GPU.
		*/
		VectorNeuronState_GPU2 * vectorNeuronState_GPU2;


		/*!
		 * \brief Default constructor with parameters.
		 *
		 * It generates a new neuron model object without being initialized.
		 */
		__device__ TimeDrivenNeuronModel_GPU2(double new_TimeDrivenStep_GPU):TimeDrivenStep_GPU(new_TimeDrivenStep_GPU), TimeDrivenStep_GPU_f(new_TimeDrivenStep_GPU){
		}


		/*!
		 * \brief Class destructor.
		 *
		 * It destroys an object of this class.
		 */
		__device__ virtual ~TimeDrivenNeuronModel_GPU2(){
			delete integrationMethod_GPU2;
			delete vectorNeuronState_GPU2;
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
		__device__ virtual void UpdateState(double CurrentTime){
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
		__device__ virtual void EvaluateDifferentialEcuation(int index, int SizeStates, float * NeuronState, float * AuxNeuronState){
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
		__device__ virtual void EvaluateTimeDependentEcuation(int index, int SizeStates, float * NeuronState, float elapsed_time){
		}


		/*!
		 * \brief It creates an vector neuron state object in GPU and store the vector allocated in the GPU by the CPU. 
		 *
		 * It creates an vector neuron state object in GPU and store the vector allocated in the GPU by the CPU.
		 *
		 * \param AuxStateGPU.
		 * \param VectorNeuronStates_GPU.
		 * \param LastUpdateGPU.
		 * \param LastSpikeTimeGPU.
		 * \param InternalSpikeGPU.
		 * \param SizeStates.
		 */
		__device__ void InitializeVectorNeuronState_GPU2(float * AuxStateGPU, float * VectorNeuronStates_GPU, double * LastUpdateGPU, double * LastSpikeTimeGPU, bool * InternalSpikeGPU, int SizeStates){
			vectorNeuronState_GPU2=new VectorNeuronState_GPU2(AuxStateGPU, VectorNeuronStates_GPU, LastUpdateGPU, LastSpikeTimeGPU, InternalSpikeGPU, SizeStates);
		}

};


#endif /* TIMEDRIVENNEURONMODEL_GPU2_H_ */
