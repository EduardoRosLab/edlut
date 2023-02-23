/***************************************************************************
 *                           BifixedStep_GPU2.cuh                          *
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

#ifndef BIFIXEDSTEP_GPU2_H_
#define BIFIXEDSTEP_GPU2_H_

/*!
 * \file BifixedStep_GPU2.cuh
 *
 * \author Francisco Naveros
 * \date May 2015
 *
 * This file declares a class which abstract all multi step integration method in GPU (this class is stored
 * in GPU memory and executed in GPU. 
 */

#include "integration_method/IntegrationMethodFast_GPU2.cuh"


//Library for CUDA
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

/*!
 * \class BifixedStep_GPU2
 *
 * \brief Multi step integration method in GPU.
 * 
 * This class abstracts the behavior of all the Bifixed step integration method for neurons in a 
 * time-driven spiking neural network in GPU.
 * It includes internal model functions which define the behavior of integration methods
 * (initialization, calculate next value, ...).
 *
 * \author Francisco Naveros
 * \date May 2015
 */
template <class Neuron_Model_GPU2>
class BifixedStep_GPU2 : public IntegrationMethodFast_GPU2<Neuron_Model_GPU2> {
	public:

		/*!
		 * \brief Number of multi step in the adapatative zone.
		*/
		int N_BifixedSteps;

		/*!
		 * \brief Elapsed time in neuron model scale of the adaptative zone.
		*/
		float BifixedElapsedTimeInNeuronModelScale;

		/*!
		 * \brief Elapsed time in second of the adaptative zone.
		*/
		float BifixedElapsedTimeInSeconds;


		/*!
		* \brief Constructor of the class with 2 parameter.
		*
		* It generates a new BifixedStep Integration Method object in GPU memory.
		*
		* \param TimeDrivenNeuronModel pointer to the time driven neuron model
		* \param Buffer_GPU This vector contains all the necesary GPU memory which have been reserved in the CPU (this memory
		*	could be reserved directly in the GPU, but this whould suppose some restriction in the amount of memory that can be reserved).
		*/
		__device__ BifixedStep_GPU2(Neuron_Model_GPU2* NewModel, void ** Buffer_GPU) :IntegrationMethodFast_GPU2<Neuron_Model_GPU2>(NewModel, Buffer_GPU){
			float * integration_method_parameters=((float*)Buffer_GPU[0]);
			BifixedElapsedTimeInSeconds=integration_method_parameters[1];
			BifixedElapsedTimeInNeuronModelScale=BifixedElapsedTimeInSeconds*NewModel->time_scale;
			N_BifixedSteps=((int)integration_method_parameters[2]);
		}

		/*!
		 * \brief Class destructor.
		 *
		 * It destroys an object of this class.
		 */
		__device__ virtual ~BifixedStep_GPU2(){
		}
		

		/*!
		 * \brief It calculate the next neural state variables of the model.
		 *
		 * It calculate the next neural state variables of the model.
		 *
		 * \param index Index of the cell inside the neuron model for method with memory (e.g. BDF).
		 * \param SizeStates Number of neurons
		 * \param NeuronState Vector of neuron state variables for all neurons.
		 */
		__device__ virtual void NextDifferentialEquationValues(int index, int SizeStates, float * NeuronState){
		}


		/*!
		* \brief It calculate the next neural state variables of the model.
		*
		* It calculate the next neural state variables of the model.
		*
		* \param SizeStates Number of neurons
		* \param NeuronState Vector of neuron state variables for all neurons.
		*/
		__device__ virtual void NextDifferentialEquationValues(int SizeStates, float * NeuronState) {
		}


		/*!
		 * \brief It reset the state of the integration method for method with memory (e.g. BDF).
		 *
		 * It reset the state of the integration method for method with memory (e.g. BDF).
		 *
		 * \param index indicate which neuron must be reseted.
		 *
		 */
		__device__ virtual void resetState(int index){
		}

		/*!
		 * \brief It calculates the conductance exponential values for time driven neuron models.
		 *
		 * It calculates the conductance exponential values for time driven neuron models.
		 */
		__device__ virtual void Calculate_conductance_exp_values(){
		}


};



#endif /* BIFIXEDSTEP_GPU2_H_ */
