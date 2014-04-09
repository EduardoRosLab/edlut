/***************************************************************************
 *                           IntegrationMethod_GPU2.h                      *
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

#ifndef INTEGRATIONMETHOD_GPU2_H_
#define INTEGRATIONMETHOD_GPU2_H_

/*!
 * \file IntegrationMethod_GPU.h
 *
 * \author Francisco Naveros
 * \date May 2013
 *
 * This file declares a class which abstracts all integration methods in GPU (this class is stored
 * in GPU memory and executed in GPU. All integration methods in GPU are fixed step due to the parallel
 * architecture of this one.
 */

class TimeDrivenNeuronModel_GPU2;

//Library for CUDA
#include <helper_cuda.h>




/*!
 * \class IntegrationMethod_GPU2
 *
 * \brief Integration method in a GPU.
 *
 * This class abstracts the behavior of all integration methods for neurons in GPU in a 
 * time-driven spiking neural network.
 * It includes internal model functions which define the behavior of integration methods
 * (initialization, calculate next value, ...).
 * This is only a virtual function (an interface) which defines the functions of the
 * inherited classes.
 *
 * \author Francisco Naveros
 * \date May 2013
 */

class IntegrationMethod_GPU2 {
	public:

		/*!
		 * \brief Number of state variables for each cell.
		*/
		int N_NeuronStateVariables;

		/*!
		 * \brief Number of state variables witch are calculate with a differential equation for each cell.
		*/
		int N_DifferentialNeuronState;

		/*!
		 * \brief Number of state variables witch are calculate with a time dependent equation for each cell.
		*/
		int N_TimeDependentNeuronState;


		/*!
		 * \brief Constructor of the class with 4 parameter.
		 *
		 * It generates a new IntegrationMethod objectin GPU memory.
		 *
		 * \param N_neuronStateVariables Number of state variables for each cell.
		 * \param N_differentialNeuronState Number of state variables witch are calculate with a differential equation for each cell.
		 * \param N_timeDependentNeuronState Number of state variables witch ara calculate with a time dependent equation for each cell.
		 * \param Total_N_thread Number of thread in GPU (in this method it is not necessary)
		 */
		__device__ IntegrationMethod_GPU2(int N_neuronStateVariables, int N_differentialNeuronState, int N_timeDependentNeuronState, int Total_N_thread):N_NeuronStateVariables(N_neuronStateVariables), N_DifferentialNeuronState(N_differentialNeuronState), N_TimeDependentNeuronState(N_timeDependentNeuronState){
		}


		/*!
		 * \brief Class destructor.
		 *
		 * It destroys an object of this class.
		 */
		__device__ ~IntegrationMethod_GPU2(){
		}

		
		/*!
		 * \brief It calculate the next neural state varaibles of the model.
		 *
		 * It calculate the next neural state varaibles of the model.
		 *
		 * \param index Index of the cell inside the neuron model for method with memory (e.g. BDF).
		 * \param SizeStates Number of neurons
		 * \param Model The NeuronModel.
		 * \param NeuronState Vector of neuron state variables for all neurons.
		 * \param elapsed_time integration time step.
		 */
		__device__ virtual void NextDifferentialEcuationValue(int index, int SizeStates, TimeDrivenNeuronModel_GPU2 * Model, float * NeuronState, float elapsed_time) {
		}


		/*!
		 * \brief It reset the state of the integration method for method with memory (e.g. BDF).
		 *
		 * It reset the state of the integration method for method with memory (e.g. BDF).
		 *
		 * \param index indicate witch neuron must be reseted.
		 *
		 */
		__device__ virtual void resetState(int index){
		}

};

#endif /* INTEGRATIONMETHOD_GPU2_H_ */
