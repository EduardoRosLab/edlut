/***************************************************************************
 *                           RK4.h                                         *
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

#ifndef RK4_H_
#define RK4_H_

/*!
 * \file RK4.h
 *
 * \author Francisco Naveros
 * \date May 2013
 *
 * This file declares a class which implement the fourth order Runge Kutta integration method. This class implement a fixed step
 * integration method.
 */

#include "./FixedStep.h"


class TimeDrivenNeuronModel;

/*!
 * \class RK4
 *
 * \brief RK4 integration methods in CPU
 *
 * This class abstracts the behavior of a fourth order Runge Kutta integration method for neurons in a 
 * time-driven spiking neural network.
 * It includes internal model functions which define the behavior of integration methods
 * (initialization, calculate next value, ...).
 *
 * \author Francisco Naveros
 * \date May 2013
 */
class RK4 : public FixedStep {
	protected:

	public:

		/*!
		 * \brief These vectors are used as auxiliar vectors.
		*/
		float * AuxNeuronState;
		float * AuxNeuronState1;
		float * AuxNeuronState2;
		float * AuxNeuronState3;
		float * AuxNeuronState4;

		/*!
		 * \brief Constructor of the class with 4 parameter.
		 *
		 * It generates a new fourth order Runge-Kutta object.
		 *
		 * \param N_neuronStateVariables number of state variables for each cell.
		 * \param N_differentialNeuronState number of state variables witch are calculate with a differential equation for each cell.
		 * \param N_timeDependentNeuronState number of state variables witch are calculate with a time dependent equation for each cell.
		 * \param N_CPU_thread number of OpenMP thread used.
		 */
		RK4(int N_neuronStateVariables, int N_differentialNeuronState, int N_timeDependentNeuronState, int N_CPU_thread);

		/*!
		 * \brief Class destructor.
		 *
		 * It destroys an object of this class.
		 */
		~RK4();
		

		/*!
		 * \brief It calculate the next neural state varaibles of the model.
		 *
		 * It calculate the next neural state varaibles of the model.
		 *
		 * \param index Index of the cell inside the neuron model for method with memory (e.g. BDF).
		 * \param Model The NeuronModel.
		 * \param NeuronState neuron state variables of one neuron.
		 * \param elapsed_time integration time step.
		 * \param CPU_thread_index index of the OpenMP thread.
		 */
		virtual void NextDifferentialEcuationValue(int index, TimeDrivenNeuronModel * Model, float * NeuronState, double elapsed_time, int CPU_thread_index);


		/*!
		 * \brief It prints the integration method info.
		 *
		 * It prints the current integration method characteristics.
		 *
		 * \param out The stream where it prints the information.
		 *
		 * \return The stream after the printer.
		 */
		virtual ostream & PrintInfo(ostream & out);


		/*!
		 * \brief It initialize the state of the integration method for method with memory (e.g. BDF).
		 *
		 * It initialize the state of the integration method for method with memory (e.g. BDF).
		 *
		 * \param N_neuron number of neuron in the neuron model.
		 * \param inicialization vector with initial values.
		 */
		void InitializeStates(int N_neurons, float * initialization){};


		/*!
		 * \brief It reset the state of the integration method for method with memory (e.g. BDF).
		 *
		 * It reset the state of the integration method for method with memory (e.g. BDF).
		 *
		 * \param index indicate witch neuron must be reseted.
		 */
		void resetState(int index){};
};

#endif /* RK4_H_ */
