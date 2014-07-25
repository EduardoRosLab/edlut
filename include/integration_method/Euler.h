/***************************************************************************
 *                           Euler.h                                       *
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

#ifndef EULER_H_
#define EULER_H_

/*!
 * \file Euler.h
 *
 * \author Francisco Naveros
 * \date May 2013
 *
 * This file declares a class which implement the Euler integration method. This class implement a fixed step
 * integration method.
 */

#include "./FixedStep.h"



class TimeDrivenNeuronModel;

/*!
 * \class Euler
 *
 * \brief Euler integration methods in CPU
 * 
 * This class abstracts the behavior of a Euler integration method for neurons in a 
 * time-driven spiking neural network.
 * It includes internal model functions which define the behavior of integration methods
 * (initialization, calculate next value, ...).
 *
 * \author Francisco Naveros
 * \date May 2013
 */
class Euler : public FixedStep {
	protected:

	public:


		/*!
		 * \brief Constructor with parameters.
		 *
		 * It generates a new Euler object.
		 *
		 * \param NewModel time driven neuron model associated to this integration method.
		 * \param N_neuronStateVariables total number of state variable for each neuron
		 * \param N_differentialNeuronState number of state variables that are diffined by a differential ecuation.
		 * \param N_timeDependentNeuronState number of state variables that are not diffined by a differential ecuation.
		 */
		Euler(TimeDrivenNeuronModel * NewModel, int N_neuronStateVariables, int N_differentialNeuronState, int N_timeDependentNeuronState);


		/*!
		 * \brief Class destructor.
		 *
		 * It destroys an object of this class.
		 */
		~Euler();


		/*!
		 * \brief It calculate the new neural state variables for a defined elapsed_time.
		 *
		 * It calculate the new neural state variables for a defined elapsed_time.
		 *
		 * \param index for method with memory (e.g. BDF1ad, BDF2, BDF3, etc.).
		 * \param NeuronState neuron state variables of one neuron.
		 * \param elapsed_time integration time step.
		 */
		virtual void NextDifferentialEcuationValue(int index, float * NeuronState, float elapsed_time);


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
		 * \brief It initialize the state of the integration method for method with memory (e.g. BDF1ad, BDF2, BDF3, etc.).
		 *
		 * It initialize the state of the integration method for method with memory (e.g. BDF1ad, BDF2, BDF3, etc.).
		 *
		 * \param N_neuron number of neurons in the neuron model.
		 * \param inicialization vector with initial values.
		 */
		void InitializeStates(int N_neurons, float * initialization){};

		/*!
		 * \brief It reset the state of the integration method for method with memory (e.g. BDF1ad, BDF2, BDF3, etc.).
		 *
		 * It reset the state of the integration method for method with memory (e.g. BDF1ad, BDF2, BDF3, etc.).
		 *
		 * \param index indicate witch neuron must be reseted.
		 */
		void resetState(int index){};
};

#endif /* EULER_H_ */
