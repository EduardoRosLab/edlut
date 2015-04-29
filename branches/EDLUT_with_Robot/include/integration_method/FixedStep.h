/***************************************************************************
 *                           FixedStep.h                                   *
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

#ifndef FIXEDSTEP_H_
#define FIXEDSTEP_H_

/*!
 * \file FixedStep.h
 *
 * \author Francisco Naveros
 * \date April 2013
 *
 * This file declares a class which abstracts all fixed step integration methods in CPU.
 */

#include "./IntegrationMethod.h"


class TimeDrivenNeuronModel;

/*!
 * \class FixedStep
 *
 * \brief Fixed step integration methods
 *
 * This class abstracts the behavior of all fixed step integration method for neurons in a 
 * time-driven spiking neural network.
 * It includes internal model functions which define the behavior of integration methods
 * (initialization, calculate next value, ...).
 * This is only a virtual function (an interface) which defines the functions of the
 * inherited classes.
 *
 * \author Francisco Naveros
 * \date April 2013
 */
class FixedStep : public IntegrationMethod {
	protected:

	public:

		/*!
		 * \brief Constructor with parameters.
		 *
		 * It generates a new FiexedStep object.
		 *
		 * \param NewModel time driven neuron model associated to this integration method.
		 * \param integrationMethodType integration method type.
		 * \param N_neuronStateVariables total number of state variable for each neuron
		 * \param N_differentialNeuronState number of state variables that are diffined by a differential ecuation.
		 * \param N_timeDependentNeuronState number of state variables that are not diffined by a differential ecuation.
		 * \param jacobian set to true if the integration method calculates the jacobian of the neuron model differential 
		 *  equation and must be allocat memory for this one (only used in the BDF methods).
		 * \param inverse set to true if the integration method calculates the inverse of a matrix and must be allocated 
		 *  memory for this method(only used in the BDF methods).
		 */
		FixedStep(TimeDrivenNeuronModel * NewModel, string integrationMethodType, int N_neuronStateVariables, int N_differentialNeuronState, int N_timeDependentNeuronState, bool jacobian, bool inverse);

		/*!
		 * \brief Class destructor.
		 *
		 * It destroys an object of this class.
		 */
		~FixedStep();

		/*!
		 * \brief It calculate the new neural state variables for a defined elapsed_time.
		 *
		 * It calculate the new neural state variables for a defined elapsed_time.
		 *
		 * \param index for method with memory (e.g. BDF1ad, BDF2, BDF3, etc.).
		 * \param NeuronState neuron state variables of one neuron.
		 * \param elapsed_time integration time step.
		 */
		virtual void NextDifferentialEcuationValue(int index, float * NeuronState, float elapsed_time) = 0;

		/*!
		 * \brief It prints the integration method info.
		 *
		 * It prints the current integration method characteristics.
		 *
		 * \param out The stream where it prints the information.
		 *
		 * \return The stream after the printer.
		 */
		virtual ostream & PrintInfo(ostream & out) = 0;


		/*!
		 * \brief It initialize the state of the integration method for method with memory (e.g. BDF1ad, BDF2, BDF3, etc.).
		 *
		 * It initialize the state of the integration method for method with memory (e.g. BDF1ad, BDF2, BDF3, etc.).
		 *
		 * \param N_neuron number of neurons in the neuron model.
		 * \param inicialization vector with initial values.
		 */
		virtual void InitializeStates(int N_neurons, float * inicialization) = 0;


		/*!
		 * \brief It reset the state of the integration method for method with memory (e.g. BDF1ad, BDF2, BDF3, etc.).
		 *
		 * It reset the state of the integration method for method with memory (e.g. BDF1ad, BDF2, BDF3, etc.).
		 *
		 * \param index indicate witch neuron must be reseted.
		 */
		virtual void resetState(int index) = 0;

		/*!
		 * \brief It loads the integration method parameters.
		 *
		 * It loads the integration method parameters from the file that define the parameter of the neuron model.
		 *
		 * \param Pointer to a neuron description file (*.cfg). At the end of this file must be included 
		 *  the integration method type and its parameters.
		 * \param Currentline line inside the neuron description file where start the description of the integration method parameter. 
		 *
		 * \throw EDLUTFileException If something wrong has happened in the file load.
		 */
		void loadParameter(FILE *fh, long * Currentline) throw (EDLUTFileException);

};

#endif /* FIXEDSTEP_H_ */
