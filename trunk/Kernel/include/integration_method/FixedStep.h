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
		 * \brief Constructor of the class with 5 parameter.
		 *
		 * It generates a new FixedStep object.
		 *
		 * \param integrationMethodType Integration method type.
		 * \param N_neuronStateVariables number of state variables for each cell.
		 * \param N_differentialNeuronState number of state variables witch are calculate with a differential equation for each cell.
		 * \param N_timeDependentNeuronState number of state variables witch are calculate with a time dependent equation for each cell.
		 * \param N_CPU_thread number of OpenMP thread used.
		 */
		FixedStep(string integrationMethodType, int N_neuronStateVariables, int N_differentialNeuronState, int N_timeDependentNeuronState, int N_CPU_thread, bool jacobian, bool inverse);

		/*!
		 * \brief Class destructor.
		 *
		 * It destroys an object of this class.
		 */
		~FixedStep();

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
		virtual void NextDifferentialEcuationValue(int index, TimeDrivenNeuronModel * Model, float * NeuronState, float elapsed_time, int CPU_thread_index) = 0;

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
		 * \brief It initialize the state of the integration method for method with memory (e.g. BDF).
		 *
		 * It initialize the state of the integration method for method with memory (e.g. BDF).
		 *
		 * \param N_neuron number of neuron in the neuron model.
		 * \param inicialization vector with initial values.
		 */
		virtual void InitializeStates(int N_neurons, float * inicialization) = 0;

		/*!
		 * \brief It gets the integration method tipe (variable step, fixed step).
		 *
		 * It gets the integration method tipe (variable step, fixed step).
		 *
		 * \return the integration method tipe (variable step, fixed step).
		 */
		virtual enum IntegrationMethodType GetMethodType();

		/*!
		 * \brief It reset the state of the integration method for method with memory (e.g. BDF).
		 *
		 * It reset the state of the integration method for method with memory (e.g. BDF).
		 *
		 * \param index indicate witch neuron must be reseted.
		 */
		virtual void resetState(int index) = 0;

		/*
		 * \brief It load the parameter of the integration method.
		 *
		 * It load the parameter of the integration method.
		 *
		 * \param fh pointer to the neuron model description.
		 * \param Currentline curren line in the file fh.
		*/
		void loadParameter(FILE *fh, long * Currentline) throw (EDLUTFileException);

};

#endif /* FIXEDSTEP_H_ */
