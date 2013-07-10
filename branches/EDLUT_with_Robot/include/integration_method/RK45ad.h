/***************************************************************************
 *                           RK45ad.h                                         *
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

#ifndef RK45AD_H_
#define RK45AD_H_

/*!
 * \file RK45ad.h
 *
 * \author Francisco Naveros
 * \date October 2012
 *
 * This file declares a class which implement the Runge Kutta 4º order integration method.
 */

#include "./VariableStep.h"
#include "./RK45.h"


class TimeDrivenNeuronModel;

/*!
 * \class RK45ad
 *
 * \brief RK45ad integration methods
 *
 * This class abstracts the behavior of Runge Kutta 4º order integration method for neurons in a 
 * time-driven spiking neural network.
 * It includes internal model functions which define the behavior of integration methods
 * (initialization, calculate next value, ...).
 * This is only a virtual function (an interface) which defines the functions of the
 * inherited classes.
 *
 * \author Francisco Naveros
 * \date October 2012
 */
class RK45ad : public VariableStep {
	protected:

	public:
		RK45 * RK;

		float e_min, e_max, h_min, h_max;

		float * PredictedNeuronState;
		bool * ValidPrediction;


		/*!
		 * \brief Constructor of the class with 4 parameter.
		 *
		 * It generates a new Euler object indicating.
		 *
		 * \param N_neuronStateVariables number of state variables for each cell.
		 * \param N_differentialNeuronState number of state variables witch are calculate with a differential equation for each cell.
		 * \param N_timeDependentNeuronState number of state variables witch are calculate with a time dependent equation for each cell.
		 * \param N_CPU_thread number of OpenMP thread used.
		 */
		RK45ad(int N_neuronStateVariables, int N_differentialNeuronState, int N_timeDependentNeuronState, int N_CPU_thread);

		/*!
		 * \brief Class destructor.
		 *
		 * It destroys an object of this class.
		 */
		~RK45ad();
		
		/*!
		 * \brief It calculate the next value for neural state varaibles of the model.
		 *
		 * It calculate the next value for neural state varaibles of the model.
		 *
		 * \param index for method with memory (e.g. BDF2).
		 * \param Model The NeuronModel.
		 * \param NeuronState neuron state variables of one neuron.
		 * \param NumberOfVariables number of varaibles.
		 * \param NumberOfEcuation number of differential ecuation.
		 * \param elapsed_time integration time step.
		 */
		virtual void NextDifferentialEcuationValue(int index, TimeDrivenNeuronModel * Model, float * NeuronState, float elapsed_time, int CPU_thread_index);

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
		 * \brief It initialize the state of the integration method for method with memory (e.g. BDF2).
		 *
		 * It initialize the state of the integration method for method with memory (e.g. BDF2).
		 *
		 * \param N_neuron number of neuron in the neuron model.
		 * \param NumberOfDifferentialEcuation number of differential ecuation in the neuron model.
		 * \param inicialization vector with initial values.
		 *
		 * \Note: this function it is not necesary for this integration method.
		 */
		void InitializeStates(int N_neurons, float * initialization);

		/*!
		 * \brief It reset the state of the integration method for method with memory (e.g. BDF2).
		 *
		 * It reset the state of the integration method for method with memory (e.g. BDF2).
		 *
		 * \param index indicate witch neuron must be reseted.
		 * \param NumberOfDifferentialEcuation number of differential ecuation in the neuron model.
		 * \param State vector witch indicate the new values.
		 *
		 * \Note: this function it is not necesary for this integration method.
		 */
		void resetState(int index){};


		void loadParameter(FILE *fh, long * Currentline) throw (EDLUTFileException);
};

#endif /* RK45AD_H_ */
