/***************************************************************************
 *                           BDFn.h                                        *
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

#ifndef BDFn_H_
#define BDFn_H_

/*!
 * \file BDFn.h
 *
 * \author Francisco Naveros
 * \date May 2013
 *
 * This file declares a class which implement six BDF (Backward Differentiation Formulas) integration methods (from 
 * first order to sixth order BDF integration method). This method implements a progressive implementation of the
 * higher order integration method using the lower order integration mehtod (BDF1->BDF2->...->BDF6). This class 
 * implement a fixed step integration method.
 */

#include "./FixedStep.h"

class TimeDrivenNeuronModel;

/*!
 * \class BDFn
 *
 * \brief BDFn integration methods in CPU
 *
 * This class abstracts the behavior of BDF1,...,BDF6 integration methods for neurons in a 
 * time-driven spiking neural network.
 * It includes internal model functions which define the behavior of integration methods
 * (initialization, calculate next value, ...).
 *
 * \author Francisco Naveros
 * \date May 2013
 */
class BDFn : public FixedStep {
	protected:

	public:

		/*!
		 * \brief These vectors are used as auxiliar vectors.
		*/
		float ** AuxNeuronState;
		float ** AuxNeuronState_p;
		float ** AuxNeuronState_p1;
		float ** AuxNeuronState_c;
		float ** jacnum;
		float ** J;
		float ** inv_J;

		/*!
		 * \brief This vector stores previous neuron state variable for all neuron. This one is used as a memory.
		*/
		float ** PreviousNeuronState;

		/*!
		 * \brief This vector stores the difference between previous neuron state variable for all neuron. This 
		 * one is used as a memory.
		*/
		float ** D;

		/*!
		 * \brief This constant matrix stores the coefficients of each BDF order.
		*/
		const static float Coeficient [7][7];

		/*!
		 * \brief This vector contains the state of each neuron (BDF order).
		*/
		int * state;

		/*!
		 * \brief This value stores the order of the integration method.
		*/
		int BDForder;


		/*!
		 * \brief Constructor of the class with 5 parameter.
		 *
		 * It generates a new BDF object indicating the order of the method.
		 *
		 * \param N_neuronStateVariables number of state variables for each cell.
		 * \param N_differentialNeuronState number of state variables witch are calculate with a differential equation for each cell.
		 * \param N_timeDependentNeuronState number of state variables witch ara calculate with a time dependent equation for each cell.
		 * \param N_CPU_thread number of OpenMP thread used.
		 * \param BDForder BDF order (1, 2, ..., 6).
		 */
		BDFn(int N_neuronStateVariables, int N_differentialNeuronState, int N_timeDependentNeuronState, int N_CPU_thread, int BDForder);

		/*!
		 * \brief Class destructor.
		 *
		 * It destroys an object of this class.
		 */
		~BDFn();
		

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
		 * \param inicialization vector with initial values.
		 */
		void InitializeStates(int N_neurons, float * initialization);


		/*!
		 * \brief It reset the state of the integration method for method with memory (e.g. BDF).
		 *
		 * It reset the state of the integration method for method with memory (e.g. BDF).
		 *
		 * \param index indicate witch neuron must be reseted.
		 */
		void resetState(int index);

};

#endif /* BDFn_H_ */
