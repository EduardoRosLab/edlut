/***************************************************************************
 *                           BDF1vs.h                                      *
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

#ifndef BDF1VS_H_
#define BDF1VS_H_

/*!
 * \file BDF1vs.h
 *
 * \author Francisco Naveros
 * \date May 2013
 *
 * This file declares a class which implement a variable step first order BDF (Backward Differentiation 
 * Formulas) integration method. This method is used in the adaptative integration method BDF1ad.
 */

#include "./VariableStep.h"

class TimeDrivenNeuronModel;

/*!
 * \class BDF1vs
 *
 * \brief BDF1vs integration methods
 *
 * This class abstracts the behavior of BDF1 integration method for neurons in a 
 * time-driven spiking neural network, using diffentes integration step sizes.
 * It includes internal model functions which define the behavior of integration methods
 * (initialization, calculate next value, ...).
 * This is only a virtual function (an interface) which defines the functions of the
 * inherited classes.
 *
 * \author Francisco Naveros
 * \date May 2013
 */
class BDF1vs : public VariableStep {
	protected:

	public:

		float * AuxNeuronState;
		float * AuxNeuronState_p;
		float * AuxNeuronState_p1;
		float * AuxNeuronState_c;
		float * jacnum;
		float * J;
		float * inv_J;


		/*!
		 * \brief This vector stores previous neuron state variable for all neuron. This one is used as a memory.
		*/
		float * D;
		float * OriginalD;



		/*!
		 * \This integration method uses a Runge-Kutta method to calculate the first two step. After that, 
		 * it uses the BDF method. This vector indicates for each neuron if the method is in the firsts two step 
		 * (state = 0, state = 1) and it must use a Runge-Kutta method or conversely, if it is in the other steps
		 * (state = 2) and it must use the BDF method. When the state of a neuron is reseted, its state variable 
		 * is reseted.
		*/
		int * State;
		int * OriginalState;




		/*!
		 * \brief Default constructor without parameters.
		 *
		 * It generates a new Euler object without being initialized.
		 */
		BDF1vs(int N_neuronStateVariables, int N_differentialNeuronState, int N_timeDependentNeuronState, int N_CPU_thread);

		/*!
		 * \brief Class destructor.
		 *
		 * It destroys an object of this class.
		 */
		~BDF1vs();
		
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
		virtual void NextDifferentialEcuationValue(int index,TimeDrivenNeuronModel * Model, float * NeuronState, float elapsed_time, int CPU_thread_index);

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
		 */
		void resetState(int index);

		void ReturnToOriginalState(int index);

		void loadParameter(FILE *fh, long * Currentline) throw (EDLUTFileException){};

};

#endif /* BDF1VS_H_ */
