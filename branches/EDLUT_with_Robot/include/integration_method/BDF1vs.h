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
 * This class abstracts the behavior of a variable step BDF1 integration method for neurons in a 
 * time-driven spiking neural network.
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


		/*!
		 * \brief This vector stores the difference between previous neuron state variable for all neurons. This 
		 * one is used as a memory.
		*/
		float * D;

		/*!
		 * \brief This vector stores a copy of vector D before it is modiffied. It can be used to restore the value of D.
		*/
		float * OriginalD;



		/*!
		 * \This integration method uses a Runge-Kutta method to calculate the first two step. After that, 
		 * it uses the BDF method. This vector indicates for each neuron if the method is in the firsts two steps 
		 * (state = 0, state = 1) and it must use a Runge-Kutta method or conversely, if it is in the other steps
		 * (state = 2) and it must use the BDF method. When the state of a neuron is reseted, its state variable 
		 * is reseted.
		*/
		int * State;

		/*!
		 * \brief This vector stores a copy of vector State before it is modiffied. It can be used to restore the value of State.
		*/
		int * OriginalState;


		/*!
		 * \brief Constructor with parameters.
		 *
		 * It generates a new BDF1vs object.
		 *
		 * \param NewModel time driven neuron model associated to this integration method.
		 * \param N_neuronStateVariables total number of state variable for each neuron
		 * \param N_differentialNeuronState number of state variables that are diffined by a differential ecuation.
		 * \param N_timeDependentNeuronState number of state variables that are not diffined by a differential ecuation.
		 */
		BDF1vs(TimeDrivenNeuronModel * NewModel, int N_neuronStateVariables, int N_differentialNeuronState, int N_timeDependentNeuronState);

		/*!
		 * \brief Class destructor.
		 *
		 * It destroys an object of this class.
		 */
		~BDF1vs();
		
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
		void InitializeStates(int N_neurons, float * initialization);

		/*!
		 * \brief It reset the state of the integration method for method with memory (e.g. BDF1ad, BDF2, BDF3, etc.).
		 *
		 * It reset the state of the integration method for method with memory (e.g. BDF1ad, BDF2, BDF3, etc.).
		 *
		 * \param index indicate witch neuron must be reseted.
		 */
		void resetState(int index);

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
		 *
		 * NOTE: this method it is never used due to this method is always used as part of the BDF1ad integration method.
		 */
		void loadParameter(FILE *fh, long * Currentline) throw (EDLUTFileException){};


		
		/*!
		 * \brief when the prediction made by the integration method does not reach the tolerance target, the prediction is discarded,
		 *  the state is restored and a new prediction it is made.
		 *
		 * when the prediction made by the integration method does not reach the tolerance target, the prediction is discarded,
		 * the state is restored and a new prediction it is made.
		 *
		 * \param index indicate witch neuron must be restored.
		 */
		void ReturnToOriginalState(int index);

};

#endif /* BDF1VS_H_ */
