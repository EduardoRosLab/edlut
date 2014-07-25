/***************************************************************************
 *                           RK45.h                                         *
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

#ifndef RK45_H_
#define RK45_H_

/*!
 * \file RK45.h
 *
 * \author Francisco Naveros
 * \date May 2013
 *
 * This file declares a class which implement a 4º and 5º order Runge Kutta integration method. This class implement a fixed step
 * integration method.
 */

#include "./FixedStep.h"


class TimeDrivenNeuronModel;

/*!
 * \class RK45
 *
 * \brief RK45 integration methods
 *
 * This class abstracts the behavior of a 4º and 5º order Runge Kutta integration method for neurons in a 
 * time-driven spiking neural network.
 * It includes internal model functions which define the behavior of integration methods
 * (initialization, calculate next value, ...).
 * This is only a virtual function (an interface) which defines the functions of the
 * inherited classes.
 *
 * \author Francisco Naveros
 * \date October 2012
 */
class RK45 : public FixedStep {
	protected:

	public:

		//List of coeficients used
		const float a1, a2, a3, a4, a5;
		const float b1, b2, b3, b4, b5, b6;
		const float c20, c21;
		const float c30, c31, c32;
		const float c40, c41, c42, c43;
		const float c51, c52, c53, c54;
		const float c60, c61, c62, c63, c64, c65;


		/*!
		 * \brief Vector which store the difference between the 4º and 5º order Runge Kutta integration method. This difference is used
		 *  as a tolerance by the RK45ad integration method to adapt the integration step size. 
		*/
		float * epsilon;


		/*!
		 * \brief Constructor with parameters.
		 *
		 * It generates a new fourth and fifth Runge-Kutta object.
		 *
		 * \param NewModel time driven neuron model associated to this integration method.
		 * \param N_neuronStateVariables total number of state variable for each neuron
		 * \param N_differentialNeuronState number of state variables that are diffined by a differential ecuation.
		 * \param N_timeDependentNeuronState number of state variables that are not diffined by a differential ecuation.
		 */
		RK45(TimeDrivenNeuronModel * NewModel, int N_neuronStateVariables, int N_differentialNeuronState, int N_timeDependentNeuronState);

		/*!
		 * \brief Class destructor.
		 *
		 * It destroys an object of this class.
		 */
		~RK45();
		
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
		void resetState(int index){};
};

#endif /* RK45_H_ */
