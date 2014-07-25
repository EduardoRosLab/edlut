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
		/*!
		 * \brief This object implements a integration method base on a 4º and 5º order Runge Kutta integration method.
		*/
		RK45 * RK;

		/*!
		 * \brief Min error tolerance.
		*/
		float e_min;
		
		/*!
		 * \brief Max error tolerance.
		*/
		float e_max;
		
		/*!
		 * \brief Min integration method step size.
		*/	
		float h_min;
		
		/*!
		 * \brief Max integration method step size.
		*/	
		float h_max;


		/*!
		 * \brief Auxiliar neuron state vector used by the RK45 object to predict the future neuron state for each
		 * neuron associated to this integration mehtod.
		*/
		float * PredictedNeuronState;
		
		/*!
		 * \brief This vector control the prediction validity made in PredicteNeuronState by the RK45
		 * object for each neuron associated to this integration method. If some spike arrives to a specific
		 * neuron, then its prediction it is not valid and must be recalculated.
		*/
		bool * ValidPrediction;


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
		RK45ad(TimeDrivenNeuronModel * NewModel, int N_neuronStateVariables, int N_differentialNeuronState, int N_timeDependentNeuronState);

		/*!
		 * \brief Class destructor.
		 *
		 * It destroys an object of this class.
		 */
		~RK45ad();
		
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

#endif /* RK45AD_H_ */
