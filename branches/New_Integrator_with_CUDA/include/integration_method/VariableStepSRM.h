/***************************************************************************
 *                           VariableStepSRM.h                             *
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

#ifndef VARIABLESTEPSRM_H_
#define VARIABLESTEPSRM_H_

/*!
 * \file VaraibleStepSRM.h
 *
 * \author Francisco Naveros
 * \date May 2013
 *
 * This file declares a class which implement a variable step integration method for SRM neuron model in CPU.
 * This class predicts the next value of the integration step size for each neuron.
*/

#include "./IntegrationMethod.h"

class TimeDrivenNeuronModel;

/*!
 * \class VaraibleStepSRM
 *
 * \brief Variable step integration methods in CPU for SRM neuron model
 *
 * \author Francisco Naveros
 * \date May 2013
 */
class VariableStepSRM : public IntegrationMethod {
	protected:

	public:

		/*!
		 * \brief Maximum integration step size.
		*/
		float h_max;

		/*!
		 * \brief Minimum integration step size
		*/
		float h_min;

		/*!
		 * \brief Maximun firing probability in SRM neuron model. If the probability is higher than this
		 * limit, the predicted integration step size is reduced.
		*/
		float p_max;

		/*!
		 * \brief Minimum firing probability in SRM neuron model. If the probability is lower than this
		 * limit, the predicted integration step size is increased.
		*/
		float p_min;


		/*!
		 * \brief Default constructor.
		 *
		 * It generates a new IntegrationMethod object.
		 *
		 * \param integrationMehtodType
		 */
		VariableStepSRM();


		/*!
		 * \brief Class destructor.
		 *
		 * It destroys an object of this class.
		 */
		~VariableStepSRM();

		
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
		void NextDifferentialEcuationValue(int index, TimeDrivenNeuronModel * Model, float * NeuronState, double elapsed_time, int CPU_thread_index) ;

		/*!
		 * \brief It prints the integration method info.
		 *
		 * It prints the current integration method characteristics.
		 *
		 * \param out The stream where it prints the information.
		 *
		 * \return The stream after the printer.
		 */
		ostream & PrintInfo(ostream & out);


		/*!
		 * \brief It initialize the state of the integration method for method with memory (e.g. BDF).
		 *
		 * It initialize the state of the integration method for method with memory (e.g. BDF).
		 *
		 * \param N_neuron number of neuron in the neuron model.
		 * \param inicialization vector with initial values.
		 */
		void InitializeStates(int N_neurons, float * inicialization);

		/*!
		 * \brief It gets the integration method tipe (variable step, fixed step).
		 *
		 * It gets the integration method tipe (variable step, fixed step).
		 *
		 * \return the integration method tipe (variable step, fixed step).
		 */
		enum IntegrationMethodType GetMethodType();

		/*!
		 * \brief It reset the state of the integration method for method with memory (e.g. BDF).
		 *
		 * It reset the state of the integration method for method with memory (e.g. BDF).
		 *
		 * \param index indicate witch neuron must be reseted.
		 */
		void resetState(int index) {}


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

#endif /* VARIABLESTEPSRM_H_ */
