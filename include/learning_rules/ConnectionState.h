/***************************************************************************
 *                           ConnectionState.h                             *
 *                           -------------------                           *
 * copyright            : (C) 2011 by Jesus Garrido                        *
 * email                : jgarrido@atc.ugr.es                              *
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 3 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

#ifndef CONNECTIONSTATE_H_
#define CONNECTIONSTATE_H_

/*!
 * \file ConnectionState.h
 *
 * \author Jesus Garrido
 * \date October 2011
 *
 * This file declares a class which abstracts the current state of a synaptic connection.
 */

/*!
 * \class ConnectionState
 *
 * \brief Synaptic connection current state.
 *
 * This class abstracts the state of a synaptic connection and defines the state variables of
 * that connection.
 *
 * \author Jesus Garrido
 * \date October 2011
 */

class ConnectionState {

	private:
		/*!
		 * \brief Number of state variables.
		 */
		unsigned int NumberOfVariables;

		/*!
	   	 * \brief Neuron state variables.
	   	 */
	   	float * StateVars;

	   	/*!
	   	 * \brief Last update time
	   	 */
	   	double LastUpdate;

	protected:
	   	/*!
		 * \brief It sets the time when the last update happened.
		 *
		 * It sets the time when the last update happened.
		 *
		 * \param NewUpdateTime The time when the last update happened.
		 */
		void SetLastUpdateTime(double NewUpdateTime);


	public:

		/*!
		 * \brief Default constructor with parameters.
		 *
		 * It generates a new state of a connection.
		 *
		 * \param NumVariables Number of the state variables this model needs.
		 */
		ConnectionState(unsigned int NumVariables);

		/*!
		 * \brief It sets the state variable in a specified position.
		 *
		 * It sets the state variable in a specified position.
		 *
		 * \param position The position of the state variable.
		 * \param NewValue The new value of that state variable.
		 */
		void SetStateVariableAt(unsigned int position,float NewValue);

		/*!
		 * \brief Class destructor.
		 *
		 * It destroys an object of this class.
		 */
		virtual ~ConnectionState();

		/*!
		 * \brief It gets the number of state variables.
		 *
		 * It gets the number of state variables.
		 *
		 * \return The number of state variables of this model.
		 */
		unsigned int GetNumberOfVariables();

		/*!
		 * \brief It gets the state variable in a specified position.
		 *
		 * It gets the state variable in a specified position.
		 *
		 * \param position The position of the state variable.
		 * \return The value of the position-th state variable.
		 */
		float GetStateVariableAt(unsigned int position);

		/*!
		 * \brief It gets the time when the last update happened.
		 *
		 * It gets the time when the last update happened.
		 *
		 * \return The time when the last update happened.
		 */
		double GetLastUpdateTime();

		/*!
		 * \brief It gets the number of variables that you can print in this state.
		 *
		 * It gets the number of variables that you can print in this state.
		 *
		 * \return The number of variables that you can print in this state.
		 */
		virtual unsigned int GetNumberOfPrintableValues();

		/*!
		 * \brief It gets a value to be printed from this state.
		 *
		 * It gets a value to be printed from this state.
		 *
		 * \return The value at position-th position in this state.
		 */
		virtual double GetPrintableValuesAt(unsigned int position);

		/*!
		 * \brief It gets the value of the accumulated presynaptic activity.
		 *
		 * It gets the value of the accumulated presynaptic activity.
		 *
		 * \return The accumulated presynaptic activity.
		 */
		virtual float GetPresynapticActivity() = 0;

		/*!
		 * \brief It gets the value of the accumulated postsynaptic activity.
		 *
		 * It gets the value of the accumulated postsynaptic activity.
		 *
		 * \return The accumulated postsynaptic activity.
		 */
		virtual float GetPostsynapticActivity() = 0;

		/*!
		 * \brief Add elapsed time to spikes.
		 *
		 * It adds the elapsed time to spikes.
		 *
		 * \param ElapsedTime The time since the last update.
		 */
		virtual void AddElapsedTime(float ElapsedTime) = 0;


		/*!
		 * \brief It implements the behaviour when it transmits a spike.
		 *
		 * It implements the behaviour when it transmits a spike. It must be implemented
		 * by any inherited class.
		 */
		virtual void ApplyPresynapticSpike() = 0;

		/*!
		 * \brief It implements the behaviour when the target cell fires a spike.
		 *
		 * It implements the behaviour when it the target cell fires a spike. It must be implemented
		 * by any inherited class.
		 */
		virtual void ApplyPostsynapticSpike() = 0;

};

#endif /* CONNECTIONSTATE_H_ */

