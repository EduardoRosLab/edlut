/***************************************************************************
 *                           SRMState.h                                    *
 *                           -------------------                           *
 * copyright            : (C) 2010 by Jesus Garrido                        *
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

#ifndef SRMSTATE_H_
#define SRMSTATE_H_

/*!
 * \file SRMState.h
 *
 * \author Jesus Garrido
 * \date February 2010
 *
 * This file declares a class which implements the state of a cell which
 * stores the last activity happened.
 */

#include "BufferedState.h"

/*!
 * \class SRMState
 *
 * \brief Spiking response model based on activity buffer.
 *
 * This class abstracts the state of a cell in a SRM Model.
 *
 * \author Jesus Garrido
 * \date February 2010
 */
class SRMState: public BufferedState {
	private:
		/*!
		 * \brief Time since last spike fired.
		 */
		double LastSpikeTime;

	public:
		/*!
		 * \brief Default constructor with parameters.
		 *
		 * It generates a new state of a cell.
		 *
		 * \param NumVariables Number of the state variables this model needs.
		 * \param BufferAmplitude Time in which the activity will be stored.
		 * \param MaxSize Maximum number of elements which can be simultaneously stored.
		 */
		SRMState(unsigned int NumVariables, float BufferAmpl, unsigned int MaxSize);

		/*!
		 * \brief Copies constructor.
		 *
		 * It generates a new objects which copies the parameter.
		 *
		 * \param OldState State being copied.
		 */
		SRMState(const SRMState & OldState);

		/*!
		 * \brief Class destructor.
		 *
		 * It destroys an object of this class.
		 */
		virtual ~SRMState();

		/*!
		 * \brief Add elapsed time to spikes.
		 *
		 * It adds the elapsed time to spikes.
		 *
		 * \param ElapsedTime The time since the last update.
		 */
		virtual void AddElapsedTime(float ElapsedTime);

		/*!
		 * \brief It adds a new fired spike to the state.
		 *
		 * It adds a new fired spike to the state.
		 */
		void NewFiredSpike();

		/*!
		 * \brief It gets the time since the last spike was fired.
		 *
		 * It gets the time since the last spike was fired.
		 *
		 * \return The time since the last spike fired.
		 */
		double GetLastSpikeTime();

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
};

#endif /* SRMSTATE_H_ */
