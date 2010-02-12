/***************************************************************************
 *                           BufferedState.h                               *
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

#ifndef BUFFEREDSTATE_H_
#define BUFFEREDSTATE_H_

/*!
 * \file BufferedState.h
 *
 * \author Jesus Garrido
 * \date February 2010
 *
 * This file declares a class which implements the state of a cell which
 * stores the last activity happened.
 */

#include "NeuronState.h"

#include "../spike/Interconnection.h"

#include <utility>
#include <list>

using namespace std;

/*!
 * \class BufferedState
 *
 * \brief Spiking neuron current state with activity buffer.
 *
 * This class abstracts the state of a cell and stores the last activity happened.
 *
 * \author Jesus Garrido
 * \date February 2010
 */

typedef pair<double,Interconnection *> InputActivity;

class BufferedState: public NeuronState {

	private:

		/*!
		 * \brief Buffer of received activity.
		 */
		InputActivity * ActivityBuffer;

		/*!
		 * \brief Time in which the activity will be stored.
		 */
		float BufferAmplitude;

		/*!
		 * \brief Maximum size of the buffer.
		 */
		unsigned int MaximumSize;

		/*!
		 * \brief Index of the first element in the list.
		 */
		unsigned int FirstIndex;

		/*!
		 * \brief Index of the last element in the list.
		 */
		unsigned int LastIndex;

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
		BufferedState(unsigned int NumVariables, float BufferAmpl, unsigned int MaxSize);

		/*!
		 * \brief It adds a new input spike into the buffer of activity.
		 *
		 * It adds a new input spike into the buffer of activity. The spike insertion
		 * must be done in ascending order by time.
		 *
		 * \param InputConnection Interconnection in which the spike was received.
		 */
		void AddActivity(Interconnection * InputConnection);

		/*!
		 * \brief It removes all the spikes happened before BufferAmplitude time.
		 *
		 * It removes all the spikes happened before BufferAmplitude time.
		 */
		void CheckActivity();

		/*!
		 * \brief Add elapsed time to spikes.
		 *
		 * It adds the elapsed time to spikes.
		 *
		 * \param ElapsedTime The time since the last update.
		 */
		void AddElapsedTime(float ElapsedTime);

		/*!
		 * \brief Class destructor.
		 *
		 * It destroys an object of this class.
		 */
		virtual ~BufferedState();

		/*!
		 * \brief It gets the number of stored spikes.
		 *
		 * It gets the number of stored spikes.
		 *
		 * \return The number of spikes stored in the current state.
		 */
		unsigned int GetNumberOfSpikes();

		/*!
		 * \brief It gets the time when a spike happened.
		 *
		 * It gets the time when a spike happened. The first spike is the first being introduced.
		 *
		 * \param Position Position of the spike.
		 * \return The time when the Position-th stored spike happened.
		 */
		double GetSpikeTimeAt(unsigned int Position);

		/*!
		 * \brief It gets the connection where a spike happened.
		 *
		 * It gets the connection where a spike happened. The first spike is the first being introduced.
		 *
		 * \param Position Position of the spike.
		 * \return The connection when the Position-th stored spike happened.
		 */
		Interconnection * GetInterconnectionAt(unsigned int Position);

};

#endif /* BUFFEREDSTATE_H_ */
