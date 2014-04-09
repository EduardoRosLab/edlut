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

/*!
 * Define a node of the activity list
 */
struct Activity{
	/*!
	 * The input activity
	 */
	InputActivity Spike;

	/*!
	 * The next node in the list
	 */
	struct Activity * NextNode;
};

typedef struct Activity ActivityNode;

class BufferedState: public NeuronState {

	private:

		/*!
		 * \brief First node of the activity list (the oldest one).
		 */
		ActivityNode ** FirstElement;

		/*!
		 * \brief Last node of the activity list (the youngest one).
		 */
		ActivityNode ** LastElement;

		/*!
		 * \brief Time in which the activity will be removed.
		 */
		float * BufferAmplitude;

		/*!
		 * \brief Number of elements inside.
		 */
		unsigned int * NumberOfElements;

		/*!
		 * \brief Number of buffers included.
		 */
		unsigned int NumberOfBuffers;


	public:
		/*!
		 * \brief Default constructor with parameters.
		 *
		 * It generates a new state of a cell. The temporal amplitude of each channel must be set by using
		 * SetBufferAmplitude.
		 *
		 * \param NumVariables Number of the state variables this model needs.
		 * \param NumBuffers Number of input channels
		 */
		BufferedState(unsigned int NumVariables, unsigned int NumBuffers);

		/*!
		 * \brief Copies constructor.
		 *
		 * It generates a new objects which copies the parameter.
		 *
		 * \param OldState State being copied.
		 */
		BufferedState(const BufferedState & OldState);

		/*!
		 * \brief It sets the amplitude of the selected buffer
		 *
		 * It sets the amplitude of the selected buffer.
		 *
		 * \param NumBuffer Number of the buffer to be set.
		 * \param BufferAmpl Temporal amplitude of the buffer.
		 */
		void SetBufferAmplitude(unsigned int NumBuffer, float BufferAmpl);

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
		virtual void AddElapsedTime(float ElapsedTime);

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
		 * \param NumBuffer Number of the buffer to get the number of stored spikes.
		 *
		 * \return The number of spikes stored in the current state.
		 */
		unsigned int GetNumberOfSpikes(unsigned int NumBuffer);

		/*!
		 * \brief It gets the time when a spike happened.
		 *
		 * It gets the time when a spike happened. The first spike is the first being introduced.
		 *
		 * \param Position Position of the spike.
		 * \param NumBuffer Number of the buffer from which the spike will be retrieved.
		 *
		 * \return The time when the Position-th stored spike happened.
		 *
		 * \note This function should be avoided in favour of iterators due to efficiency issues.
		 */
		double GetSpikeTimeAt(unsigned int Position, unsigned int NumBuffer);

		/*!
		 * \brief It gets the connection where a spike happened.
		 *
		 * It gets the connection where a spike happened. The first spike is the first being introduced.
		 *
		 * \param Position Position of the spike.
		 * \param NumBuffer Number of the buffer from which the spike will be retrieved.
		 *
		 * \return The connection when the Position-th stored spike happened.
		 *
		 * \note This function should be avoided in favour of iterators due to efficiency issues.
		 */
		Interconnection * GetInterconnectionAt(unsigned int Position, unsigned int NumBuffer);

		class Iterator {
			private:
				// Declaring buffer iterator
				ActivityNode * element;

			public:
				/*!
				 * \brief Default class constructor
				 *
				 * Default class constructor. It creates a new null-pointer iterator.
				 */
				Iterator();

				/*!
				 * \brief Copy class constructor
				 *
				 * Copy class constructor. It creates a new pointer pointing to the parameter.
				 *
				 * \param ElemAux Element to be copied.
				 */
				Iterator(const Iterator & ItAux);

				/*!
				 * \brief Copy class constructor
				 *
				 * Copy class constructor. It creates a new pointer pointing to the parameter.
				 *
				 * \param ElemAux Element to be copied.
				 */
				Iterator(ActivityNode * ElemAux);

				/*!
				 * \brief It gets the next element stored in the activity buffer.
				 *
				 * It gets the next element stored in the activity buffer.
				 *
				 * \return An iterator pointing to the next element in the buffer.
				 */
				Iterator & operator++();

				/*!
				 * \brief It compares if two iterators point the same element.
				 *
				 * It compares if two iterators point the same element.
				 *
				 * \return True if the two iterators point the same element. False otherwise.
				 */
				bool operator==(BufferedState::Iterator Aux);

				/*!
				 * \brief It compares if two iterators point different elements.
				 *
				 * It compares if two iterators point different elements.
				 *
				 * \return True if the two iterators point different elements. False otherwise.
				 */
				bool operator!=(BufferedState::Iterator Aux);

				/*! It gets the spike time of the current element pointed by the iterator.
				 *
				 * It gets the spike time of the current element pointed by the iterator.
				 *
				 * \return The spike time of the current element pointed by the iterator.
				 */
				double GetSpikeTime();

				/*! It gets the connection of the current element pointed by the iterator.
				 *
				 * It gets the connection of the current element pointed by the iterator.
				 *
				 * \return The connection of the current element pointed by the iterator.
				 */
				Interconnection * GetConnection();

		};

		/*!
		 * \brief It gets the first element stored in the activity buffer.
		 *
		 * It gets the first element stored in the activity buffer.
		 *
		 * \param NumBuffer Number of the buffer to iterate.
		 *
		 * \return An iterator pointing to the first element in the buffer.
		 */
		Iterator Begin(unsigned int NumBuffer);

		/*!
		 * \brief It gets the after-last element stored in the activity buffer.
		 *
		 * It gets the after-last element stored in the activity buffer.
		 *
		 * \return An iterator pointing to the after-last element in the buffer.
		 */
		Iterator End();


};


#endif /* BUFFEREDSTATE_H_ */
