/***************************************************************************
 *                           VectorSRMState.h                              *
 *                           -------------------                           *
 * copyright            : (C) 2012 by Jesus Garrido and Francisco Naveros  *
 * email                : jgarrido@atc.ugr.es, fnaveros@atc.ugr.es         *
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 3 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

#ifndef VECTORSRMSTATE_H_
#define VECTORSRMSTATE_H_

/*!
 * \file VectorSRMState.h
 *
 * \author Jesus Garrido
 * \author Francisco Naveros
 * \date February 2012
 *
 * This file declares a class which implements the state of a cell vector which
 * stores the last activity happened.
 *
 * \note: This class is a modification of previous SRMState class. In this new class,
 * it is generated a only object for a neuron model cell vector instead of a object for
 * each cell.
 */

#include "VectorBufferedState.h"

/*!
 * \class VectorSRMState
 *
 * \brief Spiking response model based on activity buffer.
 *
 * This class abstracts the state of a cell in a SRM Model.
 *
 * \author Jesus Garrido
 * \author Francisco Naveros
 * \date February 2012
 */
class VectorSRMState: public VectorBufferedState {

	public:
		/*!
		 * \brief Default constructor with parameters.
		 *
		 * It generates a new state of a cell.
		 *
		 * \param NumVariables Number of the state variables this model needs.
		 * \param NumBuffers Number of buffers this model needs.
		 * \param isTimeDriven It is for a time-driven or a event-driven method.
		 */
		VectorSRMState(unsigned int NumVariables, unsigned int NumBuffers, bool isTimeDriven);

		/*!
		 * \brief Copies constructor.
		 *
		 * It generates a new objects which copies the parameter.
		 *
		 * \param OldState State being copied.
		 */
		VectorSRMState(const VectorSRMState & OldState);

		/*!
		 * \brief Class destructor.
		 *
		 * It destroys an object of this class.
		 */
		virtual ~VectorSRMState();

		/*!
		 * \brief It gets the number of variables that you can print in this state.
		 *
		 * It gets the number of variables that you can print in this state.
		 *
		 * \return The number of variables that you can print in this state.
		 */
		virtual unsigned int GetNumberOfPrintableValues();

		/*!
		 * \brief It gets a value to be printed from this state for a cell.
		 *
		 * It gets a value to be printed from this state for a cell.
		 *
		 * \param index The cell index inside the vector.
		 * \param position inside a neuron state.
		 *
		 * \return The value at position-th position in this state for a cell.
		 */
		virtual double GetPrintableValuesAt(int index, int position);

		/*!
		 * \brief It initialice all vectors with size size and copy initialization inside VectorNeuronStates
		 * for each cell.
		 *
		 * It initialice all vectors with size size and copy initialization inside VectorNeuronStates
		 * for each cell.
		 *
		 * \param size cell number inside the VectorNeuronState.
		 * \param initialization initial state for each cell.
		 */
		void InitializeSRMStates(int size, float * initialization);
};

#endif /* VECTORSRMSTATE_H_ */
