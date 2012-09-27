/***************************************************************************
 *                           TimeEvent_GPU.h                                *
 *                           -------------------                           *
 * copyright            : (C) 2011 by Francisco Naveros                    *
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

#ifndef TIMEEVENT_GPU_H_
#define TIMEEVENT_GPU_H_

/*!
 * \file TimeEvent_GPU.h
 *
 * \author Francisco Naveros
 * \date January 2012
 *
 * This file declares a class which implements the behaviour of time-driven
 * neuron model events in a GPU. Each time that a time-driven step happens this class will
 * call the update methods from each cell.
 */

#include "../simulation/Event.h"

/*!
 * \class TimeEvent_GPU
 *
 * \brief Time-driven cell model event.
 *
 * This class abstract the concept of time-driven update state in a GPU. It implements the method
 * which updates the state variables from each time-driven cell in a GPU.
 *
 * \author Francisco Naveros
 * \date January 2012
 */
class TimeEvent_GPU : public Event{

public:

	/*!
	 * \brief Constructor with parameters.
	 * 
	 * It creates and initializes a new time-driven event with the parameters.
	 * 
	 * \param NewTime Time of the next state variable update.
	 */
	TimeEvent_GPU(double NewTime);
	
	/*!
	 * \brief Class destructor.
	 * 
	 * It destroies an object of this class.
	 */
	~TimeEvent_GPU();

	/*!
	 * \brief It updates the state of every time-driven cell in GPU in the network.
	 * 
	 * It updates the state of every time-driven cell in GPU in the network.
	 * 
	 * \param CurrentSimulation The simulation object where the event is working.
	 */
	void ProcessEvent(Simulation * CurrentSimulation);
};

#endif
