/***************************************************************************
 *                           TimeEvent.h                                   *
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

#ifndef TIMEEVENT_H_
#define TIMEEVENT_H_

/*!
 * \file TimeEvent.h
 *
 * \author Jesus Garrido
 * \date January 2011
 *
 * This file declares a class which implements the behaviour of time-driven
 * neuron model events. Each time that a time-driven step happens this class will
 * call the update methods from each cell.
 */

#include "../simulation/Event.h"

/*!
 * \class TimeEvent
 *
 * \brief Time-driven cell model event.
 *
 * This class abstract the concept of time-driven update state. It implements the method
 * which updates the state variables from each time-driven cell.
 *
 * \author Jesus Garrido
 * \date January 2011
 */
class TimeEvent : public Event{

public:

	/*!
	 * \brief Constructor with parameters.
	 * 
	 * It creates and initializes a new time-driven event with the parameters.
	 * 
	 * \param NewTime Time of the next state variable update.
	 */
	TimeEvent(double NewTime);
	
	/*!
	 * \brief Class destructor.
	 * 
	 * It destroies an object of this class.
	 */
	~TimeEvent();

	/*!
	 * \brief It updates the state of every time-driven cell in the network.
	 * 
	 * It updates the state of every time-driven cell in the network.
	 * 
	 * \param CurrentSimulation The simulation object where the event is working.
	 */
	void ProcessEvent(Simulation * CurrentSimulation);
};

#endif
