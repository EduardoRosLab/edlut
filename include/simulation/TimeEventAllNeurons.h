/***************************************************************************
 *                           TimeEventAllNeurons.h                         *
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

#ifndef TIMEEVENTALLNEURONS_H_
#define TIMEEVENTALLNEURONS_H_

/*!
 * \file TimeEventAllNeurons.h
 *
 * \author Francisco Naveros
 * \date May 2013
 *
 * This file declares a class which implements the behaviour of time-driven
 * neuron model events. Each time that a time-driven step happens this class will
 * call the update methods from all cell.
 */

#include "../../include/simulation/TimeEventOneNeuron.h"

/*!
 * \class TimeEventAllNeurons
 *
 * \brief Time-driven cell model event.
 *
 * This class abstract the concept of time-driven update state. It implements the method
 * which updates the state variables from all time-driven cell.
 *
 * \author Francisco Naveros
 * \date May 2013
 */
class TimeEventAllNeurons : public TimeEventOneNeuron{

public:

	/*!
	 * \brief Constructor with parameters.
	 * 
	 * It creates and initializes a new time-driven event with the parameters.
	 * 
	 * \param NewTime Time of the next state variable update.
	 * \param indexNeuronModel index neuron model inside the network
	 */
	TimeEventAllNeurons(double NewTime, TimeDrivenNeuronModel * newNeuronModel, Neuron ** newNeurons);

	
	/*!
	 * \brief Class destructor.
	 * 
	 * It destroies an object of this class.
	 */
	~TimeEventAllNeurons();

	/*!
	 * \brief It process an event in the simulation with the option of real time available.
	 * 
	 * It process an event in the simulation with the option of real time available.
	 * 
	 * \param CurrentSimulation The simulation object where the event is working.
	 * \param RealTimeRestriction watchdog variable executed in a parallel OpenMP thread that
	 * control the consumed time in each slot.
	 */
	virtual void ProcessEvent(Simulation * CurrentSimulation, volatile int * RealTimeRestriction);

	/*!
	 * \brief It process an event in the simulation without the option of real time available.
	 * 
	 * It process an event in the simulation without the option of real time available.
	 * 
	 * \param CurrentSimulation The simulation object where the event is working.
	 */
	virtual void ProcessEvent(Simulation * CurrentSimulation);

	/*!
	 * \brief this method print the event type.
	 * 
	 * This method print the event type..
	 */
	virtual void PrintType();

};

#endif
