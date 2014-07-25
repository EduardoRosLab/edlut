/***************************************************************************
 *                           TimeEventOneNeuron.h                          *
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

#ifndef TIMEEVENTONENEURON_H_
#define TIMEEVENTONENEURON_H_

/*!
 * \file TimeEventOneNeuron.h
 *
 * \author Francisco Naveros
 * \date May 2013
 *
 * This file declares a class which implements the behaviour of time-driven
 * neuron model events. Each time that a time-driven step happens this class will
 * call the update methods from a cell.
 */

#include "../simulation/Event.h"


class TimeDrivenNeuronModel;
class Neuron;

/*!
 * \class TimeEventOneNeuron
 *
 * \brief Time-driven cell model event.
 *
 * This class abstract the concept of time-driven update state. It implements the method
 * which updates the state variables from a time-driven cell.
 *
 * \author Francisco Naveros
 * \date May 2013
 */
class TimeEventOneNeuron : public Event{

public:
	
	/*!
	 * \brief Neuron model.
	*/
	TimeDrivenNeuronModel * neuronModel;

	/*!
	 * \brief Neuron model.
	*/
	Neuron** neurons;

	/*!
	 * \brief Index neuron (if IndexNeuron=-1, it represents all neurons in neuron model).
	*/
	int IndexNeuron;

	/*!
	 * \brief Constructor with parameters.
	 * 
	 * It creates and initializes a new time-driven event with the parameters.
	 * 
	 * \param NewTime Time of the next state variable update.
	 * \param indexNeuronModel index neuron model inside the network.
	 * \param indexNeuron index neuron inside the neuron model.
	 */
	TimeEventOneNeuron(double NewTime, TimeDrivenNeuronModel * newNeuronModel, Neuron ** newNeurons, int indexNeuron);
	
	/*!
	 * \brief Class destructor.
	 * 
	 * It destroies an object of this class.
	 */
	~TimeEventOneNeuron();


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
	 * \brief It gets the neuron model.
	 *
	 * It gets the neuron model.
	 *
	 * \return The neuron model.
	 */
	TimeDrivenNeuronModel * GetModel();
	

	/*!
	 * \brief It gets neuron list that use this neuron model.
	 *
	 * It gets neuron list that use this neuron model.
	 *
	 * \return The neuron list that use this neuron model.
	 */
	Neuron ** GetNeurons();


	/*!
	 * \brief It gets the index neuron inside the neuron model.
	 *
	 * It gets the index neuron inside the neuron model.
	 *
	 * \return The index neuron inside the neuron model.
	 */
	int GetIndexNeuron();

	/*!
	 * \brief this method print the event type.
	 * 
	 * This method print the event type..
	 */
	virtual void PrintType();

	/*!
	 * \brief The event queue uses this preference variable to sort the events with the same time stamp.
	 * 
	 * The event queue uses this preference variable to sort the events with the same time stamp.
	 */
	virtual int ProcessingPriority();
};

#endif
