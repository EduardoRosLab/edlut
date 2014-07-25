/***************************************************************************
 *                           TableBasedModelHFEvent.h                      *
 *                           -------------------                           *
 * copyright            : (C) 2014 by Francisco Naveros                    *
 * email                : fnaveros@ugr.es                                  *
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 3 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

#ifndef TABLEBASEDMODELHFEVENT_H_
#define TABLEBASEDMODELHFEVENT_H_

/*!
 * \file TableBasedModelHFEvent.h
 *
 * \author Francisco Naveros
 * \date April 2014
 *
 * This file declares a class which abstracts the second part of the TableBasedModelHF neuron model.
 * This object makes the prediction of this neuron model.
 */
 
#include <iostream>

#include "./InternalSpike.h"

using namespace std;

class Neuron;
class Simulation;

/*!
 * \class TableBasedModelHFEvent
 *
 * \brief Second part of the TableBasedModelHF neuron model.
 *
 * This file declares a class which abstracts the second part of the TableBasedModelHF neuron model.
 * This object makes the prediction of this neuron model.
 *
 * \author Francisco Naveros
 * \date April 2014
 */
class TableBasedModelHFEvent: public InternalSpike{
	
	public:
   		
   		/*!
   		 * \brief Default constructor.
   		 * 
   		 * It creates and initializes a new spike object.
   		 */
   		TableBasedModelHFEvent();
   	
   		/*!
   		 * \brief Constructor with parameters.
   		 * 
   		 * It creates and initializes a new spike with the parameters.
   		 * 
   		 * \param NewTime Time of the new spike.
   		 * \param NewSource Source neuron of the spike.
   		 */
   		TableBasedModelHFEvent(double NewTime, Neuron * NewSource);
   		
   		/*!
   		 * \brief Class destructor.
   		 * 
   		 * It destroies an object of this class.
   		 */
   		~TableBasedModelHFEvent();
   	

   		/*!
   		 * \brief It process an event in the simulation with the option of real time available.
   		 * 
   		 * It process an event in the simulation with the option of real time available.
   		 * 
   		 * \param CurrentSimulation The simulation object where the event is working.
		 * \param RealTimeRestriction watchdog variable executed in a parallel OpenMP thread that
		 * control the consumed time in each slot.
   		 */
   		void ProcessEvent(Simulation * CurrentSimulation, volatile int * RealTimeRestriction);

		/*!
   		 * \brief It process an event in the simulation without the option of real time available.
   		 * 
   		 * It process an event in the simulation without the option of real time available.
   		 * 
   		 * \param CurrentSimulation The simulation object where the event is working.
   		 */
		void ProcessEvent(Simulation * CurrentSimulation);
   		 
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

#endif /*TABLEBASEDMODELHFEVENT_H_*/
