/***************************************************************************
 *                           SaveWeightsEvent.h                            *
 *                           -------------------                           *
 * copyright            : (C) 2009 by Jesus Garrido and Richard Carrillo   *
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

#ifndef SAVEWEIGHTSEVENT_H_
#define SAVEWEIGHTSEVENT_H_

/*!
 * \file SaveWeightsEvent.h
 *
 * \author Jesus Garrido
 * \date November 2008
 *
 * This file declares a class which abstracts a simulation event for the save weights time.
 */
 
#include <iostream>

#include "./Event.h"

using namespace std;

class Simulation;

/*!
 * \class SaveWeightsEvent
 *
 * \brief Simulation abstract event for the save weights time.
 *
 * This class abstract the concept of event for the save weights time.
 *
 * \author Jesus Garrido
 * \date November 2008
 */
class SaveWeightsEvent: public Event{
	
	public:
   		
   		/*!
   		 * \brief Default constructor.
   		 * 
   		 * It creates and initializes a new event object.
   		 */
   		SaveWeightsEvent();
   	
   		/*!
   		 * \brief Constructor with parameters.
   		 * 
   		 * It creates and initializes a new event with the parameters.
   		 * 
   		 * \param NewTime Time of the new event.
   		 */
   		SaveWeightsEvent(double NewTime);
   		
   		/*!
   		 * \brief Class destructor.
   		 * 
   		 * It destroies an object of this class.
   		 */
   		~SaveWeightsEvent();
   	

   		/*!
   		 * \brief It process an event in the simulation.
   		 * 
   		 * It process the event in the simulation.
   		 * 
   		 * \param CurrentSimulation The simulation object where the event is working.
		 * \param RealTimeRestriction This variable indicates whether we are making a 
		 * real-time simulation and the watchdog is enabled.
   		 */
   		virtual void ProcessEvent(Simulation * CurrentSimulation, bool RealTimeRestriction);
};

#endif /*SAVEWEIGHTSEVENT_H_*/
