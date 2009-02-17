/***************************************************************************
 *                           EDLUTKernel.cpp  -  description               *
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

#ifndef COMMUNICATIONEVENT_H_
#define COMMUNICATIONEVENT_H_

/*!
 * \file CommunicationEvent.h
 *
 * \author Jesus Garrido
 * \date November 2008
 *
 * This file declares a class which abstracts a simulation event for the communication time.
 */
 
#include <iostream>

#include "./Event.h"

using namespace std;

class Simulation;

/*!
 * \class CommunicationEvent
 *
 * \brief Simulation abstract event for the communication time.
 *
 * This class abstract the concept of event for the communication time.
 *
 * \author Jesus Garrido
 * \date November 2008
 */
class CommunicationEvent: public Event{
	
	public:
   		
   		/*!
   		 * \brief Default constructor.
   		 * 
   		 * It creates and initializes a new event object.
   		 */
   		CommunicationEvent();
   	
   		/*!
   		 * \brief Constructor with parameters.
   		 * 
   		 * It creates and initializes a new event with the parameters.
   		 * 
   		 * \param NewTime Time of the new event.
   		 */
   		CommunicationEvent(double NewTime);
   		
   		/*!
   		 * \brief Class destructor.
   		 * 
   		 * It destroies an object of this class.
   		 */
   		~CommunicationEvent();
   	
   		/*!
   		 * \brief It process an event in the simulation.
   		 * 
   		 * It process the event in the simulation.
   		 * 
   		 * \param CurrentSimulation The simulation object where the event is working.
   		 */
   		virtual void ProcessEvent(Simulation * CurrentSimulation);
};

#endif /*COMMUNICATIONEVENT_H_*/
