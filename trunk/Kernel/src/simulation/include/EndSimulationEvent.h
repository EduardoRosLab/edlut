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

#ifndef ENDSIMULATIONEVENT_H_
#define ENDSIMULATIONEVENT_H_

/*!
 * \file EndSimulationEvent.h
 *
 * \author Jesus Garrido
 * \date November 2008
 *
 * This file declares a class which abstracts a simulation event for the end of simulation.
 */
 
#include <iostream>

#include "./Event.h"

using namespace std;

class Simulation;

/*!
 * \class EndSimulationEvent
 *
 * \brief Simulation abstract event for the end of simulation.
 *
 * This class abstract the concept of event for the end of simulation.
 *
 * \author Jesus Garrido
 * \date November 2008
 */
class EndSimulationEvent: public Event{
	
	public:
   		
   		/*!
   		 * \brief Default constructor.
   		 * 
   		 * It creates and initializes a new event object.
   		 */
   		EndSimulationEvent();
   	
   		/*!
   		 * \brief Constructor with parameters.
   		 * 
   		 * It creates and initializes a new event with the parameters.
   		 * 
   		 * \param NewTime Time of the new event.
   		 */
   		EndSimulationEvent(double NewTime);
   		
   		/*!
   		 * \brief Class destructor.
   		 * 
   		 * It destroies an object of this class.
   		 */
   		~EndSimulationEvent();
   	
   		/*!
   		 * \brief It process an event in the simulation.
   		 * 
   		 * It process the event in the simulation.
   		 * 
   		 * \param CurrentSimulation The simulation object where the event is working.
   		 */
   		virtual void ProcessEvent(Simulation * CurrentSimulation);
};

#endif /*ENDSIMULATIONEVENT_H_*/
