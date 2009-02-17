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
   		 */
   		virtual void ProcessEvent(Simulation * CurrentSimulation);
};

#endif /*SAVEWEIGHTSEVENT_H_*/
