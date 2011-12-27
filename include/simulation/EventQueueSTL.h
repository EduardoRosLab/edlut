/***************************************************************************
 *                           EventQueueSTL.h                                  *
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

#ifndef EVENTQUEUESTL_H_
#define EVENTQUEUESTL_H_

/*!
 * \file EventQueueSTL.h
 *
 * \author Jesus Garrido
 * \author Richard Carrido
 * \date August 2008
 *
 * This file declares a class which abstracts an event queue by using a STL vector.
 */
 
#include <cstdlib>
#include <vector>

using namespace std;

class Event;

/*!
 * \class EventQueueSTL
 *
 * \brief Event queue
 *
 * This class abstract the behaviour of an sorted by event time queue. This implementation uses a STL vector.
 *
 * \author Jesus Garrido
 * \author Richard Carrillo
 * \date August 2008
 */
class EventQueueSTL {
	private:
	
		/*!
		 * Spikes vector.
		 */
		vector<Event *> Events;
   
   		/*!
   		 * It swaps the position of two events.
   		 */
   		void SwapEvents(unsigned int c1, unsigned int c2);
   		
   	public:
   	
   		/*!
   		 * \brief Default constructor.
   		 * 
   		 * Default constructor without parameters. It creates a new event queue.
   		 */
   		EventQueueSTL();
   		
   		/*!
   		 * \brief Object destructor.
   		 * 
   		 * Default object destructor.
   		 */
   		~EventQueueSTL();
   		
   		/*!
   		 * \brief It gets the number of events in the queue.
   		 * 
   		 * It gets the number of events in the queue.
   		 * 
   		 * \return The number of events in the queue.
   		 */
   		unsigned int Size() const;
   		
   		/*!
   		 * \brief It inserts a spike in the event queue.
   		 * 
   		 * It inserts a spike in the event queue.
   		 * 
   		 * \param event The new event to insert in the queue.
   		 */
   		void InsertEvent(Event * event);
   		
   		/*!
   		 * \brief It removes the first event in the queue.
   		 * 
   		 * It removes the first event in the queue. It returns the first event sorted by time.
   		 * 
   		 * \return The first event sorted by time.
   		 */
   		Event * RemoveEvent(void);
   		
   		/*!
   		 * \brief It returns the time of the first event.
   		 * 
   		 * It returns the time of the first event.
   		 * 
   		 * \return The time of the first event.
   		 */
   		double FirstEventTime(void) const;		
};

#endif /*EVENTQUEUESTL_H_*/
