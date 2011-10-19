/***************************************************************************
 *                           ActivityRegister.h                            *
 *                           -------------------                           *
 * copyright            : (C) 2009 by Jesus Garrido                        *
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

#ifndef ACTIVITYREGISTER_H_
#define ACTIVITYREGISTER_H_

/*!
 * \file ActivityRegister.h
 *
 * \author Jesus Garrido
 * \date July 2009
 *
 * This file declares a class which abstracts a register of activity in an interconnection.
 */
 

/*!
 * \class ActivityRegister
 *
 * \brief Activity Register for a synapsis.
 *
 * This class abstract the behaviour of an activity register with the number of variables you
 * want
 * 
 * \author Jesus Garrido
 * \date July 2009
 */
class ActivityRegister {
	
	private:
		/*!
		 * \brief The state variables
		 */
		float * values;
		
		/*!
		 * \brief Number of state variables
		 */
		int numvar;
						
	public:
	
		/*!
		 * \brief Constructor with parameters.
		 * 
		 * It creates a new activity register object with the number of state variables you pass.
		 * 
		 * \param VarNumber The number of variables you need.
		 */
		ActivityRegister(int VarNumber);
		
		/*!
		 * \brief It gets the number of state variables.
		 * 
		 * It gets the number of state variables.
		 * 
		 * \return The number of state variables.
		 */
		int GetVarNumber() const;
		
		/*!
		 * \brief It gets the value of a state variable.
		 * 
		 * It gets the value of a state variable.
		 * 
		 * \param index The index of the state variable.
		 * \return The value of that state variable.
		 */
		float GetVarValueAt(unsigned int index) const;
		
		/*!
		 * \brief It sets the value of a state variable.
		 * 
		 * It sets the value of a state variable.
		 * 
		 * \param index The index of the state variable.
		 * \param value The new value of the state variable.
		 */
		void SetVarValueAt(unsigned int index, float value);
		
		
		/*!
		 * \brief Destructor.
		 * 
		 * It destroy the activity register.
		 */
		~ActivityRegister();
};
  
#endif /*ACTIVITYREGISTER_H_*/
