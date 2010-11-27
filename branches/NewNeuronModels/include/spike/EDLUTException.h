/***************************************************************************
 *                           EDLUTException.h                              *
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

#ifndef EDLUTEXCEPTION_H_
#define EDLUTEXCEPTION_H_

/*!
 * \file EDLUTException.h
 *
 * \author Jesus Garrido
 * \author Richard Carrido
 * \date August 2008
 *
 * This file declares a class which abstracts an exception in the EDLUT simulation process.
 */

#include <cstdlib>
#include <cstdio>
#include <iostream>

using namespace std;

/*!
 * \class EDLUTException
 *
 * \brief Exception in the simulation process.
 *
 * This class abstract the behaviour of an exception in the EDLUT simulation process. This exception
 * could happen by different reasons (not found files, bad input spikes, not enough memory...). When
 * an exception is thrown, the simulation is stopped without results.
 *
 * \author Jesus Garrido
 * \author Richard Carrillo
 * \date August 2008
 */
class EDLUTException {
	
	private:
		/*!
		 * Number of error.
		 */
		int ErrorNum;
		
		/*!
		 * \brief It gets an only long number from four integer values.
		 * 
		 * It gets an only long number from four integer values.
		 * 
		 * \param a The most significant integer. Task number.
		 * \param b The second most significant integer. Error number.
		 * \param c The third most significant integer. Repair number.
		 * \param d The less significant integer.
		 * 
		 * \return The error value from the four integer values.
		 */
		long GetErrorValue(int a, int b, int c, int d);
		
		/*!
		 * Task messages.
		 */
		static const char * Taskmsgs[];

		/*!
		 * Error messages.
		 */
		static const char *Errormsgs[];

		/*!
		 * Repair messages.
		 */
		static const char *Repairmsgs[];
		
		
	public:
	
		/*!
		 * \brief Class constructor.
		 * 
		 * Class constructor with parameters.
		 * 
		 * \param a The most significant integer. Task number.
		 * \param b The second most significant integer. Error number.
		 * \param c The third most significant integer. Repair number.
		 * \param d The less significant integer.
		 */ 
		EDLUTException(int a, int b, int c, int d);
		
		/*!
		 * \brief It gets the error number.
		 * 
		 * It gets the error number.
		 * 
		 * \return The error number.
		 */ 
		long GetErrorNum() const;
		
		/*!
		 * \brief It gets the task message.
		 * 
		 * It gets the task that threw the error.
		 * 
		 * \return The task message.
		 */
		const char * GetTaskMsg() const;
		
		/*!
		 * \brief It gets the error message.
		 * 
		 * It gets the error message.
		 * 
		 * \return The error message.
		 */
		const char * GetErrorMsg() const;
		
		/*!
		 * \brief It gets the repair message.
		 * 
		 * It gets the repair message.
		 * 
		 * \return The repair message.
		 */
		const char * GetRepairMsg() const;
		
		/*!
		 * \brief It prints in the standard error the error message of this exception.
		 * 
		 * It prints in the standard error the error message of this exception.
		 */
		void display_error() const;
		
};

/*!
 * \brief It prints the error message of this exception.
 * 
 * It prints the error message of this exception.
 * 
 * \param out The output stream where the message is printed.
 * \param Exception The exception to be printed.
 * 
 * \return The output stream.
 */
ostream & operator<< (ostream & out, EDLUTException Exception);

#endif /*EDLUTEXCEPTION_H_*/
