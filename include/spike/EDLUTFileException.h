/***************************************************************************
 *                           EDLUTFileException.h                          *
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

#ifndef EDLUTFILEEXCEPTION_H_
#define EDLUTFILEEXCEPTION_H_

/*!
 * \file EDLUTFileException.h
 *
 * \author Jesus Garrido
 * \author Richard Carrido
 * \date August 2008
 *
 * This file declares a class which abstracts an exception in reading/writting files in the EDLUT simulation process.
 */

#include <cstdlib>
#include <cstdio>
#include <iostream>

#include "./EDLUTException.h"

using namespace std;

/*!
 * \class EDLUTFileException
 *
 * \brief Exception reading/writting files.
 *
 * This class abstract the behaviour of an exception in the EDLUT simulation process. This exception
 * could happen by reading/writting files. When
 * an exception is thrown, the simulation is stopped without results.
 *
 * \author Jesus Garrido
 * \author Richard Carrillo
 * \date August 2008
 */
class EDLUTFileException: public EDLUTException {
	
	private:
		
		/*!
		 * File line where the error happens.
		 */
		long Currentline;
		
	public:
	
		/*!
		 * \brief Class constructor.
		 * 
		 * Class constructor with parameters.
		 * 
		 * \param a The most significant integer. Task number.
		 * \param b The second most significant integer. Error number.
		 * \param c The third most significant integer. Repair number.
		 * \param d The less significant integer,
		 * \param Line Line where the error happened.
		 */ 
		EDLUTFileException(int a, int b, int c, int d, long Line);
		
		/*!
		 * \brief It gets the line where the error happened.
		 * 
		 * It gets the line where the error happened.
		 * 
		 * \return The error line.
		 */
		long GetErrorLine() const;
		
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
ostream & operator<< (ostream & out, EDLUTFileException Exception);

#endif /*EDLUTFILEEXCEPTION_H_*/
