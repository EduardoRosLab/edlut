#ifndef PARAMETEREXCEPTION_H_
#define PARAMETEREXCEPTION_H_

/*!
 * \file ParameterException.h
 *
 * \author Jesus Garrido
 * \date September 2008
 *
 * This file declares a class which abstracts an exception in the EDLUT input parameters.
 */

#include <string>
#include <iostream>

using namespace std;

/*!
 * \class ParameterException
 *
 * \brief Exception in the readding parameter process.
 *
 * This class abstract the behaviour of an exception in the readding parameters process. This exception
 * could happen by different reasons (not found files, bad parameters...).
 *
 * \author Jesus Garrido
 * \date September 2008
 */
class ParameterException {
	
	private:
		/*!
		 * Input parameter.
		 */
		string Parameter;
		
		/*!
		 * Error message.
		 */
		string Message;
		
	public:
	
		/*!
		 * \brief Class constructor.
		 * 
		 * Class constructor with parameters.
		 * 
		 * \param ErrorParameter Parameter which throws the error.
		 * \param ErrorMessage Message describing the error.
		 */ 
		ParameterException(string ErrorParameter, string ErrorMessage);
		
		/*!
		 * \brief It gets the error parameter.
		 * 
		 * It gets the error parameter.
		 * 
		 * \return The error parameter.
		 */ 
		string GetParameter() const;
		
		/*!
		 * \brief It gets the error message.
		 * 
		 * It gets the error that threw the error.
		 * 
		 * \return The error message.
		 */
		string GetErrorMsg() const;
		
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
ostream & operator<< (ostream & out, ParameterException Exception);

#endif /*PARAMETEREXCEPTION_H_*/
