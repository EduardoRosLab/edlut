/***************************************************************************
 *                           IntegrationMethodFactory_GPU_C_INTERFACE.cuh  *
 *                           -------------------                           *
 * copyright            : (C) 2018 by Jesus Garrido                        *
 * email                : jesusgarrido@ugr.es                              *
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 3 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

#ifndef INTEGRATIONMETHODFACTORY_GPU_C_INTERFACE_H
#define INTEGRATIONMETHODFACTORY_GPU_C_INTERFACE_H

#include <string>
#include <map>
#include "boost/any.hpp"


//#include "integration_method/Euler_GPU_C_INTERFACE.cuh"

struct ModelDescription;

class IntegrationMethod_GPU_C_INTERFACE;


/*!
 * \file IntegrationMethodFactory_GPU_C_INTERFACE.cuh
 *
 * \author Jesus Garrido
 * \date June 2018
 *
 * This file declares a factory for IntegrationMethod object creation.
 */

/*!
* \class IntegrationMethodFactory
*
* \brief Integration method factory.
*
* This class declares the methods required for creating IntegrationMethod objects.
*
* \author Jesus Garrido
* \date June 2018
*/
template <class Neuron_Model>
class IntegrationMethodFactory_GPU_C_INTERFACE {

    private:
		typedef ModelDescription(*ParseIntegrationMethodFunc) (FILE *, long &);
		typedef IntegrationMethod_GPU_C_INTERFACE * (*CreateIntegrationMethodFunc_GPU) (ModelDescription, Neuron_Model*);
		typedef std::map<std::string, boost::any>(*GetDefaultParamFunc) (void);



	/*!
	* Create a class to store the pointers to the functions in the integration method for GPU
	*/
	class IntegrationMethodClass_GPU {
	public:

		ParseIntegrationMethodFunc parseFunc;

		CreateIntegrationMethodFunc_GPU createFunc;

		GetDefaultParamFunc  getDefaultParamFunc;

		IntegrationMethodClass_GPU() :
			parseFunc(0), createFunc(0), getDefaultParamFunc(0){};

		IntegrationMethodClass_GPU(ParseIntegrationMethodFunc parseFunc, CreateIntegrationMethodFunc_GPU createFunc, GetDefaultParamFunc getDefaultParamFunc) :
			parseFunc(parseFunc), createFunc(createFunc), getDefaultParamFunc(getDefaultParamFunc){};
	};


	/*!
	* Dictionary of the existing IntegrationMethods classes and the Parsing and Creation functions
	*/
	static std::map<std::string, IntegrationMethodClass_GPU > IntegrationMethodMap_GPU;

  public:


	 /*!
	 * \brief Filling of the factory.
	 *
	 * It creates a dictionary associating the name of each integration method with the pointer to their static functions in GPU.
	 */
	 static void InitializeIntegrationMethodFactory_GPU_C_INTERFACE();

	 /*!
	 * \brief It creates a new integration method object of the corresponding type.
	 *
	 * It creates a new integration method object of the corresponding type.
	 *
	 * \param lmethodDescription Description of the integration method parameters.
	 * \param nmodel Time-driven neuron model using this integration method.
	 *
	 * \throw EDLUTException If the integration method does not exist with that name.
	 */
	 static IntegrationMethod_GPU_C_INTERFACE * CreateIntegrationMethod_GPU(ModelDescription imethodDescription, Neuron_Model * nmodel);

	/*!
	* \brief It parses the integration method description from the file.
	*
	* It extracts the parameters of the integration method from the file.
	*
	* \param ident Name of the integration method to create.
	* \param fh File handler where the parameters can be readed.
	*
	* \throw EDLUTException If the integration method does not exist with that name.
	*/
	static ModelDescription ParseIntegrationMethod_GPU(std::string ident, FILE * fh, long & Currentline);


	/*!
	* \brief It returns the default parameters for the selected integration method.
	*
	* It returns the default parameters for the selected integration method.
	*
	* \param ident Name of the integration method to create.
	*
	* \return A map with the default parameters of the integration method.
	*
	* \throw EDLUTException If the integration method does not exist with that name.
	*/
	static std::map<std::string, boost::any> GetDefaultParameters_GPU(std::string ident);

};

#endif //INTEGRATIONMETHODFACTORY_GPU_C_INTERFACE_H
