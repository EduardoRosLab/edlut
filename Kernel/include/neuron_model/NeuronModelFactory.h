/***************************************************************************
 *                           NeuronModelFactory.h                          *
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

#ifndef EDLUT_NEURONMODELFACTORY_H
#define EDLUT_NEURONMODELFACTORY_H

#include <string>
#include <map>
#include "boost/any.hpp"


#include "simulation/NetworkDescription.h"

class NeuronModel;
class EDLUTException;

/*!
 * \file NeuronModelFactory.h
 *
 * \author Jesus Garrido
 * \date June 2018
 *
 * This file declares a factory for NeuronModel object creation.
 */

/*!
* \class NeuronModelFactory
*
* \brief Neuron model factory.
*
* This class declares the methods required for creating NeuronModel objects.
*
* \author Jesus Garrido
* \date June 2018
*/

typedef ModelDescription (*ParseNeuronModelFunc) (std::string);
typedef NeuronModel * (*CreateNeuronModelFunc) (ModelDescription);
typedef std::map<std::string,boost::any> (*GetDefaultParamFunc) (void);
typedef std::map<std::string,std::string> (*GetNeuronModelInfoFunc) (void);
typedef std::map<std::string, std::string> (*GetVectorizableParametersFunc) (void);

class NeuronModelFactory {

    private:

		/*!
		 * Create a class to store the pointers to the functions in the neuron models
		 */
		class NeuronModelClass {
			public:

			ParseNeuronModelFunc parseFunc;

			CreateNeuronModelFunc createFunc;

			GetDefaultParamFunc  getDefaultParamFunc;

			GetNeuronModelInfoFunc getInfoFunc;

			GetVectorizableParametersFunc getVectorizableParamFunc;

			
			/*!
			* \brief Default constructor without parameters.
			*
			* It generates a new neuron model class object without assigned functions.
			*/
			NeuronModelClass():
				parseFunc(0), createFunc(0), getDefaultParamFunc(0), getInfoFunc(0), getVectorizableParamFunc(0){};

			
			/*!
			* \brief Constructor with 4 parameters.
			*
			* It generates a new neuron model object with four assigned functions (the fifth function is automatically assigned
			* in non vectorized neuron models ).
			*
			* \param parseFunc Neuron model function used to load the neurno model parameters from a text file.
			* \param createFunc Neuron model function used to create a neuron model using a ModelDescription object.
			* \param getDefaultParamFunc Neuron model function used to get a dictionary with the default parameters of the model.
			* \param getInfoFunc Neuron model function used to get a dictionary with information about all the neuron model parameters.
			*/
			NeuronModelClass(ParseNeuronModelFunc parseFunc, CreateNeuronModelFunc createFunc, GetDefaultParamFunc getDefaultParamFunc,
				GetNeuronModelInfoFunc getInfoFunc) : parseFunc(parseFunc), createFunc(createFunc), getDefaultParamFunc(getDefaultParamFunc), 
				getInfoFunc(getInfoFunc)/*, getVectorizableParamFunc(NeuronModel::GetVectorizableParameters)*/{
			
				getVectorizableParamFunc = GetVectorizableParameters;
			};

			static std::map<std::string, std::string> GetVectorizableParameters(){
				std::map<std::string, std::string> vectorizableParameters;
				return vectorizableParameters;
			}

			/*!
			* \brief Constructor with 5 parameters.
			*
			* It generates a new neuron model object with five assigned functions.
			*
			* \param parseFunc Neuron model function used to load the neurno model parameters from a text file.
			* \param createFunc Neuron model function used to create a neuron model using a ModelDescription object.
			* \param getDefaultParamFunc Neuron model function used to get a dictionary with the default parameters of the model.
			* \param getInfoFunc Neuron model function used to get a dictionary with information about all the neuron model parameters.
			* \param getVectorizableParamFunc Neuron model function used to get a dictionary with the neuron model parameters that has
			* been vectorized (can take different values for each neuron inside the neuron model).
			*/
			NeuronModelClass(ParseNeuronModelFunc parseFunc, CreateNeuronModelFunc createFunc, GetDefaultParamFunc getDefaultParamFunc,
				GetNeuronModelInfoFunc getInfoFunc, GetVectorizableParametersFunc getVectorizableParamFunc) : parseFunc(parseFunc),
				createFunc(createFunc), getDefaultParamFunc(getDefaultParamFunc), getInfoFunc(getInfoFunc), getVectorizableParamFunc(getVectorizableParamFunc){};

		
		};

		/*!
		 * Dictionary of the existing LearningRule classes and the Parsing and Creation functions
		 */
		static std::map<std::string, NeuronModelFactory::NeuronModelClass > NeuronModelMap;

    public:

		/*!
		* \brief Get all the available Neuron Models in a vector.
		*
		* It gets all the available Neuron Models in a vector.
		*/
		static std::vector<std::string> GetAvailableNeuronModels();

		/*!
		* \brief Printing all the available Neuron Models.
		*
		* It prints all the available Neuron Models.
		*/
		static void PrintAvailableNeuronModels();


		/*!
		 * \brief Filling of the factory.
		 *
		 * It creates a dictionary associating the name of each neuron model name with the pointer to their static functions.
		 */
		 static void InitializeNeuronModelFactory();

		/*!
		 * \brief It creates a new neuron model object of the corresponding type.
		 *
		 * It creates a new neuron model object of the corresponding type.
		 *
		 * \param nmodelDescription Description of the neuron model parameters.
		 *
		 * \throw EDLUTException If the neuron model does not exist with that name.
		 */
		 static NeuronModel * CreateNeuronModel(ModelDescription nmodelDescription) noexcept(false);

		/*!
		* \brief It parses the neuron model description from the file.
		*
		* It extracts the parameters of the neuron model from the file.
		*
		* \param ident Name of the neuron model to create.
		* \param ident_param File name where the parameters can be readed.
		*
		* \throw EDLUTException If the learning rule does not exist with that name.
		*/
		 static ModelDescription ParseNeuronModel(std::string ident, std::string ident_param) noexcept(false);

		/*!
		 * \brief It returns the default parameters for the selected neuron model.
		 *
		 * It returns the default parameters for the selected neuron model.
		 *
		 * \param ident Name of the neuron model to create.
		 *
		 * \return A map with the default parameters of the neuron model.
		 *
		 * \throw EDLUTException If the neuron model does not exist with that name.
		 */
		 static std::map<std::string, boost::any> GetDefaultParameters(std::string ident) noexcept(false);

		/*!
		* \brief It returns information about the neuron model, including its parameters.
		*
		* It returns information about the neuron model, including its parameters.
		*
		* \param ident Name of the neuron model.
		*
		* \return A map with the neuron model information, including its parameters.
		*
		* \throw EDLUTException If the neuron model does not exist with that name.
		*/
		 static std::map<std::string, std::string> GetNeuronModelInfo(std::string ident) noexcept(false);

		/*!
		* \brief It prints information about the neuron model, including its parameters.
		*
		* It prints information about the neuron model, including its parameters.
		*
		* \param ident Name of the neuron model.
		*
		* \throw EDLUTException If the neuron model does not exist with that name.
		*/
		 static void PrintNeuronModelInfo(std::string ident) noexcept(false);

		/*!
		* \brief It returns information about the neuron model parameters that can be vectorized.
		*
		* It returns information about the neuron model parameters that can be vectorized.
		*
		* \param ident Name of the neuron model.
		*
		* \return A map with the neuron model parameters that can be vectorized and their description.
		*
		* \throw EDLUTException If the neuron model does not exist with that name.
		*/
		 static std::map<std::string, std::string> GetVectorizableParameters(std::string ident) noexcept(false);

		/*!
		* \brief It prints information about the neuron model parameters that can be vectorized.
		*
		* It prints information about the neuron model parameters that can be vectorized.
		*
		* \param ident Name of the neuron model.
		*
		* \throw EDLUTException If the neuron model does not exist with that name.
		*/
		 static void PrintVectorizableParameters(std::string ident) noexcept(false);
};

#endif //EDLUT_NEURONMODELFACTORY_H
