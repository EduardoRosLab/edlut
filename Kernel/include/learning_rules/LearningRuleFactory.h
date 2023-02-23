/***************************************************************************
 *                           LearningRuleFactory.h                         *
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

#ifndef EDLUT_LEARNINGRULEFACTORY_H
#define EDLUT_LEARNINGRULEFACTORY_H

#include <string>
#include <map>
#include "boost/any.hpp"


#include "simulation/NetworkDescription.h"

class LearningRule;
struct ModelDescription;


/*!
 * \file LearningRuleFactory.h
 *
 * \author Jesus Garrido
 * \date June 2018
 *
 * This file declares a factory for LearningRule object creation.
 */

/*!
* \class LearningRuleFactory
*
* \brief Learning rule factory.
*
* This class declares the methods required for creating LearningRule objects.
*
* \author Jesus Garrido
* \date June 2018
*/

typedef ModelDescription (*ParseIntegrationMethodFunc) (FILE *);
typedef LearningRule * (*CreateLearningRuleFunc) (ModelDescription);
typedef std::map<std::string,boost::any> (*GetDefaultParamFunc) (void);

class LearningRuleFactory {

    private:

		/*!
		 * Create a class to store the pointers to the functions in the learning rule
		 */
		class LearningRuleClass {
			public:

			ParseIntegrationMethodFunc parseFunc;

			CreateLearningRuleFunc createFunc;

			GetDefaultParamFunc  getDefaultParamFunc;

			LearningRuleClass():
					parseFunc(0), createFunc(0), getDefaultParamFunc(0){};

			LearningRuleClass(ParseIntegrationMethodFunc parseFunc, CreateLearningRuleFunc createFunc, GetDefaultParamFunc getDefaultParamFunc):
					parseFunc(parseFunc), createFunc(createFunc), getDefaultParamFunc(getDefaultParamFunc){};
		};

		/*!
		 * Dictionary of the existing LearningRule classes and the Parsing and Creation functions
		 */
		static std::map<std::string, LearningRuleFactory::LearningRuleClass > LearningRuleMap;

    public:

		/*!
		* \brief Get all the available Learning Rules in a vector.
		*
		* It gets all the available Learning Rules in a vector.
		*/
		static std::vector<std::string> GetAvailableLearningRules();

		/*!
		* \brief Printing all the available Learning Rules.
		*
		* It prints all the available Learning Rules.
		*/
		static void PrintAvailableLearningRules();


		/*!
		 * \brief Filling of the factory.
		 *
		 * It creates a dictionary associating the name of each learning rule with the pointer to their static functions.
		 */
		 static void InitializeLearningRuleFactory();

		/*!
		 * \brief It creates a new learning rule object of the corresponding type.
		 *
		 * It creates a new learning rule object of the corresponding type.
		 *
		 * \param lruleDescription Description of the learning rule parameters.
		 *
		 * \throw EDLUTException If the learning rule does not exist with that name.
		 */
		 static LearningRule * CreateLearningRule(ModelDescription lruleDescription);

		/*!
		* \brief It parses the learning rule description from the file.
		*
		* It extracts the parameters of the learning rule from the file.
		*
		* \param ident Name of the learning rule to create.
		* \param fh File handler where the parameters can be readed.
		*
		* \throw EDLUTException If the learning rule does not exist with that name.
		*/
		static ModelDescription ParseLearningRule(std::string ident, FILE * fh);

		/*!
		 * \brief It returns the default parameters for the selected learning rule.
		 *
		 * It returns the default parameters for the selected learning rule.
		 *
		 * \param ident Name of the learning rule to create.
		 *
		 * \return A map with the default parameters of the learning rule.
		 *
		 * \throw EDLUTException If the learning rule does not exist with that name.
		 */
		static std::map<std::string,boost::any> GetDefaultParameters(std::string ident);

};

#endif //EDLUT_LEARNINGRULEFACTORY_H
