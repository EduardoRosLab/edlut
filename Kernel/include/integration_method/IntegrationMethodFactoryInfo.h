/***************************************************************************
 *                           IntegrationMethodFactoryInfo.h                *
 *                           -------------------                           *
 * copyright            : (C) 2020 by Francisco Naveros                    *
 * email                : fnaveros@ugr.es                                  *
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 3 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

#ifndef INTEGRATIONMETHODFACTORYINFO_H
#define INTEGRATIONMETHODFACTORYINFO_H

#include <vector>
#include <string>
#include <map>
#include "boost/any.hpp"



/*!
 * \file IntegrationMethodFactoryInfo.h
 *
 * \author Francisco Naveros
 * \date April 2020
 *
 * This file declares an auxiliar class to obtain information about all the integration methods implemented for
 * time-driven neuron models in CPU.
 */

/*!
* \class IntegrationMethodFactoryInfo
*
* \brief Integration method factory info in CPU.
*
* This class declares the methods required to obtain information regarding the integration methods in CPU.
*
* \author Francisco Naveros
* \date April 2020
*/


class IntegrationMethodFactoryInfo {
public:
  /*!
  * \brief Get all the available Integration Methods in a vector.
  *
  * It gets all the available Integration Methods in a vector.
  */
  static std::vector<std::string> GetAvailableIntegrationMethods();



  /*!
  * \brief Printing all the available Integration Methods.
  *
  * It prints all the available Integration Methods.
  */
  static void PrintAvailableIntegrationMethods();
};

#endif //INTEGRATIONMETHODFACTORYINFO_H
