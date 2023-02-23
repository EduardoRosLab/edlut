/***************************************************************************
 *                           IntegrationMethodFactoryInfo.cpp              *
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

#include "../../include/integration_method/IntegrationMethodFactoryInfo.h"


//Integration methods in CPU
 #include "../../include/integration_method/BDFn.h"
 //#include "../../include/integration_method/Bifixed_BDFn.h"   //It doesn't work properly
 #include "../../include/integration_method/Bifixed_Euler.h"
 #include "../../include/integration_method/Bifixed_RK2.h"
 #include "../../include/integration_method/Bifixed_RK4.h"
 //#include "../../include/integration_method/BifixedStep.h"
 #include "../../include/integration_method/Euler.h"
 //#include "../../include/integration_method/FixedStep.h"
 //#include "../../include/integration_method/FixedStepSRM.h"
 #include "../../include/integration_method/RK2.h"
 #include "../../include/integration_method/RK4.h"

//Neuron model example in CPU
#include "../../include/neuron_model/LIFTimeDrivenModel.h"


std::vector<std::string> IntegrationMethodFactoryInfo::GetAvailableIntegrationMethods(){

	std::vector<std::string> availableIntegratonMethods;
	availableIntegratonMethods.push_back(BDFn<LIFTimeDrivenModel>::GetName());
//	availableIntegratonMethods.push_back(Bifixed_BDFn<LIFTimeDrivenModel>::GetName());
	availableIntegratonMethods.push_back(Bifixed_Euler<LIFTimeDrivenModel>::GetName());
	availableIntegratonMethods.push_back(Bifixed_RK2<LIFTimeDrivenModel>::GetName());
	availableIntegratonMethods.push_back(Bifixed_RK4<LIFTimeDrivenModel>::GetName());
	availableIntegratonMethods.push_back(Euler<LIFTimeDrivenModel>::GetName());
	availableIntegratonMethods.push_back(RK2<LIFTimeDrivenModel>::GetName());
	availableIntegratonMethods.push_back(RK4<LIFTimeDrivenModel>::GetName());

	return availableIntegratonMethods;
}


void IntegrationMethodFactoryInfo::PrintAvailableIntegrationMethods(){
	std::vector<std::string> availableIntegrationMethods = GetAvailableIntegrationMethods();
	cout << "- Available Integration Methods in CPU for EDLUT:" << endl;
	for (std::vector<std::string>::const_iterator it = availableIntegrationMethods.begin(); it != availableIntegrationMethods.end(); ++it){
		std::cout << "\t" << *it << std::endl;
	}
}
