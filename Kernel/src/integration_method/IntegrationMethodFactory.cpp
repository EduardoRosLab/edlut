/***************************************************************************
 *                           IntegrationMethodFactory.cpp                  *
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

#include "../../include/integration_method/IntegrationMethodFactory.h"
//CPU

#include "../../include/integration_method/BDFn.h"
//#include "../../include/integration_method/Bifixed_BDFn.h"   //It doesn't work properly
#include "../../include/integration_method/Bifixed_Euler.h"
#include "../../include/integration_method/Bifixed_RK2.h"
#include "../../include/integration_method/Bifixed_RK4.h"
#include "../../include/integration_method/Euler.h"
#include "../../include/integration_method/RK2.h"
#include "../../include/integration_method/RK4.h"

#include "../../include/neuron_model/AdExTimeDrivenModel.h"
#include "../../include/neuron_model/AdExTimeDrivenModelVector.h"
#include "../../include/neuron_model/EgidioGranuleCell_TimeDriven.h"
#include "../../include/neuron_model/HHTimeDrivenModel.h"
#include "../../include/neuron_model/IzhikevichTimeDrivenModel.h"
#include "../../include/neuron_model/LIFTimeDrivenModel.h"
#include "../../include/neuron_model/ALIFTimeDrivenModel.h"
#include "../../include/neuron_model/LIFTimeDrivenModel_IS.h"
#include "../../include/neuron_model/TimeDrivenInferiorOliveCell.h"
#include "../../include/neuron_model/TimeDrivenPurkinjeCell.h"
#include "../../include/neuron_model/TimeDrivenPurkinjeCell_IP.h"



template class IntegrationMethodFactory<AdExTimeDrivenModel>;
template class IntegrationMethodFactory<AdExTimeDrivenModelVector>;
template class IntegrationMethodFactory<EgidioGranuleCell_TimeDriven>;
template class IntegrationMethodFactory<HHTimeDrivenModel>;
template class IntegrationMethodFactory<IzhikevichTimeDrivenModel>;
template class IntegrationMethodFactory<LIFTimeDrivenModel>;
template class IntegrationMethodFactory<ALIFTimeDrivenModel>;
template class IntegrationMethodFactory<LIFTimeDrivenModel_IS>;
template class IntegrationMethodFactory<TimeDrivenInferiorOliveCell>;
template class IntegrationMethodFactory<TimeDrivenPurkinjeCell>;
template class IntegrationMethodFactory<TimeDrivenPurkinjeCell_IP>;

template <class Neuron_Model>
std::map<std::string, class IntegrationMethodFactory<Neuron_Model>::IntegrationMethodClass> IntegrationMethodFactory<Neuron_Model>::IntegrationMethodMap;



template<class Neuron_Model>
void IntegrationMethodFactory<Neuron_Model>::InitializeIntegrationMethodFactory(){
	if (IntegrationMethodFactory<Neuron_Model>::IntegrationMethodMap.empty()){
		IntegrationMethodFactory<Neuron_Model>::IntegrationMethodMap[BDFn<Neuron_Model>::GetName()] =
			IntegrationMethodFactory<Neuron_Model>::IntegrationMethodClass(
				BDFn<Neuron_Model>::ParseIntegrationMethod,
				BDFn<Neuron_Model>::CreateIntegrationMethod,
				BDFn<Neuron_Model>::GetDefaultParameters);
//		IntegrationMethodFactory<Neuron_Model>::IntegrationMethodMap[Bifixed_BDFn<Neuron_Model>::GetName()] =
//			IntegrationMethodFactory<Neuron_Model>::IntegrationMethodClass(
//				Bifixed_BDFn<Neuron_Model>::ParseIntegrationMethod,
//				Bifixed_BDFn<Neuron_Model>::CreateIntegrationMethod,
//				Bifixed_BDFn<Neuron_Model>::GetDefaultParameters);
		IntegrationMethodFactory<Neuron_Model>::IntegrationMethodMap[Bifixed_Euler<Neuron_Model>::GetName()] =
			IntegrationMethodFactory<Neuron_Model>::IntegrationMethodClass(
				Bifixed_Euler<Neuron_Model>::ParseIntegrationMethod,
				Bifixed_Euler<Neuron_Model>::CreateIntegrationMethod,
				Bifixed_Euler<Neuron_Model>::GetDefaultParameters);
		IntegrationMethodFactory<Neuron_Model>::IntegrationMethodMap[Bifixed_RK2<Neuron_Model>::GetName()] =
			IntegrationMethodFactory<Neuron_Model>::IntegrationMethodClass(
				Bifixed_RK2<Neuron_Model>::ParseIntegrationMethod,
				Bifixed_RK2<Neuron_Model>::CreateIntegrationMethod,
				Bifixed_RK2<Neuron_Model>::GetDefaultParameters);
		IntegrationMethodFactory<Neuron_Model>::IntegrationMethodMap[Bifixed_RK4<Neuron_Model>::GetName()] =
			IntegrationMethodFactory<Neuron_Model>::IntegrationMethodClass(
				Bifixed_RK4<Neuron_Model>::ParseIntegrationMethod,
				Bifixed_RK4<Neuron_Model>::CreateIntegrationMethod,
				Bifixed_RK4<Neuron_Model>::GetDefaultParameters);
		IntegrationMethodFactory<Neuron_Model>::IntegrationMethodMap[Euler<Neuron_Model>::GetName()] =
			IntegrationMethodFactory<Neuron_Model>::IntegrationMethodClass(
				Euler<Neuron_Model>::ParseIntegrationMethod,
				Euler<Neuron_Model>::CreateIntegrationMethod,
				Euler<Neuron_Model>::GetDefaultParameters);
		IntegrationMethodFactory<Neuron_Model>::IntegrationMethodMap[RK2<Neuron_Model>::GetName()] =
			IntegrationMethodFactory<Neuron_Model>::IntegrationMethodClass(
				RK2<Neuron_Model>::ParseIntegrationMethod,
				RK2<Neuron_Model>::CreateIntegrationMethod,
				RK2<Neuron_Model>::GetDefaultParameters);
		IntegrationMethodFactory<Neuron_Model>::IntegrationMethodMap[RK4<Neuron_Model>::GetName()] =
			IntegrationMethodFactory<Neuron_Model>::IntegrationMethodClass(
				RK4<Neuron_Model>::ParseIntegrationMethod,
				RK4<Neuron_Model>::CreateIntegrationMethod,
				RK4<Neuron_Model>::GetDefaultParameters);
	}
}


template<class Neuron_Model>
IntegrationMethod * IntegrationMethodFactory<Neuron_Model>::CreateIntegrationMethod(ModelDescription imethodDescription, Neuron_Model * nmodel){
    IntegrationMethodFactory<Neuron_Model>::InitializeIntegrationMethodFactory();
    // Find the rule description name in the learning rule map
typename
    std::map<std::string, IntegrationMethodFactory<Neuron_Model>::IntegrationMethodClass>::const_iterator it =
            IntegrationMethodFactory<Neuron_Model>::IntegrationMethodMap.find(imethodDescription.model_name);
    if (it==IntegrationMethodFactory<Neuron_Model>::IntegrationMethodMap.end()){
        throw EDLUTException(TASK_INTEGRATION_METHOD_TYPE, ERROR_INTEGRATION_METHOD_TYPE,
                                 REPAIR_INTEGRATION_METHOD_TYPE);
    }
	  IntegrationMethod * imethod = it->second.createFunc(imethodDescription, nmodel);
    return imethod;
}


template<class Neuron_Model>
ModelDescription IntegrationMethodFactory<Neuron_Model>::ParseIntegrationMethod(std::string ident, FILE * fh, long & Currentline) {
    IntegrationMethodFactory<Neuron_Model>::InitializeIntegrationMethodFactory();
    // Find the rule description name in the learning rule map
typename
    std::map<std::string, IntegrationMethodFactory<Neuron_Model>::IntegrationMethodClass >::const_iterator it =
            IntegrationMethodFactory<Neuron_Model>::IntegrationMethodMap.find(ident);
    if (it == IntegrationMethodFactory<Neuron_Model>::IntegrationMethodMap.end()) {
        throw EDLUTException(TASK_INTEGRATION_METHOD_TYPE, ERROR_INTEGRATION_METHOD_TYPE,
                             REPAIR_INTEGRATION_METHOD_TYPE);
    }
    return it->second.parseFunc(fh, Currentline);
}


template<class Neuron_Model>
std::map<std::string,boost::any> IntegrationMethodFactory<Neuron_Model>::GetDefaultParameters(std::string ident){
    IntegrationMethodFactory<Neuron_Model>::InitializeIntegrationMethodFactory();
    // Find the rule description name in the learning rule map
typename
    std::map<std::string, IntegrationMethodFactory<Neuron_Model>::IntegrationMethodClass >::const_iterator it =
            IntegrationMethodFactory<Neuron_Model>::IntegrationMethodMap.find(ident);
    if (it == IntegrationMethodFactory<Neuron_Model>::IntegrationMethodMap.end()) {
        throw EDLUTException(TASK_INTEGRATION_METHOD_TYPE, ERROR_INTEGRATION_METHOD_TYPE,
                             REPAIR_INTEGRATION_METHOD_TYPE);
    }
    return it->second.getDefaultParamFunc();
}
