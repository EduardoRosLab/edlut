/***************************************************************************
 *                           NetworkDescription.h                          *
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

#ifndef NETWORKDESCRIPTION_H_
#define NETWORKDESCRIPTION_H_

#include <cstring>
#include <map>
#include <vector>
#include <boost/any.hpp>

/*!
 * Struct defining learning rule information description
 */
struct ModelDescription{
    /*!
     * Learning rule model name
     */
    std::string model_name;

    /*!
     * Parameter map with the values of this learning rule
     */
    std::map<std::string, boost::any> param_map;
};


/*!
 * Struct defining neuron layer information description
 */
struct NeuronLayerDescription{
    /*!
     * Number of neurons
     */
    int num_neurons;

    /*!
     * Defined with parameter file
     */
    ModelDescription neuron_model;

    /*!
     * Indication of logging activity
     */
    bool log_activity;

    /*!
     * Indication of sending activity to output devices
     */
    bool output_activity;

};


    /*!
     * Struct defining synaptic layer information description
     */
struct SynapticLayerDescription{
    /*!
     * List of source neurons
     */
    std::vector<int> source_neuron_list;

    /*!
     * List of target neurons
     */
    std::vector<int> target_neuron_list;

    /*!
     * Parameter map with the values of this layer
     */
    std::map<std::string, boost::any> param_map;
};

#endif //EDLUT_NETWORKDESCRIPTION_H
