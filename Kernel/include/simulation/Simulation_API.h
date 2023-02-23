/***************************************************************************
 *                           Simulation_API.h                              *
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

#ifndef SIMULATION_API_H_
#define SIMULATION_API_H_

#include <string>
#include <list>
#include <map>
#include <vector>
#include <boost/any.hpp>

#include "../../include/communication/ArrayInputSpikeDriver.h"
#include "../../include/communication/ArrayInputCurrentDriver.h"
#include "../../include/communication/ArrayOutputSpikeDriver.h"

#include "../../include/communication/FileInputSpikeDriver.h"
#include "../../include/communication/FileInputCurrentDriver.h"
#include "../../include/communication/FileOutputSpikeDriver.h"
#include "../../include/communication/FileOutputWeightDriver.h"

#include "../../include/simulation/PrintableObject.h"

/*!
 * \file Simulation_API.h
 *
 * \author Jesus Garrido
 * \date June 2018
 *
 * This file declares the objects required for management of EDLUT simulations.
 */

class Simulation;
class EDLUTException;
struct NeuronLayerDescription;
struct ModelDescription;
struct SynapticLayerDescription;

/*!
 * \class Simulation_API
 *
 * \brief Spiking neural network simulation API.
 *
 * This class abstracts the data and functions required in order to create, run and manage simulations in EDLUT.
 *
 * \author Jesus Garrido
 * \date June 2018
 */
class Simulation_API{

	private:

    /*!
     * Simulation object
     */
    Simulation * simulation;

    /*!
     * Simulation is initialized
     */
    bool initialized;

    /*!
     * Simulation properties
     */
    std::map<std::string, boost::any> simulation_properties;

    /*!
     * Input spike drive
     */
    ArrayInputSpikeDriver input_spike_driver;

    /*!
     * Output spike driver
     */
    ArrayOutputSpikeDriver output_spike_driver;

    /*!
     * Input current driver
     */
    ArrayInputCurrentDriver input_current_driver;

  	/*!
  	* File input spike driver
  	*/
  	FileInputSpikeDriver * file_input_spike_driver;

  	/*!
  	* Output spike driver
  	*/
  	FileOutputSpikeDriver * file_output_spike_driver;

  	/*!
  	* File input current driver
  	*/
  	FileInputCurrentDriver * file_input_current_driver;

  	/*!
  	* File output spike and state monitor driver
  	*/
  	FileOutputSpikeDriver * file_output_monitor_driver;

  	/*!
  	* File output weight driver
  	*/
  	FileOutputWeightDriver * file_output_weight_driver;

  	/*!
  	* Periodic time used by the file_output_weight_driven used to store the synaptic weights.
  	*/
  	float save_weight_period;

    /*!
     * Max index of neuron created
     */
    int num_neuron_created;

    /*!
     * List of neuron layer descriptions
     */
    std::list<NeuronLayerDescription> neuron_layer_list;

    /*!
     * List of learning rule descriptions
     */
    std::list<ModelDescription> learning_rule_list;

    /*!
     * Max index of synapses created
     */
    int num_synapses_created;

    /*!
     * List of neuron layer descriptions
     */
    std::list<SynapticLayerDescription> synaptic_layer_list;

  public:

    /*!
     * \brief Class constructor with no parameters.
     *
     * It creates a new object an initializes its elements.
     */
    Simulation_API();

    /*!
     * \brief Class destructor with no parameters.
     *
     * It destroyes an object and releases the memory.
     */
    ~Simulation_API();

    /*!
     * \brief Add a new neuron layer with the given properties included in a dictionary.
     *
     * It adds a new neuron layer with the parameters indicated in the dictionary.
     *
     * \param num_neurons Number of neurons to include in this layer.
     * \param model_name Name of the neuron type.
     * \param param_dict Dictionary including the parameters of the model.
     * \param log_activity Indicating whether the activity will be registered in the logging devices.
     * \param output_activity Indicating whether the activity will be sent through the output devices.
     *
     * \return A vector with the indexes of the newly created neurons.
     *
     * \note This function does not creates the neuron layer in the network objects. It only adds the data in the
     * neuron layer register. The layer will be effectively created when the initialize function is called.
     */
    std::vector<int> AddNeuronLayer(int num_neurons, std::string model_name, std::map<std::string, boost::any> param_dict, bool log_activity, bool output_activity) noexcept(false);

    /*!
    * \brief Add a new synaptic layer with the given properties included in a dictionary.
    *
    * It adds a new synaptic layer with the parameters indicated in the dictionary.
    *
    * \param source_neuron List of indexes of the source_neurons.
    * \param target_neuron List of indexes of the target_neurons
    * \param param_dict Dictionary including the parameters of the synapses.
    *
    * \return A vector with the indexes of the newly created synapses.
    *
    * \note This function does not creates the synaptic layer in the network objects. It only adds the data in the
    * synaptic layer register. The layer will be effectively created when the initialize function is called.
    */
    std::vector<int> AddSynapticLayer(const std::vector<int> & source_neuron, const std::vector<int> & target_neuron, std::map<std::string, boost::any> param_dict) noexcept(false);

    /*!
    * \brief Inject spike activity to the output of the indicated neurons.
    *
    * It injects spike activity to the output of the indicated neurons.
    *
    * \param event_time Vector of spike times.
    * \param neuron_index Vector of spike indexes.
    */
    void AddExternalSpikeActivity(const std::vector<double> & event_time, const std::vector<long int> & neuron_index) noexcept(false);

    /*!
    * \brief Inject current spike to the output of the indicated neurons.
    *
    * It injects spike activity to the output of the indicated neurons.
    *
    * \param event_time Vector of current change events.
    * \param neuron_index Vector of current change neuron indexes.
    * \param current_value Vector of new current values.
    */
    void AddExternalCurrentActivity(const std::vector<double> & event_time, const std::vector<long int> & neuron_index, const std::vector<float> & current_value) noexcept(false);

    /*!
    * \brief Return the activity produced by the network.
    *
    * It retrieves the spike activity produced by the network.
    *
    * \param event_time Vector where the event times will be stored.
    * \param neuron_index Vector where the event neuron indexes will be stored.
    *
    * \note It remove the existing activity from the existing buffer.
    */
    void GetSpikeActivity(std::vector<double> & event_time, std::vector<long int> & neuron_index) noexcept(false);

  	/*!
  	* \brief create an input spike activity driver from a file.
  	*
  	* It creates an input spike activity driver from a file.
  	*
  	* \param FileName name of the file where the input activity is stored.
  	*/
  	void AddFileInputSpikeActivityDriver(string FileName) noexcept(false);

  	/*!
  	* \brief create an input current activity driver from a file.
  	*
  	* It creates an input current activity driver from a file.
  	*
  	* \param FileName name of the file where the input activity is stored.
  	*/
  	void AddFileInputCurrentActivityDriver(string FileName) noexcept(false);

  	/*!
  	* \brief create an output spike activity driver from a file.
  	*
  	* It create an output spike activity driver from a file.
  	*
  	* \param FileName name of the file where the input activity is stored.
  	*/
  	void AddFileOutputSpikeActivityDriver(string FileName) noexcept(false);

  	/*!
  	* \brief create an output monitor from a file.
  	*
  	* It create an output monitor driver from a file.
  	*
  	* \param FileName name of the file where the input activity is stored.
  	* \param monitor_state: when this value is false, the monitor just register spikes. By contrast, when this value is true, the monitor register both spikes and neuron states.
  	*/
  	void AddFileOutputMonitorDriver(string FileName, bool monitor_state) noexcept(false);

  	/*!
  	* \brief create an output monitor from a file.
  	*
  	* It create an output monitor driver from a file.
  	*
  	* \param FileName name of the file where the input activity is stored.
  	* \param save_period periodict time used to save the synaptic weights.
  	*/
  	void AddFileOutputWeightDriver(string FileName, float save_period) noexcept(false);

  	/*!
  	* \brief It returns all the synaptic weigths in a compressed format with two vector.
  	*
  	* It returns all the synaptic weigths in a compressed format with two vector.
  	*
  	* \param N_equal_weights vector storing in each value the number of synapses with the same weight.
  	* \param equal_weights vector storign in each value the weight of the synapses with the same weights.
  	*/
  	void GetCompressedWeights(std::vector<int> & N_equal_weights, std::vector<float> & equal_weights) noexcept(false);

  	/*!
  	* \brief It returns all the synaptic weigths in a extended format with a vector.
  	*
  	* It returns all the synaptic weigths in a extended format with a vector.
  	*
  	* \return vector with the synaptic weights.
  	*/
  	std::vector<float> GetWeights() noexcept(false);

  	/*!
  	* \brief It returns the solicited synaptic weigths in a extended format with a vector.
  	*
  	* It returns the solicited synaptic weigths in a extended format with a vector.
  	*
  	* \param synaptic_indexes synaptic indexes
  	*
  	* \return vector with the solicityed synaptic weights.
  	*/
  	std::vector<float> GetSelectedWeights(std::vector<int> synaptic_indexes) noexcept(false);

    /*!
    * \brief Initialize the network and simulation objects.
    *
    * It initializes the network and simulation objects.
    *
    * \note This function has to be called before calling RunSimulation.
    */
    void Initialize() noexcept(false);

    /*!
    * \brief Set EDLUT simulation parameters.
    *
    * It sets EDLUT general simulation parameters.
    *
    * \param param_dict Dictionary including the parameters of the simulation to be set.
    *
    * \note param_dict may include 'num_threads', 'num_simulation_queues' or 'resolution'.
    */
    void SetSimulationParameters(std::map<std::string, boost::any> param_dict) noexcept(false);

    /*!
    * \brief Run EDLUT simulation
    *
    * It simulates the network until the time indicated as a parameter.
    *
    * \param simulation_time Ending simulation time.
    *
    */
    void RunSimulation(double end_simulation_time) noexcept(false);

    // ----------------------------------------------
    // Learning rule related functions
    // ----------------------------------------------

    /*!
     * \brief Add a new learning rule with the given properties included in a dictionary.
     *
     * It adds a new learning rule with the parameters indicated in the dictionary.
     *
     * \param model_name Name of the learning rule type.
     * \param param_dict Dictionary including the parameters of the model.
     *
     * \return The index of the new learning rule.
     *
     * \note This function does not creates the learning rule in the network objects. It only adds the data in the
     * learning rule register. The learning rule will be effectively created when the initialize function is called.
     */
    int AddLearningRule(std::string model_name, std::map<std::string, boost::any> param_dict) noexcept(false);

    /*!
     * \brief Retrieve the default parameters of a learning rule.
     *
     * It retrieves the default parameters of a learning rule.
     *
     * \param model_name Name of the learning rule type.
     *
     * \return A dictionary with the default parameter values in the learning rule.
     *
     * \note An exception is raised if a learning rule does not exists with the provided name.
     */
    std::map<std::string, boost::any> GetLearningRuleDefParams(std::string model_name) noexcept(false);

    /*!
     * \brief Retrieve the default parameters of a neuron model.
     *
     * It retrieves the default parameters of a neuron model.
     *
     * \param model_name Name of the neuron model type.
     *
     * \return A dictionary with the default parameter values in the neuron model.
     *
     * \note An exception is raised if a learning rule does not exists with the provided name.
     */
    std::map<std::string, boost::any> GetNeuronModelDefParams(std::string model_name) noexcept(false);

    /*!
     * \brief Retrieve the default parameters of an integration method.
     *
     * It retrieves the default parameters of an integration method.
     *
     * \param model_name Name of the integration method.
     *
     * \return A dictionary with the default parameter values in the integration method.
     *
     * \note An exception is raised if the integration method does not exists with the provided name.
     */
    std::map<std::string, boost::any> GetIntegrationMethodDefParams(std::string model_name) noexcept(false);


    /*!
     * \brief Retrieve the parameters of a learning rule.
     *
     * It retrieves the parameters of a learning rule.
     *
     * \param lrule_index Index of the learning rule as returned by AddLearningRule function.
     *
     * \return A dictionary with the parameter values in the learning rule.
     *
     * \note An exception is raised if the learning rule index does not exist.
     */
    std::map<std::string, boost::any> GetLearningRuleParams(int lrule_index) noexcept(false);

    /*!
     * \brief Set the parameters of a learning rule.
     *
     * It sets the parameters of a learning rule.
     *
     * \param lrule_index Index of the learning rule as returned by AddLearningRule function.
     * \param newParam The new parameter values to be set.
     *
     * \note An exception is raised if the learning rule index does not exist.
     */
    void SetLearningRuleParams(int lrule_index, std::map<std::string, boost::any> newParam) noexcept(false);

    /*!
     * \brief Retrieve the parameters of a set of neurons (the original parameters share by all the neurons in the same neural layer).
     *
     * It retrieves the parameters of a set of neurons  (the original parameters share by all the neurons in the same neural layer).
     *
     * \param neuron_index Index of the neurons as returned by AddNeuronLayer function.
     *
     * \return A vector with one dictionary with the parameter values for each neuron.
     *
     * \note An exception is raised if the neuron index does not exist.
     */
    //float* GetLayerState(std::vector<int> neuron_indices, int &n_states) noexcept(false);

    /*!
     * \brief Retrieve the parameters of a set of neurons (the original parameters share by all the neurons in the same neural layer).
     *
     * It retrieves the parameters of a set of neurons  (the original parameters share by all the neurons in the same neural layer).
     *
     * \param neuron_index Index of the neurons as returned by AddNeuronLayer function.
     *
     * \return A vector with one dictionary with the parameter values for each neuron.
     *
     * \note An exception is raised if the neuron index does not exist.
     */
    std::vector<std::map<std::string, boost::any> > GetNeuronParams(std::vector<int> neuron_index) noexcept(false);

  	/*!
  	* \brief Retrieve the parameters of a set of neurons (in some neuron models this parameter can be different for each specific neuron).
  	*
  	* It retrieves the parameters of a set of neurons (in some neuron models this parameter can be different for each specific neuron).
  	*
  	* \param neuron_index Index of the neurons as returned by AddNeuronLayer function.
  	*
  	* \return A vector with one dictionary with the parameter values for each neuron.
  	*
  	* \note An exception is raised if the neuron index does not exist.
  	*/
  	std::vector<std::map<std::string, boost::any> > GetSpecificNeuronParams(std::vector<int> neuron_index) noexcept(false);

    /*!
     * \brief Set the parameters of a neuron layer.
     *
     * It sets the parameters of a neuron layer.
     *
     * \param nlayer_index Index of any of the neurons included in the neuron layer as returned by AddNeuronLayer function.
     * \param newParam The new parameter values to be set.
     *
     * \note An exception is raised if the neuron layer index does not exist.
     */
    void SetSpecificNeuronParams(int neuron_index, std::map<std::string, boost::any> newParam) noexcept(false);

    /*!
     * \brief Set the seed for the Random Generator
     *
     * It sets the seed for the Random Generator.
     *
     * \param seed new seed.
     */
     void SetRandomGeneratorSeed(int seed);


  	/*!
  	* \brief Get the available neuron models in EDLUT.
  	*
  	* It gets the available neuron models in EDLUT.
  	*/
  	static std::vector<std::string>  GetAvailableNeuronModels();

  	/*!
  	* \brief Print the available neuron models in EDLUT.
  	*
  	* It prints the available neuron models in EDLUT.
  	*/
  	static void PrintAvailableNeuronModels();


  	/*!
  	* \brief Get information about a neuron model.
  	*
  	* It gets information about a neuron model.
  	*
  	* \param neuronModelName neuron model name.
  	*
  	* \return A map with the neuron model information.
  	*
  	* \note An exception is raised if the neuron model does not exist.
  	*/
  	static std::map<std::string, std::string>  GetNeuronModelInfo(string neuronModelName) noexcept(false);

  	/*!
  	* \brief Print the available neuron models in EDLUT.
  	*
  	* It prints the available neuron models in EDLUT.
  	*
  	* \note An exception is raised if the neuron model does not exist.
  	*/
  	static void PrintNeuronModelInfo(string neuronModelName) noexcept(false);

  	/*!
  	* \brief Get the available integration methods in CPU for EDLUT.
  	*
  	* It gets the available integration methods in CPU for EDLUT.
  	*/
  	static std::vector<std::string>  GetAvailableIntegrationMethods();

  	/*!
  	* \brief Print the available integration methods in CPU for EDLUT.
  	*
  	* It prints the available integration methods in CPU for EDLUT.
  	*/
  	static void PrintAvailableIntegrationMethods();

  	/*!
  	* \brief Get the available learning rules in EDLUT.
  	*
  	* It gets the available learning rules in EDLUT.
  	*/
  	static std::vector<std::string>  GetAvailableLearningRules();

  	/*!
  	* \brief Print the available learning rules in EDLUT.
  	*
  	* It prints the available learning rules in EDLUT.
  	*/
  	static void PrintAvailableLearningRules();

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

#endif /*SIMULATION_API_H_*/
