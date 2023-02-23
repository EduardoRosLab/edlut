from cython.operator cimport dereference, preincrement
from libcpp.string cimport string as cpp_string
from libcpp.map cimport map as cpp_map
from libcpp.vector cimport vector as cpp_vector

from simulation_api cimport Simulation_API, any_cast, ModelDescription, TimeScale
from simulation_api cimport any as cpp_any

#cimport numpy as np
import numpy as np

#from cython cimport view


class PyModelDescription:
    #TODO: Docstrings

    def __init__(self, model_name, params_dict):
        self.model_name = model_name
        self.params_dict = params_dict


cdef class PySimulation_API:
    """
    Python wrapper class for the C++ class Simulation_API.

    Author: Alvaro Gonzalez
    Date: July 2018
    """

    cdef Simulation_API cpp_sim  #Wrapped C++ instance

    # vector[int] AddNeuronLayer(int, string, map[string,any], bool, bool) except +
    def AddNeuronLayer(self, num_neurons=1, model_name='InputSpikeNeuronModel', param_dict={}, log_activity=False, output_activity=False):
        """
        Adds a layer of neurons. Returns a list with created neuron's indices.

        Note: This function does not creates the neuron layer in the network objects. It only adds the data in the neuron layer register. The layer will be effectively created when the initialize function is called.

        :param num_neurons: Number of neurons to create.
        :param model_name: Name of the neuron model.
        :param param_dict: 'dict' with all the parameters required by the neuron model.
        :param log_activity: Indicates if activity of this layer should be logged.
        :param output_activity: Indicates if activity of this layer will be sent to output devices.
        :return: A list with created neuron's indices.

        :type num_neurons: int
        :type model_name: str
        :type param_dict: dict
        :type log_activity: bool
        :type output_activity: bool
        :rtype: list[int]
        """
        cpp_model_name = py2cpp_str2string(model_name)
        cdef cpp_map[cpp_string, cpp_any] cpp_param_dict = py2cpp_dict2map(param_dict)
        neuron_list = self.cpp_sim.AddNeuronLayer(num_neurons, cpp_model_name, cpp_param_dict, log_activity, output_activity)

        return neuron_list

    # map[string,any] GetNeuronModelDefParams(string) except +
    def GetNeuronModelDefParams(self, param_name):
        """
        Returns the default parameters for the selected neuron model.

        :param param_name: Name of the neuron model to read.
        :return: A dict with the defaults parameters of the neuron model.
        :type param_name: str
        :rtype: dict
        :exception: An exception is raised if a learning rule does not exists with the provided name. C++ exceptions turns to Python exceptions following this rules: http://cython.readthedocs.io/en/latest/src/userguide/wrapping_CPlusPlus.html#exceptions
        """
        cpp_param_name = py2cpp_str2string(param_name)
        cdef cpp_map[cpp_string, cpp_any] cpp_param_dict
        cpp_param_dict = self.cpp_sim.GetNeuronModelDefParams(cpp_param_name)
        py_dict = {}

        cdef cpp_map[cpp_string, cpp_any].iterator m_end = cpp_param_dict.end()
        cdef cpp_map[cpp_string, cpp_any].iterator m_it = cpp_param_dict.begin()
        while m_it != m_end:
            key = cpp2py_string2str(dereference(m_it).first)
            value = cpp2py_any2py(dereference(m_it).second)
            py_dict[key] = value
            preincrement(m_it)

        return py_dict

    # void Initialize() except +
    def Initialize(self):
        """
        Sets EDLUT general simulation parameters.

        Note: This function has to be called before calling RunSimulation.

        :return: None
        """
        self.cpp_sim.Initialize()

    # vector[int] AddSynapticLayer(const vector[int] &, const vector[int] &, map[string,any]) except +
    def AddSynapticLayer(self, source_list, target_list, param_dict):
        """
        Add a new synaptic layer with the given properties included in a dictionary.

        Note: This function does not creates the synaptic layer in the network objects. It only adds the data in the synaptic layer register. The layer will be effectively created when the initialize function is called.

        :param source_list: List of indexes of the source_neurons.
        :param target_list: List of indexes of the target_neurons
        :param param_dict: Dictionary including the parameters of the synapses.
        :return: A list with the indices of the newly created synapses.

        :type source_list: list[int]
        :type target_list: list[int]
        :type param_dict: dict
        :rtype: list[int]
        """
        synapse_list = self.cpp_sim.AddSynapticLayer(
                py2cpp_list2vectori(source_list),
                py2cpp_list2vectori(target_list),
                py2cpp_dict2map(param_dict))
        return synapse_list

    # void AddExternalSpikeActivity(const vector[double] &, const vector[long] &) except +
    def AddExternalSpikeActivity(self, event_times_list, neuron_indices_list):
        """
        It injects spike activity to the output of the indicated neurons.

        :param event_times_list: List of spike times.
        :param neuron_indices_list: Vector of spike indexes.
        :return: None

        :type event_times_list: list[float]
        :type neuron_indices_list: list[int]
        """
        self.cpp_sim.AddExternalSpikeActivity(
                py2cpp_list2vectord(event_times_list),
                py2cpp_list2vectorl(neuron_indices_list))

    # void AddExternalCurrentActivity(const vector[double] &, const vector[long] &, const vector[float] &) except +
    def AddExternalCurrentActivity(self, event_times_list, neuron_indices_list, current_values_list):
        """
        It injects spike activity to the output of the indicated neurons.

        :param event_times_list: List of current change events.
        :param neuron_indices_list: List of current change neuron indexes.
        :param current_values_list: List of new current values.
        :return: None

        :type event_times_list: list[float]
        :type neuron_indices_list: list[int]
        :type current_values_list: list[float]
        """
        self.cpp_sim.AddExternalCurrentActivity(
                py2cpp_list2vectord(event_times_list),
                py2cpp_list2vectorl(neuron_indices_list),
                py2cpp_list2vectorf(current_values_list))

    # void GetSpikeActivity(vector[double] &, vector[long] &) except +
    # This will return two lists: event_times and neuron_indices
    def GetSpikeActivity(self):
        """
        It retrieves the spike activity produced by the network.

        Note: It remove the existing activity from the existing buffer.

        :return: A tuple (event_time, neuron_index) with two lists, one for the times of the events, one for the neuron indices.
        :rtype: tuple(list[float], list[int])
        """
        cdef cpp_vector[double] cpp_event_times
        cdef cpp_vector[long]   cpp_neuron_indices
        self.cpp_sim.GetSpikeActivity(cpp_event_times, cpp_neuron_indices)

        event_times = cpp_event_times
        neuron_indices = cpp_neuron_indices

        return (event_times, neuron_indices)

    # void GetCompressedWeights(vector[int] &, vector[float] &) except +
    # This will return two lists: N_equal_synapses and equal_synapses
    def GetCompressedWeights(self):
        """
        It retrieves the synaptic weights of all the network in a compressed format.

        :return: A tuple (N_equal_synapses, equal_synapses) with two lists, one for the number of synpases with the same weight, one for the synaptics weights.
        :rtype: tuple(list[int], list[float])
        """
        cdef cpp_vector[int] cpp_N_equal_synapses
        cdef cpp_vector[float] cpp_equal_synapses
        self.cpp_sim.GetCompressedWeights(cpp_N_equal_synapses, cpp_equal_synapses)

        N_equal_synapses = cpp_N_equal_synapses
        equal_synapses = cpp_equal_synapses

        return (N_equal_synapses, equal_synapses)

    # vector[float] GetWeights() except +
    # This will return a lists: weights
    def GetWeights(self):
        """
        It retrieves the synaptic weights of all the synapses in a extended format (one value for each synapse).

        :return: A list (weights) with the synaptics weights.
        :rtype: list[float]
        """
        cdef cpp_vector[float] cpp_weights
        cpp_weights = self.cpp_sim.GetWeights()

        weights = cpp_weights

        return (weights)

    # vector[float] GetSelectedWeights(vector[int]) except +
    # This will return a lists: weights
    def GetSelectedWeights(self, indexes):
        """
        It retrieves the synaptic weights of the solicited synapses (one value for each synapse).

        :param indexes: list of solicited synapses
        :return: A list (weights) with the synaptics weights.
        :rtype: list[float]
        """
        cdef cpp_vector[int] cpp_indexes = py2cpp_list2vectori(indexes)
        cdef cpp_vector[float] cpp_weights
        cpp_weights = self.cpp_sim.GetSelectedWeights(cpp_indexes)

        weights = cpp_weights

        return (weights)

    # void SetSimulationParameters(map[string,any]) except +
    def SetSimulationParameters(self, params_dict):
        """
        It sets EDLUT general simulation parameters.

        :param params_dict: Dictionary including the parameters of the simulation to be set. 'param_dict' may include 'num_threads', 'num_simulation_queues' or 'resolution' keys.
        :return: None
        """
        cdef cpp_map[cpp_string, cpp_any] cpp_params = py2cpp_dict2map(params_dict)
        self.cpp_sim.SetSimulationParameters(cpp_params)

    # void RunSimulation(double) except +
    def RunSimulation(self, end_simulation_time):
        """
        It simulates the network until the time indicated as a parameter.

        :param end_simulation_time: Ending simulation time, in seconds.
        :type end_simulation_time: float
        :return: None
        """
        self.cpp_sim.RunSimulation(end_simulation_time)

    # int AddLearningRule(string, map[string,any]) except +
    def AddLearningRule(self, model_name, param_dict):
        """
        It adds a new learning rule with the parameters indicated in the dictionary.

        Note: This function does not creates the learning rule in the network objects. It only adds the data in the learning rule register. The learning rule will be effectively created when the initialize function is called.

        :param model_name: Name of the learning rule type.
        :param param_dict: Dictionary including the parameters of the model.
        :return: The index of the new learning rule.

        :type model_name: str
        :type param_dict: dict
        :rtype: int
        """
        return self.cpp_sim.AddLearningRule(
                py2cpp_str2string(model_name),
                py2cpp_dict2map(param_dict))

    # map[string,any] GetLearningRuleDefParams(string) except +
    def GetLearningRuleDefParams(self, model_name):
        """
        It retrieves the default parameters of a learning rule.

        :param model_name: Name of the learning rule type.
        :return: A dictionary with the default parameter values in the learning rule.

        :type model_name: str
        :rtype: dict

        :exception: An exception is raised if a learning rule does not exists with the provided name. C++ exceptions turns to Python exceptions following this rules: http://cython.readthedocs.io/en/latest/src/userguide/wrapping_CPlusPlus.html#exceptions
        """
        cdef cpp_map[cpp_string,cpp_any] cpp_params
        cpp_params = self.cpp_sim.GetLearningRuleDefParams(
                py2cpp_str2string(model_name))
        return cpp2py_map2dict(cpp_params)

    # map[string,any] GetIntegrationMethodDefParams(string) except +
    def GetIntegrationMethodDefParams(self, model_name):
        """
        It retrieves the default parameters of an integration method.

        :param model_name: Name of the integration method.
        :return: A dictionary with the default parameter values in the integration method.

        :type model_name: str
        :rtype: dict

        :exception: An exception is raised if the integration method does not exists with the provided name. C++ exceptions turns to Python exceptions following this rules: http://cython.readthedocs.io/en/latest/src/userguide/wrapping_CPlusPlus.html#exceptions
        """
        cdef cpp_map[cpp_string,cpp_any] cpp_params
        cpp_params = self.cpp_sim.GetIntegrationMethodDefParams(
                py2cpp_str2string(model_name))
        return cpp2py_map2dict(cpp_params)

    # map[string,any] GetLearningRuleParams(int) except +
    def GetLearningRuleParams(self, rule_index):
        """
        It retrieves the parameters of a learning rule.

        :param rule_index: Index of the learning rule as returned by AddLearningRule function.
        :return: A dictionary with the parameter values in the learning rule.

        :exception: An exception is raised if the learning rule index does not exist. C++ exceptions turns to Python exceptions following this rules: http://cython.readthedocs.io/en/latest/src/userguide/wrapping_CPlusPlus.html#exceptions
        """
        cdef cpp_map[cpp_string,cpp_any] cpp_params
        cpp_params = self.cpp_sim.GetLearningRuleParams(<int>rule_index)
        return cpp2py_map2dict(cpp_params)

    # void SetLearningRuleParams(int, map[string,any]) except +
    def SetLearningRuleParams(self, rule_index, params_dict):
        """
        It sets the parameters of a learning rule.

        :param rule_index: Index of the learning rule as returned by AddLearningRule function.
        :param params_dict: The new parameter values to be set.
        :return: None
        :type rule_index: int
        :type params_dict: dict

        :exception: An exception is raised if the learning rule index does not exist. C++ exceptions turns to Python exceptions following this rules: http://cython.readthedocs.io/en/latest/src/userguide/wrapping_CPlusPlus.html#exceptions
        """
        self.cpp_sim.SetLearningRuleParams(
                <int>rule_index,
                py2cpp_dict2map(params_dict))

    ## vector[vector[float]] GetNeuronState(vector[int]) except +
    #def GetLayerState(self, neuron_indices_list):
    #    """
    #    It retrieves the state of a set of neurons.

    #    :param neuron_indices_list: Indices of the neurons as returned by AddNeuronLayer function.
    #    :return: A Numpy array with the state values of size <neurons,states>.

    #    :type neuron_indices_list: list[int]
    #    :rtype: list[list[float]]

    #    :exception: An exception is raised if the neuron index does not exist. C++ exceptions turns to Python exceptions following this rules: http://cython.readthedocs.io/en/latest/src/userguide/wrapping_CPlusPlus.html#exceptions
    #    """
    #    #cdef cpp_vector[int] cpp_indices = py2cpp_list2vectori(neuron_indices_list)
    #    n_neurons = len(neuron_indices_list)
    #    cdef int n_states

    #    cdef float* cpp_state_ptr = self.cpp_sim.GetLayerState(neuron_indices_list, n_states)
    #    cdef view.array cpp_state = view.array(shape=(n_neurons,n_states), itemsize=sizeof(float), format="f", allocate_buffer=False)
    #    cpp_state.data = <char *> cpp_state_ptr

    #    return np.asarray(cpp_state)

    #    #my_arr = <float[:n_states]> cpp_state_ptr
    #    #return np.asarray(my_arr)

    # vector[map[string,any]] GetNeuronParams(vector[int]) except +
    def GetNeuronParams(self, neuron_indices_list):
        """
        It retrieves the parameters of a set of neurons.

        :param neuron_indices_list: Indices of the neurons as returned by AddNeuronLayer function.
        :return: A vector with one dictionary with the parameter values for each neuron.

        :type neuron_indices_list: list[int]
        :rtype: list[dict]

        :exception: An exception is raised if the neuron index does not exist. C++ exceptions turns to Python exceptions following this rules: http://cython.readthedocs.io/en/latest/src/userguide/wrapping_CPlusPlus.html#exceptions
        """
        cdef cpp_vector[int] cpp_indices
        cpp_indices = py2cpp_list2vectori(neuron_indices_list)
        cdef cpp_vector[cpp_map[cpp_string,cpp_any]] cpp_vector_of_maps
        cpp_vector_of_maps = self.cpp_sim.GetNeuronParams(cpp_indices)

        cdef cpp_vector[cpp_map[cpp_string,cpp_any]].iterator v_it, v_end
        cdef cpp_map[cpp_string,cpp_any] m
        cdef cpp_map[cpp_string, cpp_any].iterator m_it, m_end

        # Traverse all the maps in the vector
        py_list = []
        v_it  = cpp_vector_of_maps.begin()
        v_end = cpp_vector_of_maps.end()
        while v_it != v_end:
            # Traverse all the items in the map
            m = dereference(v_it)
            py_dict = {}
            m_end = m.end()
            m_it  = m.begin()
            while m_it != m_end:
                key = cpp2py_string2str(dereference(m_it).first)
                value = cpp2py_any2py(dereference(m_it).second)
                py_dict[key] = value
                preincrement(m_it)

            py_list.append(py_dict)
            preincrement(v_it)

        return py_list

    # vector[map[string,any]] GetSpecificNeuronParams(vector[int]) except +
    def GetSpecificNeuronParams(self, neuron_indices_list):
        """
        It retrieves the parameters of a set of neurons.

        :param neuron_indices_list: Indices of the neurons as returned by AddNeuronLayer function.
        :return: A vector with one dictionary with the parameter values for each neuron.

        :type neuron_indices_list: list[int]
        :rtype: list[dict]

        :exception: An exception is raised if the neuron index does not exist. C++ exceptions turns to Python exceptions following this rules: http://cython.readthedocs.io/en/latest/src/userguide/wrapping_CPlusPlus.html#exceptions
        """
        cdef cpp_vector[int] cpp_indices
        cpp_indices = py2cpp_list2vectori(neuron_indices_list)
        cdef cpp_vector[cpp_map[cpp_string,cpp_any]] cpp_vector_of_maps
        cpp_vector_of_maps = self.cpp_sim.GetSpecificNeuronParams(cpp_indices)

        cdef cpp_vector[cpp_map[cpp_string,cpp_any]].iterator v_it, v_end
        cdef cpp_map[cpp_string,cpp_any] m
        cdef cpp_map[cpp_string, cpp_any].iterator m_it, m_end

        # Traverse all the maps in the vector
        py_list = []
        v_it  = cpp_vector_of_maps.begin()
        v_end = cpp_vector_of_maps.end()
        while v_it != v_end:
            # Traverse all the items in the map
            m = dereference(v_it)
            py_dict = {}
            m_end = m.end()
            m_it  = m.begin()
            while m_it != m_end:
                key = cpp2py_string2str(dereference(m_it).first)
                value = cpp2py_any2py(dereference(m_it).second)
                py_dict[key] = value
                preincrement(m_it)

            py_list.append(py_dict)
            preincrement(v_it)

        return py_list

    # void SetSpecificNeuronParams(int, map[string,any]) except +
    def SetSpecificNeuronParams(self, neuron_index, params_dict):
        """
        It sets the parameters of a neuron layer.

        :param neuron_index: Index of any of the neurons included in the neuron layer as returned by AddNeuronLayer function.
        :param params_dict: The new parameter values to be set.
        :return: None

        :type neuron_index: int
        :type params_dict: dict

        :exception: An exception is raised if the neuron layer index does not exist. C++ exceptions turns to Python exceptions following this rules: http://cython.readthedocs.io/en/latest/src/userguide/wrapping_CPlusPlus.html#exceptions
        """
        self.cpp_sim.SetSpecificNeuronParams(
                <int>neuron_index,
                py2cpp_dict2map(params_dict))

    # map[string,any] GetLearningRuleDefParams(string) except +
    def GetLearningRuleDefParams(self, rule_name):
        """
        It retrieves the default parameters of a learning rule.

        :param rule_name: Name of the learning rule type.
        :return: A dictionary with the default parameter values in the learning rule.

        :type rule_name: str
        :rtype: dict

        :exception: An exception is raised if a learning rule does not exists with the provided name. C++ exceptions turns to Python exceptions following this rules: http://cython.readthedocs.io/en/latest/src/userguide/wrapping_CPlusPlus.html#exceptions
        """
        cdef cpp_map[cpp_string,cpp_any] cpp_params
        cpp_params = self.cpp_sim.GetLearningRuleDefParams(
                py2cpp_str2string(rule_name))
        return cpp2py_map2dict(cpp_params)

    # vector[string] GetAvailableNeuronModels() except +
    def GetAvailableNeuronModels(self):
        """
        Gets all the available neuron models in EDLUT.
        """
        cdef cpp_vector[cpp_string] cpp__neuron_models
        cpp__neuron_models = self.cpp_sim.GetAvailableNeuronModels()
        cdef cpp_vector[cpp_string].iterator v_it, v_end

        py__neuron_models = []
        v_it = cpp__neuron_models.begin()
        v_end = cpp__neuron_models.end()
        while v_it != v_end:
            py__neuron_model = cpp2py_string2str(dereference(v_it))
            py__neuron_models.append(py__neuron_model)
            preincrement(v_it)

        return py__neuron_models

    # void PrintAvailableNeuronModels() except +
    def PrintAvailableNeuronModels(self):
        """
        Prints all the available neuron models in EDLUT.
        """
        self.cpp_sim.PrintAvailableNeuronModels()

    # map[string,string] GetNeuronModelInfo(string) except +
    def GetNeuronModelInfo(self, model_name):
        """
        It retrieves information about a neuron model.

        :param model_name: Name of the neuron model.
        :return: A dictionary with all the information.

        :type model_name: str
        :rtype: dict

        :exception: An exception is raised if a neuron model does not exists with the provided name. C++ exceptions turns to Python exceptions following this rules: http://cython.readthedocs.io/en/latest/src/userguide/wrapping_CPlusPlus.html#exceptions
        """
        cdef cpp_map[cpp_string,cpp_string] cpp_params
        cpp_params = self.cpp_sim.GetNeuronModelInfo(py2cpp_str2string(model_name))
        return cpp2py_map2dict_string(cpp_params)

    # void PrintNeuronModelInfo(string) except +
    def PrintNeuronModelInfo(self, model_name):
        """
        It print information about a neuron model.

        :param model_name: Name of the neuron model.
        :return: A dictionary with all the information.

        :type model_name: str
        :rtype: void

        :exception: An exception is raised if a neuron model does not exists with the provided name. C++ exceptions turns to Python exceptions following this rules: http://cython.readthedocs.io/en/latest/src/userguide/wrapping_CPlusPlus.html#exceptions
        """
        self.cpp_sim.PrintNeuronModelInfo(py2cpp_str2string(model_name))

    # map[string,string] GetVectorizableParameters(string) except +
    def GetVectorizableParameters(self, model_name):
        """
        It retrieves information about a neuron model.

        :param model_name: Name of the neuron model.
        :return: A dictionary with all the information.

        :type model_name: str
        :rtype: dict

        :exception: An exception is raised if a neuron model does not exists with the provided name. C++ exceptions turns to Python exceptions following this rules: http://cython.readthedocs.io/en/latest/src/userguide/wrapping_CPlusPlus.html#exceptions
        """
        cdef cpp_map[cpp_string,cpp_string] cpp_params
        cpp_params = self.cpp_sim.GetVectorizableParameters(py2cpp_str2string(model_name))
        return cpp2py_map2dict_string(cpp_params)

    # void PrintVectorizableParameters(string) except +
    def PrintVectorizableParameters(self, model_name):
        """
        It print information about a neuron model.

        :param model_name: Name of the neuron model.
        :return: A dictionary with all the information.

        :type model_name: str
        :rtype: void

        :exception: An exception is raised if a neuron model does not exists with the provided name. C++ exceptions turns to Python exceptions following this rules: http://cython.readthedocs.io/en/latest/src/userguide/wrapping_CPlusPlus.html#exceptions
        """
        self.cpp_sim.PrintVectorizableParameters(py2cpp_str2string(model_name))

    # vector[string] GetAvailableIntegrationMethods() except +
    def GetAvailableIntegrationMethods(self):
        """
        Gets all the available integration methods in CPU for EDLUT.
        """
        cdef cpp_vector[cpp_string] cpp__integration_methods
        cpp__integration_methods = self.cpp_sim.GetAvailableIntegrationMethods()
        cdef cpp_vector[cpp_string].iterator v_it, v_end

        py__integration_methods = []
        v_it = cpp__integration_methods.begin()
        v_end = cpp__integration_methods.end()
        while v_it != v_end:
            py__integration_method = cpp2py_string2str(dereference(v_it))
            py__integration_methods.append(py__integration_method)
            preincrement(v_it)

        return py__integration_methods

    # void PrintAvailableIntegrationMethods() except +
    def PrintAvailableIntegrationMethods(self):
        """
        Prints all the available integration methods in CPU for EDLUT.
        """
        self.cpp_sim.PrintAvailableIntegrationMethods()

    # vector[string] GetAvailableLearningRules() except +
    def GetAvailableLearningRules(self):
        """
        Gets all the available learning rules in EDLUT.
        """
        cdef cpp_vector[cpp_string] cpp__learning_rues
        cpp__learning_rules = self.cpp_sim.GetAvailableLearningRules()
        cdef cpp_vector[cpp_string].iterator v_it, v_end

        py__learning_rules = []
        v_it = cpp__learning_rules.begin()
        v_end = cpp__learning_rules.end()
        while v_it != v_end:
            py__learning_rule = cpp2py_string2str(dereference(v_it))
            py__learning_rules.append(py__learning_rule)
            preincrement(v_it)

        return py__learning_rules

    # void PrintAvailableLearningRules() except +
    def PrintAvailableLearningRules(self):
        """
        Prints all the available learning rules in EDLUT.
        """
        self.cpp_sim.PrintAvailableLearningRules()

    # void AddFileInputSpikeActivityDriver(string) except +
    def AddFileInputSpikeActivityDriver(self, file_name):
        """
        It loads inputs spikes from a file.

        :param file_name: Name of the file where the input spikes are defined.

        :type file_name: str
        :rtype: void

        :exception: An exception is raised if the input file can not be read. C++ exceptions turns to Python exceptions following this rules: http://cython.readthedocs.io/en/latest/src/userguide/wrapping_CPlusPlus.html#exceptions
        """
        self.cpp_sim.AddFileInputSpikeActivityDriver(py2cpp_str2string(file_name))


    # void AddFileInputCurrentActivityDriver(string) except +
    def AddFileInputCurrentActivityDriver(self, file_name):
        """
        It loads inputs spikes from a file.

        :param file_name: Name of the file where the input spikes are defined.

        :type file_name: str
        :rtype: void

        :exception: An exception is raised if the input file can not be read. C++ exceptions turns to Python exceptions following this rules: http://cython.readthedocs.io/en/latest/src/userguide/wrapping_CPlusPlus.html#exceptions
        """
        self.cpp_sim.AddFileInputCurrentActivityDriver(py2cpp_str2string(file_name))


    # void AddFileOutputSpikeActivityDriver(string) except +
    def AddFileOutputSpikeActivityDriver(self, file_name):
        """
        It stores the output activity in the specified file.

        :param file_name: Name of the file where the output activity will be stored.

        :type file_name: str
        :rtype: void

        :exception: An exception is raised if the output file can not created. C++ exceptions turns to Python exceptions following this rules: http://cython.readthedocs.io/en/latest/src/userguide/wrapping_CPlusPlus.html#exceptions
        """
        self.cpp_sim.AddFileOutputSpikeActivityDriver(py2cpp_str2string(file_name))


    # void AddFileOutputMonitorDriver(string, bool) except +
    def AddFileOutputMonitorDriver(self, file_name, monitor_state):
        """
        It stores the monitored activity and neuron states (this last one just in case monitor_state fixed to true) in the specified file.

        :param file_name: Name of the file where the monitores activity and neuron states will be stored.
        :param monitor_state: when this value is false, the monitor just register spikes. By contrast, when this value is true, the monitor register both spikes and neuron states.

        :type file_name: str
        :type monitor_state: bool
        :rtype: void

        :exception: An exception is raised if the output file can not created. C++ exceptions turns to Python exceptions following this rules: http://cython.readthedocs.io/en/latest/src/userguide/wrapping_CPlusPlus.html#exceptions
        """
        self.cpp_sim.AddFileOutputMonitorDriver(py2cpp_str2string(file_name), monitor_state)


    # void AddFileOutputWeightDriver(string, double) except +
    def AddFileOutputWeightDriver(self, file_name, save_period):
        """
        It stores all the synapstic weights the specified file each save_period interval.

        :param file_name: Name of the file where the synaptic weights will be stored.
        :param save_period: periodic time used to store the synaptic weights.

        :type file_name: str
        :type save_period: double
        :rtype: void

        :exception: An exception is raised if the output file can not created. C++ exceptions turns to Python exceptions following this rules: http://cython.readthedocs.io/en/latest/src/userguide/wrapping_CPlusPlus.html#exceptions
        """
        self.cpp_sim.AddFileOutputWeightDriver(py2cpp_str2string(file_name), save_period)

    # void SetRandomGeneratorSeed(int)
    def SetRandomGeneratorSeed(self, seed):
        """
        It sets the Random Generator seed.

        :param seed: Random Generator seed.
        :return: None
        :type seed: int

        :exception: An exception is raised if the seed value is not valid. C++ exceptions turns to Python exceptions following this rules: http://cython.readthedocs.io/en/latest/src/userguide/wrapping_CPlusPlus.html#exceptions
        """
        self.cpp_sim.SetRandomGeneratorSeed(
                <int>seed)


cdef object cpp2py_string2str(cpp_string s):
    """
    Casts a C++ string to a Python str

    :param s: C++ string to be casted.
    :type s: std::string
    :return: A Python string object.
    :rtype: str
    """
    return s.decode('utf-8')


cdef object cpp2py_map2dict(cpp_map[cpp_string,cpp_any] m):
    """
    Casts a C++ map to a Python dict object.

    :param m: A C++ map to be casted.
    :type m: std::map<std::string,boost::any>
    :return: A Python dict object.
    :rtype: dict
    """
    py_dict = {}

    cdef cpp_map[cpp_string, cpp_any].iterator m_end = m.end()
    cdef cpp_map[cpp_string, cpp_any].iterator m_it  = m.begin()
    while m_it != m_end:
        key = cpp2py_string2str(dereference(m_it).first)
        value = cpp2py_any2py(dereference(m_it).second)
        py_dict[key] = value
        preincrement(m_it)

    return py_dict

cdef object cpp2py_map2dict_string(cpp_map[cpp_string,cpp_string] m):
    """
    Casts a C++ map to a Python dict object.

    :param m: A C++ map to be casted.
    :type m: std::map<std::string,std::string>
    :return: A Python dict object.
    :rtype: dict
    """
    py_dict = {}

    cdef cpp_map[cpp_string, cpp_string].iterator m_end = m.end()
    cdef cpp_map[cpp_string, cpp_string].iterator m_it  = m.begin()
    while m_it != m_end:
        key = cpp2py_string2str(dereference(m_it).first)
        value = cpp2py_string2str(dereference(m_it).second)
        py_dict[key] = value
        preincrement(m_it)

    return py_dict


cdef object cpp2py_any2py(cpp_any o):
    """
    Tries to cast a C++ boost::any object to some Python type.

    :param o: A C++ object to be casted.
    :type o: boost::any
    :return: A Python object. Supported types are int, float, str, PyModelDescription, TimeScale.
    """
    #TODO: Docstrings

    # Basic types
    try:
        return any_cast[int](o)
    except:
        pass

    try:
        return any_cast[float](o)
    except:
        pass

    # String type
    try:
        cpp_text = any_cast[cpp_string](o)
        return cpp2py_string2str(cpp_text)
    except:
        pass

    # ModelDescription type
    cdef ModelDescription m
    cdef cpp_map[cpp_string, cpp_any].iterator m_end, m_it
    try:
        m = any_cast[ModelDescription](o)

        model_name = cpp2py_string2str(m.model_name)
        param_dict = {}

        m_end = m.param_map.end()
        m_it = m.param_map.begin()
        while m_it != m_end:
            key = cpp2py_string2str(dereference(m_it).first)
            value = cpp2py_any2py(dereference(m_it).second)
            param_dict[key] = value
            preincrement(m_it)

        return PyModelDescription(model_name, param_dict)
    except:
        pass

    # TimeScale enum
    cdef TimeScale t
    try:
        t = any_cast[TimeScale](o)
        return t
    except:
        pass

    return 'ERROR: Type unknown'


cdef cpp_map[cpp_string, cpp_any] py2cpp_dict2map(o):
    """
    Casts a Python dict to a C++ map type

    :param o: A Python dict object.
    :type o: dict
    :return: A C++ map object.
    :rtype: std::map<std::string,boost::any>
    """

    cdef cpp_map[cpp_string, cpp_any] cpp_param_dict
    cdef ModelDescription m
    cdef cpp_vector[int] vi
    cdef cpp_vector[float] vf

    for key, value in o.items():
        cpp_key = py2cpp_str2string(key)
        if issubclass(type(value), int):
            cpp_param_dict[cpp_key] = cpp_any(<int> value)
        elif issubclass(type(value), float):
            cpp_param_dict[cpp_key] = cpp_any(<float> value)
        elif type(value) is str:
            cpp_param_dict[cpp_key] = cpp_any(<cpp_string> py2cpp_str2string(value))
        elif type(value) is dict:
            cpp_param_dict[cpp_key] = cpp_any(py2cpp_dict2map(value))
        #elif type(value) is PyModelDescription:
        elif isinstance(value, PyModelDescription):
            m.model_name = py2cpp_str2string(value.model_name)
            m.param_map = py2cpp_dict2map(value.params_dict)
            cpp_param_dict[cpp_key] = cpp_any(m)
        elif issubclass(type(value), list) or issubclass(type(value), np.ndarray):
            # if len(value)==0: #We assume that the list contains something
            if issubclass(type(value[0]), int):
                vi = py2cpp_list2vectori(value)
                cpp_param_dict[cpp_key] = cpp_any(<cpp_vector[int]> vi)
            elif issubclass(type(value[0]), float):
                vf = py2cpp_list2vectorf(value)
                cpp_param_dict[cpp_key] = cpp_any(<cpp_vector[float]> vf)

        else:
            #TODO: More possible types?
            print('ERROR, type unknown: {} of type {}'.format(value, type(value)))

    return cpp_param_dict


cdef cpp_vector[int] py2cpp_list2vectori(o):
    """
    Casts a Python list of ints to a C++ vector of ints.

    :param o: A Python list of ints.
    :type o: list[int]
    :return: A C++ vector of ints.
    :rtype: std::vector<int>
    """
    cdef cpp_vector[int] v
    for i in o:
        v.push_back(i)
    return v


cdef cpp_vector[float] py2cpp_list2vectorf(o):
    """
    Casts a Python list of floats to a C++ vector of ints.

    :param o: A Python list of floats.
    :type o: list[float]
    :return: A C++ vector of floats.
    :rtype: std::vector<float>
    """
    cdef cpp_vector[float] v
    for i in o:
        v.push_back(i)
    return v


cdef cpp_vector[double] py2cpp_list2vectord(o):
    """
    Casts a Python list of floats to a C++ vector of doubles.

    :param o: A Python list of floats
    :type o: list[float]
    :return: A C++ vector of doubles
    :rtype: std::vector<double>
    """
    cdef cpp_vector[double] v
    for i in o:
        v.push_back(i)
    return v


cdef cpp_vector[long] py2cpp_list2vectorl(o):
    """
    Casts a Python list of ints to a C++ vector of longs.

    :param o: A Python list of ints
    :type o: list[int]
    :return: A C++ vector of longs
    :rtype: std::vector<long>
    """
    cdef cpp_vector[long] v
    for i in o:
        v.push_back(i)
    return v

cdef bytearray py2cpp_str2string(o):
    """
    Casts a Python str object to a C++ string.

    :param o: A Python str object.
    :type o: str
    :return: A C++ string.
    :rtype: std::string
    """
    cpp_name = bytearray()
    cpp_name.extend(map(ord, o))
    return cpp_name
