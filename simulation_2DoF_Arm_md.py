# These variables I'm expecting to receive from outsite the namespace of this function
# using the func_globals dictionary. It's kind of an ugly hack...
# So it's necessary to 'create' them, or an error message is generated.
input_gain=0
STP_OFF=0
noisy=0
stratification=0
gaussian_pop=0
w_SD=0
i_noiseA=0
offset_noise=0
initial_volt=0
random_reset=0
Net_shape=0
Number_of_input_layers=0
Number_of_neurons_inputs=0
lbd_value=0
noisy_currents=0
noisy_input_weights=0
noisy_input=0
input_noise_level=0
AII=0
AIE=0
AEI=0
AEE=0

killed_neurons=[] # indices of the liquid's neurons that should not spike

def simulation_2DoF_Arm(brian_clock, brian):
    """    
    #############################################################################
    The variables:
    my_seed: value (integer) used as the random seed to keep the liquid structure the same (or not)
    input_gain: the multiplier (float) used within the input layer
    noisy: 0=>always uses my_seed to start the random generator; 1=>uses the default random state every run
    gaussian_pop: 1=>gaussian distribution as the input layers gains; 0=>random weights (just like in Maass's papers)
                 -1=>gaussian distribution with an offset noise
    w_SD: standard deviation of the gaussian used to distribute the input weights
    stratification: 0=>no stratification; 1=>stratified inputs
    STP_OFF: True=>No STP; False=>STP
    Must be set from outside using the .func_globals dictionary. Example:
    simulation_2DoF_Arm.func_globals['my_seed']=93200
    #############################################################################
    """
    
    import numpy # I could use "brian." because Brian imports numpy, but I prefer not to.
    import time
    import sys

#
#     THE VARIABLES BELOW MUST BE SET OUTSIDE OR THE SIMULATION IS NOT GOING TO WORK!
#
#     Example:
#     simulation_2DoF_Arm.func_globals['my_seed']=93200         # This seed is important to always repeat the liquid's structure
#     simulation_2DoF_Arm.func_globals['input_gain']=70.0    # Base value used within the input weights
#     simulation_2DoF_Arm.func_globals['noisy']=1            # Controls if the liquid is going to use the noisy currents/input weights
#     simulation_2DoF_Arm.func_globals['gaussian_pop']=1         # Controls which type of input weight configuration is used (1=>gaussian distributed)
#     simulation_2DoF_Arm.func_globals['w_SD']=3.0           # The width of the gaussian used above.
#     simulation_2DoF_Arm.func_globals['stratification']=1   # Type of connections: stratified or 30% random
#     simulation_2DoF_Arm.func_globals['STP_OFF']=False      # Uses or not STP (False means it uses STP)

#     simulation_2DoF_Arm.func_globals['i_noiseA'] = 1.0             # amplitude (dimensionless) of the random currents (updated every time step).
#     simulation_2DoF_Arm.func_globals['offset_noise'] = (13.5,14.5) 
                              # Range of the uniform distribution used to generate the constant offset current.
                              # Maass 2002, should be drawn from a uniform distr [14.975nA,15.025nA]
                              # Joshi/Maass 2005 does [13.5nA,14.5nA]. Maybe it is to avoid too many spikes without inputs...

#     simulation_2DoF_Arm.func_globals['initial_volt'] = (13.5,14.9) 
                              # Range of the uniform distribution used to generate the initial membrane voltage.
                              # Maass 2002, should be drawn from a uniform distr [13.5mV,15.0mV]
                              # Joshi/Maass 2005 does [13.5mV,14.9mV], this way it's impossible to generate spikes (14.9mV is below threshold)

#     simulation_2DoF_Arm.func_globals['random_reset'] = (13.8,14.5)    # Range of the uniform distribution used to generate the constant reset voltages.
#     simulation_2DoF_Arm.func_globals['Net_shape'] = (20,5,6)          # Defines the shape of the network (liquid state machine)
#     simulation_2DoF_Arm.func_globals['Number_of_input_layers'] = 6    # It means I will have 6 input layers...
#     simulation_2DoF_Arm.func_globals['Number_of_neurons_inputs'] = 50 # ...with 50 neurons each.
#     simulation_2DoF_Arm.func_globals['lbd_value'] = 1.2               # lbd controls the connection probability

#     simulation_2DoF_Arm.func_globals['noisy_currents'] = True # Injects noisy currents the neurons
#     simulation_2DoF_Arm.func_globals['noisy_input_weights'] = True # Injects noise into the input weights
#     simulation_2DoF_Arm.func_globals['noisy_input'] = True # Ignores the random seed for the input noise
#     simulation_2DoF_Arm.func_globals['input_noise_level'] = 0.1 # 1/SNR

      # STP parameters
#     simulation_2DoF_Arm.func_globals['AII']=47.0  # Joshi/Maass2006=>47.0 / Maass2002=>2.8
#     simulation_2DoF_Arm.func_globals['AIE']=47.0  # Joshi/Maass2006=>47.0 / Maass2002=>3.0
#     simulation_2DoF_Arm.func_globals['AEI']=150.0 # Joshi/Maass2006=>150.0 / Maass2002=>1.6
#     simulation_2DoF_Arm.func_globals['AEE']=70.0  # Joshi/Maass2006=>70.0 / Maass2002=>1.2

    Number_of_neurons_lsm=Net_shape[0]*Net_shape[1]*Net_shape[2] # Just to avoid calculating this everytime it's used

    
    # These lines make easier to use the Brian objects without the "brian." at the beginning
    ms = brian.ms
    mV = brian.mV
    nA = brian.nA
    nF = brian.nF
    NeuronGroup = brian.NeuronGroup
    SpikeGeneratorGroup = brian.SpikeGeneratorGroup
    Synapses = brian.Synapses
    SpikeMonitor = brian.SpikeMonitor
    network_operation = brian.network_operation 
    defaultclock = brian.defaultclock

    if my_seed!=0:
        # This seed can guarantee that the liquid is going to have the same structure because this object 
        # is passed to the lsm_connections_probability module.
        liq_rs = numpy.random.RandomState(my_seed) # this is the random state used within the liquid structure generation
    else:
        liq_rs = numpy.random.RandomState() #random liquid
    
    import lsm_connections_probability # Creates the 3D Grid and the connections according to Maass 2002
    reload(sys.modules['lsm_connections_probability']) # Makes sure the interpreter is going to reload the module
    lm = lsm_connections_probability

    import lsm_dynamical_synapses_v1 # Creates the dynamical synapses according to Maass 2002 and using the output 
                                     # from the lsm_connections_probability    
    reload(sys.modules['lsm_dynamical_synapses_v1']) # Makes sure the interpreter is going to reload the module
    ls = lsm_dynamical_synapses_v1

    
    # Here I'm creating the random state for the rest of the simulation
    # This way I can have a fixed liquid structure, but with all the other noisy sources.
    if noisy or (my_seed==0):
        sim_rs = numpy.random.RandomState()
    else:
        sim_rs = numpy.random.RandomState(my_seed)

    if noisy_input or (my_seed==0):
        input_rs = numpy.random.RandomState()
    else:
        input_rs = numpy.random.RandomState(my_seed)
        
    initial_time = time.time()

    print "Initial time:",initial_time

    print "#"*78
    print "#"*78
    print "Liquid State Machine - 2 DoF arm experiments!"
    print "#"*78
    print "#"*78


    defaultclock = brian_clock  # Receives the clock from the step-by-step simulator
                                # I'm setting it to the defaultclock because all connected NeuronGroups 
                                # must use the same clock.


    lsm_3dGrid_flat = numpy.zeros(Number_of_neurons_lsm) 
    # This creates a numpy 1D array with 'Number_of_neurons_lsm' positions
    # I'm using a numpy array to be able to use the reshape method to 
    # change from 1D (vector) to 3D (matrix)


    def randon_connections_gen(Number_of_neurons, ratio, number=None):
        '''
        Generate the random neuron indexes list according to the number of neurons and the ratio
        '''
        # List used to generate the randoms 'ratio' indexes for the Liquid
        l_range = range(Number_of_neurons)

        # Generates the random "indexes" of the flattened version of the 3DGrid.
        # At each iteration one random item is extracted from l_range and inserted in connection_index.
        # This is the way I've found to sample without repetitions.
        # - Another way is using the shuffle from numpy and then grabbing the first N values!      
        if number==None:
            connection_index = [l_range.pop(liq_rs.randint(0,len(l_range))) for i in range(int(Number_of_neurons*ratio))] 
        else:
            connection_index = [l_range.pop(liq_rs.randint(0,len(l_range))) for i in range(int(number))] 
        connection_index.sort() #This is only useful to make easier to human beings to read the list :)

        return connection_index


    #
    # Number of Inhibitory and Excitatory neurons - LIQUID - 20% of the total neurons
    inhibitory_index_L = randon_connections_gen(Number_of_neurons_lsm, 0.2)


    # This is the dictionary that has all the connections parameters according to Maass 2002.
    # It is necessary to create the 3D connections and the STP configuration matrices
    # E=>1 (excitatory) and I=>0 (inhibitory)
    # Ex.: (0,0) => II
    # Dynamical Synapses Parameters (STP):
    Connections_Parameters={
                  (0,0):[ # II
                          0.1,       # CGupta=0.1        # Parameter used at the connection probability - from Maass2002 paper
                          0.32,      # UMarkram=0.32     # Use (U) - Parameter used at the Dynamic Synapse - from Maass2002 paper
                          0.144,     # DMarkram=0.144    # Time constant for Depression (tau_rec) - used at the Dynamic Synapse - from Maass2002 paper                    
                          0.06,      # FMarkram=0.06     # Time constant for Facilitation (tau_facil) - used at the Dynamic Synapse - from Maass2002 paper
                          AII,       # AMaass=2.8        # (nA) In the Maass2002 paper the value is negative, but because I need a positive scale (random.normal parameter) and there is a negative sign in front of the abs function I changed this to positive
                          0.8        # Delay_trans = 0.8 # In Maass paper the transmission delay is 0.8 to II, IE and EI        
                      ],
                  (0,1):[ # IE
                          0.4,    # CGupta=0.4
                          0.25,   # UMarkram=0.25
                          0.7,    # DMarkram=0.7
                          0.02,   # FMarkram=0.02
                          AIE,    # AMaass=3.0 #in the Maass2002 paper the value is negative, but because I need a positive scale (random.normal parameter) and there is a negative sign in front of the abs function I changed this to positive
                          0.8     # Delay_trans = 0.8 # in Maass paper the transmission delay is 0.8 to II, IE and EI
                      ],
                  (1,0):[ # EI
                          0.2,    # CGupta=0.2
                          0.05,   # UMarkram=0.05
                          0.125,  # DMarkram=0.125
                          1.2,    # FMarkram=1.2
                          AEI,    # AMaass=1.6
                          0.8     # Delay_trans = 0.8 # in Maass paper the transmission delay is 0.8 to II, IE and EI
                      ],
                  (1,1):[ # EE
                          0.3,    # CGupta=0.3 
                          0.5,    # UMarkram=0.5
                          1.1,    # DMarkram=1.1
                          0.05,   # FMarkram=0.05
                          AEE,    # AMaass=1.2 #scaling parameter or absolute synaptic efficacy or weight - from Maass2002 paper
                          1.5     # Delay_trans = 1.5 # in Maass paper the transmission delay is 1.5 to EE connection
                      ]
                  }


    # Utilizes the functions in the lsm_connections_probability.py
    # =>output = {'exc':connections_list_exc,'inh':connections_list_inh, '3Dplot_a':positions_list_a, '3Dplot_b':positions_list_b}
    # connections_list_exc= OR connections_list_inh=
        # ((i,j), # PRE and POS synaptic neuron indexes
        # pconnection, # probability of the connection
        # (W_n, U_ds, D_ds, F_ds), # parameters according to Maass2002
        # Delay_trans, # parameters according to Maass2002
        # connection_type)

    # Generate the connections matrix inside the Liquid (Liquid->Liquid) - according to Maass2002
    #
    print "Liquid->Liquid connections..."

    print "Generating the Liquid->Liquid connections..."
    output_L_L = lm.generate_connections(lsm_3dGrid_flat, inhibitory_index_L, Net_shape, 
                                      CParameters=Connections_Parameters, lbd=lbd_value, randomstate=liq_rs) 
                                    # lbd controls the connections
                                    # randomstate receives a numpy.random.RandomState object, or automatically sets a random one.
                                                                    
    

    print "Liquid->Liquid connections...Done!"

    
    #
    # These are the cell (neuron) parameters according to Maass 2002
    #
    cell_params_lsm = {'cm'          : 30.0,    # [nF]  Capacitance of the membrane 
                                                # =>>>> MAASS PAPER DOESN'T MENTION THIS PARAMETER DIRECTLY
                                                #       but the paper mention a INPUT RESISTANCE OF 1MEGA Ohms and tau_m=RC=30ms, so cm=30nF
                       'tau_m'       : 30.0,    # [ms] Membrane time constant => Maass2002
                       'tau_refrac_E': 3.0,     # [ms] Duration of refractory period - 3mS for EXCITATORY => Maass2002
                       'tau_refrac_I': 2.0,     # [ms] Duration of refractory period - 2mS for INHIBITORY => Maass2002
                       'tau_syn_E'   : 3.0,     # [ms] Decay time of excitatory synaptic current => Maass2002
                       'tau_syn_I'   : 6.0,     # [ms] Decay time of inhibitory synaptic current => Maass 2002
                       'v_reset'     : 13.5,    # [mV] Reset potential after a spike => Maass 2002
                       'v_rest'      : 0.0,     # [mV] Resting membrane potential => Maass 2002
                       'v_thresh'    : 15.0,    # [mV] Spike threshold => Maass 2002
                       'i_noise'     : i_noiseA # [nA] Used in Joshi/Maass 2005: mean 0 and SD=1nA; Maass 2002=>SD=0.2nA
                    }

    offset_cur = (offset_noise[0],offset_noise[1])  # Values for the uniform distribution of constant offset currents
    init_volt = (initial_volt[0],initial_volt[1])   # Values for the uniform distribution of initial membrane voltages
    rand_res = (random_reset[0],random_reset[1])    # Values for the uniform distribution of constant random resets


    # IF_curr_exp - MODEL EXPLAINED
    # Leaky integrate and fire model with fixed threshold and
    # decaying-exponential post-synaptic current. 
    # (Separate synaptic currents for excitatory and inhibitory synapses)
    lsm_neuron_eqs='''
      dv/dt  = (ie + ii + i_offset + i_noise)/c_m + (v_rest-v)/tau_m : mV
      die/dt = -ie/tau_syn_E                : nA
      dii/dt = -ii/tau_syn_I                : nA
      tau_syn_E                             : ms
      tau_syn_I                             : ms
      tau_m                                 : ms
      c_m                                   : nF
      v_rest                                : mV
      i_offset                              : nA
      i_noise                               : nA
      '''



    ########################################################################################################################
    #
    # LIQUID - Setup
    #
    print "LIQUID - Setup..."

    # Creates a vector with the corresponding refractory period according to the type of neuron (inhibitory or excitatory)
    # IT MUST BE A NUMPY ARRAY OR BRIAN GIVES CRAZY ERRORS!!!!!
    refractory_vector = [ cell_params_lsm['tau_refrac_E'] ]*Number_of_neurons_lsm # fills the list with the value corresponding to excitatory neurons
    for i in range(Number_of_neurons_lsm):
        if i in inhibitory_index_L:
            refractory_vector[i]=cell_params_lsm['tau_refrac_I'] # only if the neuron is inibitory, changes the refractory period value!
    refractory_vector=numpy.array(refractory_vector)*ms # Here it is converted to a NUMPY ARRAY
    

    
    # This is the population (neurons) used exclusively to the Liquid (pop_lsm).
    pop_lsm = NeuronGroup(Number_of_neurons_lsm, model=lsm_neuron_eqs, 
                                                 threshold=cell_params_lsm['v_thresh']*mV, 
                                                 reset='v=sim_rs.uniform(rand_res[0],rand_res[1])*mV', 
                                                 refractory=refractory_vector, 
                                                 max_refractory=max(cell_params_lsm['tau_refrac_E']*ms, 
                                                                    cell_params_lsm['tau_refrac_I']*ms))
#     else:
#         print "Noisy resets OFF!"
#         # This is the population (neurons) used exclusively to the Liquid (pop_lsm).
#         pop_lsm = NeuronGroup(Number_of_neurons_lsm, model=lsm_neuron_eqs, 
#                                                      threshold=cell_params_lsm['v_thresh']*mV, 
#                                                      reset=cell_params_lsm['v_reset']*mV,
#                                                      refractory=refractory_vector, 
#                                                      max_refractory=max(cell_params_lsm['tau_refrac_E']*ms, 
#                                                                         cell_params_lsm['tau_refrac_I']*ms))


    # Here I'm mixing numpy.fill with the access of the state variable "c_m" in Brian (because Brian is using a numpy.array)
    # Sets the value of the capacitance according to the cell_params_lsm (same value to all the neurons)
    pop_lsm.c_m.fill(cell_params_lsm['cm']*nF)


    # Sets the value of the time constant RC (or membrane constant) according to the cell_params_lsm (same value to all the neurons)
    pop_lsm.tau_m.fill(cell_params_lsm['tau_m']*ms)

    # Sets the fixed i_offset current.
    # The i_offset current is random, but never changes during the same simulation.
    # This current, according to Maass 2002, should be drawn from a uniform distr [14.975nA,15.025nA]
    # Joshi/Maass 2005 does [13.5nA,14.5nA]. Maybe is to avoid too many spikes without inputs...
    pop_lsm.i_offset=sim_rs.uniform(offset_cur[0],offset_cur[1], Number_of_neurons_lsm)*nA

    pop_lsm.tau_syn_E.fill(cell_params_lsm['tau_syn_E']*ms) # (same value to all the neurons)
    pop_lsm.tau_syn_I.fill(cell_params_lsm['tau_syn_I']*ms) # (same value to all the neurons)

    pop_lsm.v_rest.fill(cell_params_lsm['v_rest']*mV) # (same value to all the neurons)

    # This current changes (randomly) at each time step (here is only the initialization).
    pop_lsm.i_noise=sim_rs.normal(loc=0, scale=cell_params_lsm['i_noise'],size=Number_of_neurons_lsm)*nA

    
    # Sets the initial membrane voltage. Doesn't change during the simulation.
    # According to Maass 2002, this voltage should be drawn from a uniform distr [13.5mV,15.0mV]
    # Joshi/Maass 2005 does [13.5mV,14.9mV], this way it's impossible to generate spikes (14.9mV is below threshold)
    pop_lsm.v=sim_rs.uniform(init_volt[0],init_volt[1], Number_of_neurons_lsm)*mV

    #Kills neurons by setting their initial voltage to a crazily LOW value
    pop_lsm.v[killed_neurons]=-1E12*mV
    
    #
    # Loading or creating the Synapses objects used within the Liquid
    print "Liquid->Liquid connections..."

    syn_lsm_obj = ls.LsmConnections(pop_lsm, pop_lsm, output_L_L, nostp=STP_OFF)

    # Generates the Liquid->Liquid - EXCITATORY synapses
    syn_lsm_exc = syn_lsm_obj.create_synapses('exc')

    # Generates the Liquid->Liquid - INHIBITORY synapses
    syn_lsm_inh = syn_lsm_obj.create_synapses('inh')
    

    print "Liquid->Liquid connections...Done!"

    total_number_of_connections_liquid = len(syn_lsm_exc) + len(syn_lsm_inh)

    print "Number of excitatory synapses in the Liquid: " + str(len(syn_lsm_exc)) # DEBUG to verify if it is working
    print "Number of inhibitory synapses in the Liquid: " + str(len(syn_lsm_inh)) # DEBUG to verify if it is working

    # To understand what is being returned:
    # pop_lsm: it is necessary to connect the neuron network with the rest of the world
    # [syn_lsm_obj, syn_lsm_exc, syn_lsm_inh]: to include these objects at the simulation (net=Net(...); net.run(total_sim_time*ms)); 
    # It is a list because is easy to concatenate lists :D

    print "LIQUID - Setup...Done!"

    #
    # End of the LIQUID - Setup
    ########################################################################################################################



    ########################################################################################################################
    #
    # INPUT - Setup
    #
    print "INPUT - Setup..."
  
    spiketimes = [] # The spikes are going to be received during the simulation, 
                    # so this is always an empty list when using the step_by_step_brian_sim!
    
    # I'm using only one big input layer because Brian docs say it is better for the performance
    SpikeInputs = SpikeGeneratorGroup(Number_of_input_layers*Number_of_neurons_inputs, spiketimes)
    

    #
    #
    # Here the synapses are created. The synapses created are ALWAYS excitatory because it is 
    # connecting through 'ie' in the neuron model!

    syn_world_Input = Synapses(SpikeInputs, pop_lsm,
                                         model='''w : 1''',
                                         pre='''ie+=w''')


    weights_input_liquid = [] # remember that the weights must follow the same order of the creation of synapses


    def gaussian(lamb,n,nt):
        '''
        Generates a gaussian centered at 'n'
        '''
        # return input_gain*(1/(lamb*numpy.sqrt(2*numpy.pi)))*numpy.exp(-((nt-n)**2)/(2*(lamb)**2))  # Energy normalized version
        return input_gain*numpy.exp(-((nt-n)**2)/(2*(lamb)**2))  # No energy normalization
  
    def simple_inputs(*args):
        '''
        This function ignores the arguments and return a random value from a gaussian distribution with mean input_gain and SD=input_gain/2.0
        '''
        return (abs(sim_rs.normal(loc=input_gain, scale=input_gain/2.0))) # The "abs" function is to guarantee all inputs are excitatories!    

    def gaussian_noise(lamb,n,nt):
        '''
        Generates a gaussian centered at 'n' with a background noise (1/3 of the amplitude)
        '''
        return 3*input_gain*numpy.exp(-((nt-n)**2)/(2*(lamb)**2)) + (abs(sim_rs.normal(loc=input_gain, scale=input_gain/2.0)))


    # Verifies the user selection and sets the proper weight generation function.
    if gaussian_pop==1:
        weight_func = gaussian
    elif gaussian_pop==0:
        weight_func = simple_inputs
    else:
        weight_func = gaussian_noise
        


    if stratification==0:
        # List with the indexes of all the excitatory neurons in the liquid
        excitatory_index_L = [i for i in range(Number_of_neurons_lsm) if i not in inhibitory_index_L]

        # Here I connect the input neurons only to excitatories neurons in the liquid:
        sim_rs.shuffle(excitatory_index_L) # Shuffles the excitatory index vector.
        rand_connections = excitatory_index_L[:int(len(excitatory_index_L)*0.3)] # Gets randomly 30% of the excitatory connections

        # Here I connect the input neurons to any type of neurons in the liquid:
        # rand_connections = randon_connections_gen(Number_of_neurons_lsm, 0, number=int(Number_of_neurons_lsm*0.3)) # generates random connections to 30% of the neurons in the liquid!

        # Goes through the liquid to generate the proper connections
        for inp in range(Number_of_input_layers):
            for i in range(inp*Number_of_neurons_inputs,Number_of_neurons_inputs*(inp+1)):
                for j,ji in zip(rand_connections,range(len(rand_connections))):
                    syn_world_Input[i,j] = True # So it is one-to-random NoIN neurons, each input neuron is connect to all the "input layer" of the liquid.
                    # All inputs have the same connections to the Liquid (I would say that they are all connect to the input layer of the liquid)
                    # If they are all connected to the same neurons, seems to me that is less probable that the readout is going to learn
                    # only to filter the input...
                    centre_position=(i-(inp*Number_of_neurons_inputs))*(len(rand_connections)-1)/float(Number_of_neurons_inputs)
                    weights_input_liquid.append(weight_func(w_SD,centre_position,ji)*nA)
    else:

        # Goes through the liquid to generate the proper connections, but dividing the liquid into the same number of input layers
        liquid_input_layer_size = int(Number_of_neurons_lsm/float(Number_of_input_layers))
        for inp in range(Number_of_input_layers):
            for i in range(inp*Number_of_neurons_inputs,Number_of_neurons_inputs*(inp+1)):
                for j,ji in zip(range(inp*liquid_input_layer_size,liquid_input_layer_size*(inp+1)),range(liquid_input_layer_size)):
                    if j not in inhibitory_index_L:
                        syn_world_Input[i,j] = True
                        centre_position=(i-(inp*Number_of_neurons_inputs))*(liquid_input_layer_size-1)/float(Number_of_neurons_inputs)
                        weights_input_liquid.append(weight_func(w_SD,centre_position,ji)*nA)
                        



  
    weights_input_liquid = numpy.array(weights_input_liquid)
    
    
    syn_world_Input.w = weights_input_liquid
    syn_world_Input.delay=0*ms

    print "INPUT - Setup...Done!"

    #
    # End of the INPUT - Setup (creation of the connections between the Poisson input and the Liquid!)
    #
    ########################################################################################################################


    # Generates the noisy current at each time step (as seen in Joshi/Maass 2005)
    @network_operation(clock=defaultclock)
    def generate_i_noise():
        if noisy_currents:
            # These are the noise currents inside each liquid's neuron
            pop_lsm.i_noise=cell_params_lsm['i_noise']*sim_rs.normal(loc=0,scale=1,size=Number_of_neurons_lsm)*nA
        
        if noisy_input_weights:
            # This is my new version where the noise level can be controlled
            syn_world_Input.w[:]=weights_input_liquid+((input_noise_level*input_gain)*input_rs.normal(loc=0,scale=1,size=len(weights_input_liquid))*nA)        


    populations_sim = [pop_lsm, SpikeInputs]

    synapses_sim = [syn_lsm_exc, syn_lsm_inh, syn_world_Input]

    monitors_sim = [generate_i_noise] 

    Input_layer, Output_layer, pop_objects, syn_objects, monitors_objects = SpikeInputs, pop_lsm, populations_sim, synapses_sim, monitors_sim

    print "Setup time:", time.time()-initial_time
    
    # Returns these objects to be used with the step_by_step_brian_sim class
    return Input_layer, Output_layer, pop_objects, syn_objects, monitors_objects