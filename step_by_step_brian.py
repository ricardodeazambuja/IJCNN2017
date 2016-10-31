
class step_by_step_brian_sim(object):
    '''
    Step-by-Step Brian Simulation (Nov/2014 - ricardo.deazambuja@plymouth.ac.uk)
    
    This class was created to make it easier to run a Brian simulation step-by-step, passing input spikes without 
    running out of memory or having to create the input spikes beforehand. It also makes the code more clear because
    you generate a separated function with your Brian simulation code. This function (here called simulation) receives:
    simulation(brian.defaultclock, brian)
    brian.defaultclock: Brian defaultclock to be used
    brian: the result of the command "import brian", so the user doesn't need to import Brian and have to use "brian.".
    And must return a tuple:
    (Input_layer, Output_layer, pop_objects, syn_objects, monitors_objects)
    Input_layer: is a SpikeGenerator
    Output_layer: the layer the user wants to output the spikes
    pop_objects: a list with all the NeuronGroups used
    syn_objects: a list with all the Synapses objects used
    monitors_objects: a list with all the Monitors or functions used with the @network_operation decorator
    
    At initialization the simulation step size (in ms) can be passed (default is 2).
    After the creation of the instance, calling the method "run_step(input_spike_index_list)" sends the spikes 
    to the simulation and simulates one step (according to the initialization).
    The method run_step returns returns a tuple:
    int(number_of_the_run), 
    float(current_simulation_time), 
    numpy.array(tuple(processed_received_spikes)),
    list(list(output_spikes)), 
    list(float(output_spikes_times))
    '''
    
    def __init__(self, simulation, init_step_size=2):
        print "Initializing the simulation..."
        self.step_size = init_step_size
        self._generator = self._return_generator(simulation)
        print "Initializing the simulation...Done"
        print "Call .run_step(input_spikes_list) to run one step of the simulation!"
    
    def run_step(self,input_spikes=None):
        '''
        Calls the generator .next and send methods and returns the spikes and times generated.
        '''
        self._generator.next() # Runs up to the first yield (where the generator waits for the .send method)
        ans = self._generator.send(input_spikes) # Sends the spikes and runs to the second yield 
                                                 #(where the generator returns the result of the simulation)
        return ans
    
    def _return_generator(self, simulation):
        '''
        Defines a simulation using a python generator.
        '''

        import brian
        import numpy

        print "Starting the simulation!"

        print "Reseting the Brian Simulation object...",
        brian.reinit() # This is only necessary when using the same enviroment over and over (like with iPython).
        print "Done!"

        clock_mult = self.step_size
        brian.defaultclock.dt = clock_mult*brian.ms
        
        print "Initial simulation time:", brian.defaultclock.t
        print "Simulation step:", brian.defaultclock.dt
        
        # Calls the user function with the Brian objects to be used in the simulation
        Input_layer, Output_layer, pop_objects, syn_objects, monitors_objects = simulation(brian.defaultclock, brian)

        output_spikes = []
        output_spikes_time = []

        # Every time spikes occur at the SpikeMonitor related to the output neuron group, this function is called
        def output_spikes_proc(spikes):
            if len(spikes):
                output_spikes.append(spikes.tolist()) # Saves the indexes of the neurons who generated spikes
                output_spikes_time.append(1000*float(brian.defaultclock.t)) # Converts and save the actual time in ms
                # The spike monitor and all this code could be replaced by the .get_spikes() method of neurongroups.
                # I need to check what is fastest way!

        OutputMonitor=brian.SpikeMonitor(Output_layer, record=False, function=output_spikes_proc)
        # Because it is not saving, the system is not going to run out of memory after a long simulation.
        
        net = brian.Network(pop_objects + syn_objects + monitors_objects + [OutputMonitor])

        r=0
        while True:
            spiketimes = yield # Receives the content from the Python generator method .send()          
            if spiketimes:
                spiketimes = [(i,brian.defaultclock.t) for i in spiketimes] # The spikes received are inserted as the last simulated time
                Input_layer.set_spiketimes(spiketimes)
            net.run(clock_mult*brian.ms) # I'm running one step each time this function is called
            r+=1
            yield (
                  r,
                  float(brian.defaultclock.t)*1000, 
                  numpy.array(Input_layer.get_spiketimes()).astype(dtype=numpy.float), # I'm doing this way to prove the spikes were received
                  output_spikes,
                  output_spikes_time 
                  )# After the .send method, the generator executes this line and stops here

            output_spikes=[] # Cleans the output_spikes list so only the last spikes generated are sent
            output_spikes_time=[] # Cleans the output_spikes list so only the last spikes generated are sent            
            
            # I'm using the .astype(numpy.float) because the arrays have Brian objects (units),
            # and I think using only floats the memory footprint can be smaller.