def generates_lsm_start(tji,trajectories,x_values,y_values):
    xstart,ystart = trajectories[tji-1][0]
    xdest,ydest = trajectories[tji-1][1]


    # The original system, proposed by Joshi/Maass 2006 starts without any information about the current
    # position because the proprioceptive feedback comes only after a time delay.

    # Now I will add some noise
    # noise_sd = 0.1
    # xstart,ystart = xstart+numpy.random.normal(loc=0,scale=noise_sd),ystart+numpy.random.normal(loc=0,scale=noise_sd)
    # xdest,ydest = xdest+numpy.random.normal(loc=0,scale=noise_sd),ydest+numpy.random.normal(loc=0,scale=noise_sd)

    # print "Original:", trajectories[tji-1][0],trajectories[tji-1][1]
    # print "Noisy:",[xstart,ystart],[xdest,ydest]


    # Indexes of the normalized initial position values to use in the LSM simulation
    xstart_idx =  abs(x_values-xstart).argmin()
    ystart_idx =  abs(y_values-ystart).argmin()

    # Indexes of the normalized final position values to use in the LSM simulation
    xdest_idx =  abs(x_values-xdest).argmin()
    ydest_idx =  abs(y_values-ydest).argmin()

    (xstart_idx,ystart_idx),(xdest_idx,ydest_idx)


    # The input (one big neurongroup with 300 neurons) will be divided like this:
    # 6 groups of 50 neurons.
    # - Group 1: xdest => offset:0
    # - Group 2: ydest => offset:50
    # - Group 3: teta1 => offset:100
    # - Group 4: teta2 => offset:150
    # - Group 5: tau1  => offset:200
    # - Group 6: tau2  => offset:250

    ############# Using ONLY the initial (start) position
    # x_lsm = xstart_idx + 0
    # y_lsm = ystart_idx + 50

    ############# Using ONLY the final (dest) position
    x_lsm = xdest_idx + 0
    y_lsm = ydest_idx + 50
    return x_lsm,y_lsm,xstart,ystart,xdest,ydest