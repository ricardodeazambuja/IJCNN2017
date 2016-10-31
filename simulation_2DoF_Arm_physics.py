
# 2 DOF Simulator module
# Strongly based on http://www.gribblelab.org/compneuro/index.html

import numpy

# forward kinematics
def joints_to_hand(A,aparams):
    """
    Given joint angles A=(a1,a2) and anthropometric params aparams,
    returns hand position H=(hx,hy) and elbow position E=(ex,ey)
    Note1: A must be type matrix (or array([[a1,a2],...]))
    Note2: If A has multiple lines, H and E will have the same number of lines.
    """
    l1 = aparams['l1']
    l2 = aparams['l2']
    n = numpy.shape(A)[0]
    E = numpy.zeros((n,2))
    H = numpy.zeros((n,2))
    for i in range(n):
        E[i,0] = l1 * numpy.cos(A[i,0])
        E[i,1] = l1 * numpy.sin(A[i,0])
        H[i,0] = E[i,0] + (l2 * numpy.cos(A[i,0]+A[i,1]))
        H[i,1] = E[i,1] + (l2 * numpy.sin(A[i,0]+A[i,1]))
    return H,E


# I could do the inverse kinematics using all the possible values (workspace) of the arm and creating a numpy array. Then I 
# could use the argmin trick to find the value.
# In order to solve the problem when multiple solutions appear I could use the minimum jerk criterion. The user should enter
# the actual position and the next one, then the system solves according to the one that uses mininum energy.
# One curious thing about inverse kinematics is that as human beings we cannot do a inverse kinematic of our hand position
# without taking in account the actual position. The function, below, doesn't care about the actual position, and that is why 
# more than one solution appears.
# So, I don't think the brain solves the problem of multiple solutions. Who solves this problem is the morphology of the limbs.
# It is impossible to change trajectories instantaneously, therefore the continuity of the movements is guaranteed.
# Summary: there are no positions, but trajectories :)

# inverse kinematics
def hand_to_joints(H,aparams):
    """
    Given hand position H=(hx,hy) and anthropometric params aparams,
    returns joint angles A=(a1,a2)
    Note1: H must be type matrix (or array([[hx,hy],...]))
    Note2: If H has multiple lines, A will have the same number of lines.
    """
    l1 = aparams['l1']
    l2 = aparams['l2']
    n = numpy.shape(H)[0]
    A = numpy.zeros((n,2))
    for i in range(n):
        A[i,1] = numpy.arccos(((H[i,0]*H[i,0])+(H[i,1]*H[i,1])-(l1*l1)-(l2*l2))/(2.0*l1*l2))
        A[i,0] = numpy.arctan2(H[i,1],H[i,0]) - numpy.arctan2((l2*numpy.sin(A[i,1])),(l1+(l2*numpy.cos(A[i,1]))))
#         if A[i,0] < 0:
#             print "<0:",A[i,0]
#             A[i,0] = A[i,0] + pi
#         elif A[i,0] > pi:
#             print ">0:",A[i,0]
#             A[i,0] = A[i,0] - pi
    return A


# inverse kinematics
def hand_to_joints(H,aparams,ang_error=0.01):
    """
    Given hand position H=(hx,hy) and anthropometric params aparams,
    returns joint angles A=(a1,a2)
    Note1: H must be type matrix (or array([[hx,hy],...]))
    Note2: If H has multiple lines, A will have the same number of lines.
    """
    l1 = aparams['l1']
    l2 = aparams['l2']
    n = numpy.shape(H)[0]
    A = numpy.zeros((n,2))
    t_bias=[0,0] 
    for i in range(n):
        A[i,1] = numpy.arccos(((H[i,0]*H[i,0])+(H[i,1]*H[i,1])-(l1*l1)-(l2*l2))/(2.0*l1*l2)) + t_bias[1]
        A[i,0] = numpy.arctan2(H[i,1],H[i,0]) - numpy.arctan2((l2*numpy.sin(A[i,1])),(l1+(l2*numpy.cos(A[i,1])))) + t_bias[0]
        if i>0:
            # Here I'm trying to avoid descontinuity problems when there's a 2pi difference between them!
            if 0<=abs(abs((A[i,1]-A[i-1,1])/numpy.pi)-2)<=ang_error:
                print "Correction on Joint 2:",(A[i,1],A[i-1,1])
                if (A[i,1]-A[i-1,1])>0:
                    A[i,1]-=2*numpy.pi
                    t_bias[1]-=2*numpy.pi
                else:
                    A[i,1]+=2*numpy.pi
                    t_bias[1]+=2*numpy.pi  
                    
            if 0<=abs(abs((A[i,0]-A[i-1,0])/numpy.pi)-2)<=ang_error:
                print "Correction on Joint 1:",(A[i,0],A[i-1,0])
                if (A[i,0]-A[i-1,0])>0:
                    A[i,0]-=2*numpy.pi
                    t_bias[0]-=2*numpy.pi
                else:
                    A[i,0]+=2*numpy.pi
                    t_bias[0]+=2*numpy.pi
    return A


# Generates the movements according to:
# Flash, Tamar and Neville Hogan. 1985. The Coordination of Arm Movements: An Experimentally Confirmed Mathematical Model. The Journal of Neuroscience 5 (7): 1688-1703
def cartesian_movement_generation_training(xstart,ystart,xdest,ydest,MT,t):
    '''
    xstart,ystart: initial position of the trajectory
    xdest,ydest: final position of the trajectory
    MT: total time spent doing the trajectory
    t: current time
    
    returns a matrix: [[x0,y0],[x1,y1],...]
    '''
    x_t=xstart+(xstart-xdest)*(15*(t/MT)**4-6*(t/MT)**5-10*(t/MT)**3)
    y_t=ystart+(ystart-ydest)*(15*(t/MT)**4-6*(t/MT)**5-10*(t/MT)**3)    
    return numpy.array([x_t,y_t]).T


# Used to generate the velocities and the accelerations using the position and time vectors
def derivator(v,t):
    return numpy.array([(v[i+1]-v[i])/(t[i+1]-t[i]) for i in range(len(t)-1)])


def twojointarm_torques(state, t, aparams):
    """
    Calculates the necessaries torques to generate the accelerations
    """
    import numpy

    a1,a2,a1d,a2d,a1dd,a2dd = state # joint_angle_a1,joint_angle_a2,joint_vel_a1,joint_vel_a2,joint_acc_a1,joint_acc_a2

    l1,l2 = aparams['l1'], aparams['l2'] # lenght link 1 and 2
    m1,m2 = aparams['m1'], aparams['m2'] # mass link 1 and 2
    i1,i2 = aparams['i1'], aparams['i2'] # moment of inertia link 1 and 2
    lc1,lc2 = aparams['lc1'], aparams['lc2'] # distance to the center of mass of link 1 and 2

    M11 = i1 + i2 + (m1*lc1*lc1) + (m2*((l1*l1) + (lc2*lc2) + (2*l1*lc2*numpy.cos(a2))))
    M12 = i2 + (m2*((lc2*lc2) + (l1*lc2*numpy.cos(a2))))
    M21 = M12
    M22 = i2 + (m2*lc2*lc2)
    M = numpy.matrix([[M11,M12],[M21,M22]]) # H matrix

    C1 = -(m2*l1*a2d*a2d*lc2*numpy.sin(a2)) - (2*m2*l1*a1d*a2d*lc2*numpy.sin(a2))
    C2 = m2*l1*a1d*a1d*lc2*numpy.sin(a2)
    C = numpy.matrix([[C1],[C2]])

    ACC = numpy.array([[a1dd],[a2dd]])

    T = M*ACC + C

    return numpy.array([T[0,0],T[1,0]])


# forward dynamics equations of our two-joint arm
def twojointarm(state, t, aparams, torque):
    import numpy
    
    """
    two-joint arm in plane
    X is fwd(+) and back(-)
    Y is up(+) and down(-)
    shoulder angle a1 relative to Y vert, +ve counter-clockwise
    elbow angle a2 relative to upper arm, +ve counter-clockwise
    """
    a1,a2,a1d,a2d = state # joint_angle_a1, joint_angle_a2, joint_velocity_a1, joint_velocity_a2

    l1,l2 = aparams['l1'], aparams['l2'] # lenght link 1 and 2
    m1,m2 = aparams['m1'], aparams['m2'] # mass link 1 and 2
    i1,i2 = aparams['i1'], aparams['i2'] # moment of inertia link 1 and 2
    lc1,lc2 = aparams['lc1'], aparams['lc2'] # distance to the center of mass of link 1 and 2

    M11 = i1 + i2 + (m1*lc1*lc1) + (m2*((l1*l1) + (lc2*lc2) + (2*l1*lc2*numpy.cos(a2))))
    M12 = i2 + (m2*((lc2*lc2) + (l1*lc2*numpy.cos(a2))))
    M21 = M12
    M22 = i2 + (m2*lc2*lc2)
    M = numpy.matrix([[M11,M12],[M21,M22]]) # H matrix

    C1 = -(m2*l1*a2d*a2d*lc2*numpy.sin(a2)) - (2*m2*l1*a1d*a2d*lc2*numpy.sin(a2))
    C2 = m2*l1*a1d*a1d*lc2*numpy.sin(a2)
    C = numpy.matrix([[C1],[C2]])

    T = numpy.matrix([[torque[0]],[torque[1]]])

    ACC = numpy.linalg.inv(M) * (T-C) # calculates the accelerations of joints 1 and 2

    a1dd,a2dd = ACC[0,0], ACC[1,0]

    return [a1d, a2d, a1dd, a2dd] # It returns the first and second derivatives of the joints


def animatearm(state,t,aparams):
    """
    animate the twojointarm
    """
    import matplotlib.pyplot as plt
    import numpy
    import time

    A = state[:,[0,1]] # Gets the angles a1 and a2 from the states matrix
    A[:,0] = A[:,0]
    H,E = joints_to_hand(A,aparams)
    l1,l2 = aparams['l1'], aparams['l2']
    plt.figure()
    plt.plot(0,0,'b.')
    plt.plot(H[:,0],H[:,1],'g.-');
    p1, = plt.plot(E[0,0],E[0,1],'b.')
    p2, = plt.plot(H[0,0],H[0,1],'b.')
    p3, = plt.plot((0,E[0,0],H[0,0]),(0,E[0,1],H[0,1]),'b-')
    plt.xlim([-l1-l2, l1+l2])
    plt.ylim([-l1-l2, l1+l2])
    dt = t[1]-t[0]
    tt = plt.title("Click on this plot to continue...")
    plt.ginput(1)
    for i in xrange(0,numpy.shape(state)[0]):
        time.sleep(0.05)
        p1.set_xdata((E[i,0]))
        p1.set_ydata((E[i,1]))
        p2.set_xdata((H[i,0]))
        p2.set_ydata((H[i,1]))
        p3.set_xdata((0,E[i,0],H[i,0]))
        p3.set_ydata((0,E[i,1],H[i,1]))
        tt.set_text("Current time:%4.2f sec - click to next slide!" % (i*dt))
        plt.draw()
    tt.set_text("Current time:%4.2f sec - finished!" % ((numpy.shape(state)[0]-1)*dt))
    plt.draw()        

    
def animatearm_JS(state,t,aparams):
    """
    animate the twojointarm
    """
    import matplotlib.pyplot as plt
    import numpy
    from JSAnimation import IPython_display
    from matplotlib import animation


    A = state[:,[0,1]] # Gets the angles a1 and a2 from the states matrix
    A[:,0] = A[:,0]
    H,E = joints_to_hand(A,aparams)
    l1,l2 = aparams['l1'], aparams['l2']

    # Set up the axes, making sure the axis ratio is equal
#     ax = fig.add_axes([0, 0, 1, 1], xlim=(-0.02, 13.02), ylim=(-0.02, 5.02),
#                       xticks=range(14), yticks=range(6), aspect='equal', frameon=False)

    fig = plt.figure(figsize=(6, 6),dpi=100)
    ax = plt.axes(xlim=(-1, 1), ylim=(-1, 1), aspect='equal')
    ax.plot(0,0,'b.')
    ax.plot(H[:,0],H[:,1],'g.-');
    p1, = ax.plot(E[0,0],E[0,1],'b.')
    p2, = ax.plot(H[0,0],H[0,1],'b.')
    p3, = ax.plot((0,E[0,0],H[0,0]),(0,E[0,1],H[0,1]),'b-')    
    
    def init():
        p1.set_data([],[])
        p2.set_data([],[])
        p3.set_data([],[])        
        return p1,p2,p3
    
    def animate(i):
        p1.set_data([E[i,0]],[E[i,1]])
        p2.set_data(H[i,0],H[i,1])
        p3.set_data((0,E[i,0],H[i,0]),(0,E[i,1],H[i,1]))
        return p1,p2,p3
    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=len(E[:,0]), interval=20, blit=True)
    return anim
    # In order to make the JSAnimation to work it is necessary that the function returns the animation object!
    
    
def odeint_arms(twojointarm, state, t, aparams, torque):
    '''
    twojointarm: function object. Must receive (state,t,aparams,torque) and return [a1d,a2d,a1dd,a2dd]
    state: current states => [a1(t),a2(t),a1d(t),a2d(t)]
    t: array([t,t+1]) => current time step and next (t+1)
    returns next states [a1(t+1),a2(t+1),a1d(t+1),a2d(t+1)]
    '''
    from scipy.integrate import odeint

    return odeint(twojointarm, state, t, args=(aparams,torque))


def moving_average (values,window=6):
    weights = numpy.repeat(1.0, window)/window
    sma = numpy.convolve(numpy.concatenate((numpy.zeros(int((window-1)/2.0)),values,numpy.zeros((window-1)-int((window-1)/2.0)))), weights, 'valid')
    # I should try the function numpy.lib.pad instead of concatenating manually
    return sma


def moving_average (values, window=6):
    weights = numpy.repeat(1.0, window)/window
    sma = numpy.convolve(values, weights, 'valid')
    # I should try the function numpy.lib.pad instead of concatenating manually
    return numpy.lib.pad(sma, (int((window-1)/2.0),(window-1)-int((window-1)/2.0)), 'edge')