import numpy as np

EPSILON_OMEGA = 1e-3

def compute_dynamics(xvec, u, dt, compute_jacobians=True):
    """
    Compute Turtlebot dynamics (unicycle model).

    Inputs:
                     xvec: np.array[3,] - Turtlebot state (x, y, theta).
                        u: np.array[2,] - Turtlebot controls (V, omega).
        compute_jacobians: bool         - compute Jacobians Gx, Gu if true.
    Outputs:
         g: np.array[3,]  - New state after applying u for dt seconds.
        Gx: np.array[3,3] - Jacobian of g with respect to xvec.
        Gu: np.array[3,2] - Jacobian of g with respect to u.
    """
    ########## Code starts here ##########
    # TODO: Compute g, Gx, Gu
    # HINT: To compute the new state g, you will need to integrate the dynamics of x, y, theta
    # HINT: Since theta is changing with time, try integrating x, y wrt d(theta) instead of dt by introducing om
    # HINT: When abs(om) < EPSILON_OMEGA, assume that the theta stays approximately constant ONLY for calculating the next x, y
    #       New theta should not be equal to theta. Jacobian with respect to om is not 0.
    V, w = u
    x, y, theta = xvec
    new_theta = theta + w * dt
    if abs(w) > EPSILON_OMEGA: # Make EPSILON OMEGA


        cos_theta_sub = np.cos(new_theta) - np.cos(theta)
        sin_theta_sub = np.sin(new_theta) - np.sin(theta)
        
        g = np.array([x + (V / w) * (sin_theta_sub), y - (V / w) * (cos_theta_sub), new_theta])
        # np.sin(theta) and np.cos(theta) term drop out in derivatives.
        Gx = np.array([[1, 0, (cos_theta_sub) * V / w ], [0, 1, (sin_theta_sub) * V / w], [0, 0, 1]])
        Gu = np.array([[(sin_theta_sub) / w, (-V * (sin_theta_sub) / (w ** 2)) + (V * np.cos(new_theta) * dt) / w], [-(cos_theta_sub) / w, -V * (cos_theta_sub) / w ** 2 + (V * (-np.sin(new_theta)) * dt) / w], [0, dt]])
    else:
        g = np.array([x,y, theta])
        Gx = np.array([[1, 0, np.cos(theta) * V], [0, 1, V * np.sin(theta)], [0, 0, 1]] )
        Gu = np.array([[np.sin(theta), 0], [-np.cos(theta), 0], [0, 0]])
        # derive with resp to w = 0 in this case -> TODO we don't know what else it should be. 
        
     
        
        
   


    ########## Code ends here ##########

    if not compute_jacobians:
        return g

    return g, Gx, Gu

def transform_line_to_scanner_frame(line, x, tf_base_to_camera, compute_jacobian=True):
    """
    Given a single map line in the world frame, outputs the line parameters
    in the scanner frame so it can be associated with the lines extracted
    from the scanner measurements.

    Input:
                     line: np.array[2,] - map line (alpha, r) in world frame.
                        x: np.array[3,] - pose of base (x, y, theta) in world frame.
        tf_base_to_camera: np.array[3,] - pose of camera (x, y, theta) in base frame.
         compute_jacobian: bool         - compute Jacobian Hx if true.
    Outputs:
         h: np.array[2,]  - line parameters in the scanner (camera) frame.
        Hx: np.array[2,3] - Jacobian of h with respect to x.
    """
    alpha, r = line

    ########## Code starts here ##########
    # TODO: Compute h, Hx
    # HINT: Calculate the pose of the camera in the world frame (x_cam, y_cam, th_cam), a rotation matrix may be useful.
    # HINT: To compute line parameters in the camera frame h = (alpha_in_cam, r_in_cam), 
    #       draw a diagram with a line parameterized by (alpha,r) in the world frame and 
    #       a camera frame with origin at x_cam, y_cam rotated by th_cam wrt to the world frame
    # HINT: What is the projection of the camera location (x_cam, y_cam) on the line r? 
    # HINT: To find Hx, write h in terms of the pose of the base in world frame (x_base, y_base, th_base)
    cam_world = np.zeros_like(tf_base_to_camera)
    
    rotation_matrix = np.array([[np.cos(x[2]), -np.sin(x[2])],
                                [np.sin(x[2]), np.cos(x[2])]])
    
    cam_world = np.dot(rotation_matrix, tf_base_to_camera[:2])
    cam_world = np.append(cam_world, tf_base_to_camera[2])
    
    cam_world += x
    
    x_cam, y_cam, th_cam = cam_world
    
    h = np.zeros_like(line)
    h[0] = alpha - th_cam
    h[1] = r - np.cos(alpha - th_cam) * np.sqrt(x_cam ** 2 + y_cam ** 2)
    
    
    
    xc = np.cos(x[2]) * tf_base_to_camera[0] - np.sin(x[2]) * tf_base_to_camera[1] + x[0]
    yc = np.sin(x[2]) * tf_base_to_camera[0] + np.cos(x[2]) * tf_base_to_camera[1] + x[1]
    
    sum_of_squares = xc ** 2 + yc ** 2
   
    alpha_minus_theta = h[0]
    
    partial_wrt_x = np.cos(alpha_minus_theta) * 0.5 / np.sqrt(sum_of_squares)
    
    partial_wrt_x *= 2 * (xc)
    
    
    partial_wrt_y = np.cos(alpha_minus_theta) * 0.5 / np.sqrt(sum_of_squares)
    
    partial_wrt_y *= 2 * yc
    
    
    xc_wrt_th = (tf_base_to_camera[0] * -np.sin(x[2]) - np.cos(x[2]) * tf_base_to_camera[1])
    yc_wrt_th = tf_base_to_camera[0] * np.cos(x[2]) + tf_base_to_camera[1] * -np.sin(x[2])
    
    partial_wrt_th = -np.sin(alpha_minus_theta) * -1 * np.sqrt(sum_of_squares) + (np.cos(alpha_minus_theta) * (0.5 / np.sqrt(sum_of_squares)) * ((2 * xc) * (xc_wrt_th) + 2 * yc * yc_wrt_th) )  # product rule.
    
    
    
    Hx = np.array([[0, 0, -1], [partial_wrt_x, partial_wrt_y, partial_wrt_th]])
    ########## Code ends here ##########

    if not compute_jacobian:
        return h

    return h, Hx


def normalize_line_parameters(h, Hx=None):
    """
    Ensures that r is positive and alpha is in the range [-pi, pi].

    Inputs:
         h: np.array[2,]  - line parameters (alpha, r).
        Hx: np.array[2,n] - Jacobian of line parameters with respect to x.
    Outputs:
         h: np.array[2,]  - normalized parameters.
        Hx: np.array[2,n] - Jacobian of normalized line parameters. Edited in place.
    """
    alpha, r = h
    if r < 0:
        alpha += np.pi
        r *= -1
        if Hx is not None:
            Hx[1,:] *= -1
    alpha = (alpha + np.pi) % (2*np.pi) - np.pi
    h = np.array([alpha, r])

    if Hx is not None:
        return h, Hx
    return h
