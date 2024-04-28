import random
import numpy as np
import scipy.io
import skopt
from pyDOE import lhs

random.seed(1234)

def generate_xyt(type, sampling_size, range_t, src_pos, range_xy, method):
    if type == 'no_src':
        if method == 'sobol':
            sampler = skopt.sampler.Sobol(skip=0, randomize=False)
            space = [(0.0, 1.0)] * 3
            xyt = np.array(sampler.generate(space, sampling_size+2)[2:])
            xy = xyt[:, 0:2]*range_xy
            t = xyt[:, 2:3]*range_t
            xyt = np.hstack((xy, t))
            
        elif method == 'lhs':
            xyt = lhs(3, sampling_size)
            xy = xyt[:, 0:2]*range_xy
            t = xyt[:, 2:3]*range_t
            xyt = np.hstack((xy, t))
            
        return xyt
        
    elif type == 'src':
        if method == 'sobol':
            xy = np.tile(src_pos, (int(sampling_size/2000), 1))
            sampler = skopt.sampler.Sobol(skip=0, randomize=False)
            space = [(0.0, 1.0)] * 1
            t = np.array(sampler.generate(space, int(sampling_size+2))[2:])*range_t
            xyt = np.hstack((xy, t))
            
        elif method == 'lhs':
            xy = np.tile(src_pos, (int(sampling_size), 1))
            t = lhs(1, int(sampling_size))*range_t
            xyt = np.hstack((xy, t))
            
        return xyt

    elif type == 'boundary':
        dim = 2
        if method == 'sobol':
            sampler = skopt.sampler.Sobol(skip=0, randomize=False)
            space = [(0.0, 1.0)] * 3
            xyt = np.array(sampler.generate(space, int(sampling_size+2))[2:])
            xy = xyt[:, 0:2]
            t = xyt[:, 2:3]*range_t

            boundary_points = []
            for i in range(dim):
                ratio = 1/2
                num_sample = int(sampling_size * ratio)
                temp_boundary_points = xy[num_sample*i:num_sample*(i+1), :]
                temp_boundary_points[np.arange(num_sample), i] = np.round(temp_boundary_points[np.arange(num_sample), i])  # 四舍五入
                boundary_points.append(temp_boundary_points)
            boundary_data_xy = np.concatenate((boundary_points), 0)*range_xy
            boundary_data_xy = np.random.permutation(boundary_data_xy)
            xyt = np.hstack((boundary_data_xy, t))
            
        elif method == 'lhs':
            xyt = lhs(3, int(sampling_size))
            xy = xyt[:, 0:2]
            t = xyt[:, 2:3]*range_t
            boundary_points = []
            for i in range(dim):
                ratio = 1/2
                num_sample = int(sampling_size * ratio)
                temp_boundary_points = xy[num_sample*i:num_sample*(i+1), :]
                temp_boundary_points[np.arange(num_sample), i] = np.round(temp_boundary_points[np.arange(num_sample), i])  # 四舍五入
                boundary_points.append(temp_boundary_points)
            boundary_data_xy = np.concatenate((boundary_points), 0)*range_xy
            boundary_data_xy = np.random.permutation(boundary_data_xy)
            xyt = np.hstack((boundary_data_xy, t))
            
        return xyt

def generate_para(num_example):
    coord_eps_range = np.array([2.5, 3.5])
    coord_r_range = np.array([0.5, 1.5])  # Range of the radius of a circle
    src_x_range = np.array([2.0, 3.0])  # The value range of the circular origin coordinate x axis
    src_y_range = np.array([2.5, 3.5])  # The value range of the circular origin coordinate y axis
    src_pos = np.array([2.5, 1.25])
    
    

    para_list = []
    i = 0
    while i < num_example:
        src_x = round(random.uniform(src_x_range[0], src_x_range[1]), 4)
        src_y = round(random.uniform(src_y_range[0], src_y_range[1]), 4)
        src_x_y = np.array([src_x, src_y])

        ## Randomly generate the radius of the circle
        coord_r = round(random.uniform(coord_r_range[0], coord_r_range[1]), 4)

        if (src_x+coord_r) > 4.5 or (src_y+coord_r) > 4.5 or (src_y-coord_r)<0.5 or (src_x-coord_r)<0.5 or (np.linalg.norm((src_x_y - src_pos))-coord_r)<0.6:
            continue
        else:
            coord_eps = round(random.uniform(coord_eps_range[0], coord_eps_range[1]), 4)
            para_np = np.array([coord_eps, coord_r, src_x, src_y])
            para_list.append(para_np)
        i = i + 1
    
    para_np = np.array(para_list)
    return para_np

def generate_data(num_example, para_numpy, xyt_no_src, xyt_src, xyt_boundary, args):
    dataset = scipy.io.loadmat(args.init_path)
    Ez_usol = dataset['Ez']  
    Hx_usol = dataset['Hx']  
    Hy_usol = dataset['Hy'] 
    
    c0 = 1/np.sqrt(args.e0*args.u0)
    dt =0.5/(c0*np.sqrt(1/(args.dx**2)+1/(args.dy**2)))
    
    sampling_size = args.sampling_size 
    A = 10  
    tau = 10*dt
    t0 = 20*dt

    x_max = args.dx*args.Nx
    y_max = args.dy*args.Ny
    t_max = dt*args.steps

    x_ = np.linspace(start=0, stop=x_max, num=args.Nx, endpoint=False)  
    y_ = np.linspace(start=0, stop=y_max, num=args.Ny, endpoint=False) 
    t_ = np.linspace(start=0, stop=t_max, num=args.steps, endpoint=False)  
    X, T, Y = np.meshgrid(x_, t_, y_) 

    xyt_init = np.hstack((X[0, :, :].flatten()[:, None], Y[0, :, :].flatten()[:, None], T[0, :, :].flatten()[:, None]))  # (25600, 3)
    Ez_init_true = Ez_usol.flatten()[:, None]
    Hx_init_true = Hx_usol.flatten()[:, None]
    Hy_init_true = Hy_usol.flatten()[:, None]
    u_init_true = np.hstack((Ez_init_true, Hx_init_true, Hy_init_true))

    # gene dataset
    
    u_init_true_list = []
    input_init_list = []
    input_f_list = []
    input_b_list = []

    for i in range(num_example):
        coord_eps_ = para_numpy[i, 0]
        coord_r_ = para_numpy[i, 1]
        src_x_ = para_numpy[i, 2]
        src_y_ = para_numpy[i, 3]                     
        src_x_y_ = np.array([src_x_, src_y_])
        # init
        N_u = sampling_size
        idx_init = np.random.choice(xyt_init.shape[0], N_u, replace=False)
        xyt_init_ = xyt_init[idx_init, :]  
        u_init_true_ = u_init_true[idx_init, :]  
        input_init_list.append(xyt_init_)
        u_init_true_list.append(u_init_true_)

        # bc
        N_b = sampling_size
        idx_bc = np.random.choice(xyt_boundary.shape[0], N_b, replace=False)
        xyt_bc_ = xyt_boundary[idx_bc, :]  
        input_b_list.append(xyt_bc_)

        # data_f
        src_t = xyt_src[:, [2]]
        Esrc = A * np.exp(-(np.power((src_t+(40*dt)-t0)/tau, 2))).flatten()[:, None]  
        xyt_src_Esrc = np.hstack((xyt_src, Esrc))
        xyt_no_src_Esrc = np.hstack((xyt_no_src, np.zeros((xyt_no_src.shape[0], 1))))
        xyt_Esrc_f = np.vstack((xyt_src_Esrc, xyt_no_src_Esrc))
        N_f = sampling_size
        idx_f = np.random.choice(xyt_Esrc_f.shape[0], N_f, replace=False)
        input_f_xyt_Esrc = xyt_Esrc_f[idx_f, :]

        eps_f_norm = np.linalg.norm((input_f_xyt_Esrc[:, 0:2] - src_x_y_), axis=-1) <= coord_r_
        eps_f = np.where((eps_f_norm >= 1), coord_eps_, 1)[:, None]  
        input_f_xyt_Esrc_eps = np.hstack((input_f_xyt_Esrc, eps_f))
        input_f_list.append(input_f_xyt_Esrc_eps)

    input_init_total_np = np.array(input_init_list)  
    u_init_true_np = np.array(u_init_true_list)
    input_f_total_np = np.array(input_f_list)
    input_b_total_np = np.array(input_b_list)

    return input_init_total_np, u_init_true_np, input_f_total_np, input_b_total_np



    