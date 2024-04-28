import numpy as np
import torch
import time

from gene_data20 import generate_data, generate_xyt, generate_para
from architecture.warmup import wram_up



class Model:
    def __init__(self, network, optimizer=None, Maxwell=None, 
                criterion=None, args=None,
                **kwargs):
        self._train_network = network
        self.Maxwell = Maxwell
        self.criterion = criterion
        self._optimizer=optimizer 
        self.args = args                                                    
    
    def get_data_xyt(self):
        sampling_size = self.args.sampling_size
        c0 = 1/np.sqrt(self.args.e0*self.args.u0)
        dt =0.5/(c0*np.sqrt(1/(self.args.dx**2)+1/(self.args.dy**2)))
        range_t = dt*self.args.steps 
        src_pos = np.array([2.5, 1.25])  
        range_xy = self.args.dx*self.args.Nx
        method = 'lhs'  # 'sobol'
        xyt_no_src = generate_xyt(type="no_src", sampling_size=sampling_size, range_t=range_t, src_pos=src_pos, range_xy=range_xy, method=method)  
        xyt_src = generate_xyt(type='src', sampling_size=sampling_size, range_t=range_t, src_pos=src_pos, range_xy=range_xy, method=method)
        xyt_boundary = generate_xyt(type='boundary', sampling_size=sampling_size, range_t=range_t, src_pos=src_pos, range_xy=range_xy, method=method)
        return xyt_no_src, xyt_src, xyt_boundary
    
    def inference(self, num_items, optimizer):
        # get lr
        warmup_start_lr = 1e-4
        warmup_steps = 1000
        max_iter = num_items
        lr0 = 1e-3
        power = 0.9
        lrs = np.array(list(wram_up(warmup_start_lr, warmup_steps, max_iter, lr0, power)))

        start_time = time.time()
        self._train_network.train()
        for item in range(num_items):
            print('item', item)
            
            # Generating device parameter
            num_example = 1
            para_numpy = generate_para(num_example=num_example)
            print('para_numpy:', para_numpy)
        
            xyt_no_src, xyt_src, xyt_boundary = self.get_data_xyt()  # generate xyt points
            input_init_total_np, u_init_true_np, input_f_total_np, input_b_total_np = generate_data(
                                num_example=num_example, para_numpy=para_numpy,
                                xyt_no_src=xyt_no_src, xyt_src=xyt_src, xyt_boundary=xyt_boundary,
                                args=self.args
                                )
            
            
            para_input = torch.from_numpy(para_numpy).float().cuda()  
            input_init = torch.from_numpy(input_init_total_np[0, :, :]).float().cuda()  
            u_init = torch.from_numpy(u_init_true_np[0, :, :]).float().cuda() 
            input_f_total = torch.from_numpy(input_f_total_np[0, :, :]).float().cuda()  
            input_b_total = torch.from_numpy(input_b_total_np[0, :, :]).float().cuda()
            input_f = input_f_total[:, 0:3]
            src_train_f = input_f_total[:, 3:4]
            eps_train_f = input_f_total[:, 4:5]
            
                
            
            optimizer.zero_grad()
            
            batch_size = input_f.shape[0]
            para_input = para_input.view(num_example, 4).repeat([(batch_size), 1])  
            input_f.requires_grad = True
            out_f = self._train_network(input_f, para_input)
            out_init = self._train_network(input_init, para_input) 
            out_boundary = self._train_network(input_b_total, para_input)

            loss_init = self.Maxwell.initial_condition(out=out_init, u_true=u_init)
            loss_f = self.Maxwell.governing_equation(out=out_f, input_data=input_f, src=src_train_f, eps=eps_train_f)
            loss_boundary = self.Maxwell.boundary_condition(out=out_boundary)

            total_loss = 100*loss_init + loss_f + loss_boundary

            total_loss.backward()
            optimizer.step()
            for params in optimizer.param_groups:                        
                params['lr'] = lrs[item]  
                print(params['lr']) 

            print('total_loss: ', total_loss)
        end_time = time.time()
        print('Training time: %.2f' % (end_time-start_time))
           
    def train(self):
        self.inference(num_items=self.args.num_items, optimizer=self._optimizer)
        
                    
            
