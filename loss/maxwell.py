import torch.autograd as autograd
import torch


class Maxwell2DMur():
    def __init__(self, criterion, args):
        super(Maxwell2DMur, self).__init__()
        self.e0 = args.e0
        self.u0 = args.u0
        self.criterion = criterion

    def governing_equation(self, out, input_data, src, eps):
        uEz = out[:, [0]]
        uHx = out[:, [1]]
        uHy = out[:, [2]]

        uEz_x_y_t = autograd.grad(uEz, input_data, torch.ones([input_data.shape[0], 1]).cuda(), retain_graph=True, create_graph=True)[0]
        uHx_x_y_t = autograd.grad(uHx, input_data, torch.ones([input_data.shape[0], 1]).cuda(), retain_graph=True, create_graph=True)[0]
        uHy_x_y_t = autograd.grad(uHy, input_data, torch.ones([input_data.shape[0], 1]).cuda(), retain_graph=True, create_graph=True)[0]
        
        uEz_x = uEz_x_y_t[:, [0]]
        uEz_y = uEz_x_y_t[:, [1]]
        uEz_t = uEz_x_y_t[:, [2]] 

        uHx_y = uHx_x_y_t[:, [1]]
        uHx_t = uHx_x_y_t[:, [2]]

        uHy_x = uHy_x_y_t[:, [0]]
        uHy_t = uHy_x_y_t[:, [2]]

        f_Ez = -1/(eps*self.e0)*(uHy_x - uHx_y) + (uEz_t) + src
        f_Hx = (uEz_y) + self.u0*uHx_t
        f_Hy = (uEz_x) - self.u0*uHy_t

        f_hat = torch.zeros_like(f_Ez).cuda()

        loss_f1 = self.criterion(f_Ez, f_hat)
        loss_f2 = self.criterion(f_Hx, f_hat)
        loss_f3 = self.criterion(f_Hy, f_hat)
        loss_f = loss_f1 + loss_f2 + loss_f3
        return loss_f

    def boundary_condition(self, out):
        boundary_hat = torch.zeros_like(out)
        loss_boundary = self.criterion(out, boundary_hat)
        return loss_boundary
    
    def initial_condition(self, out, u_true):
        loss_u = self.criterion(out, u_true)
        return loss_u






