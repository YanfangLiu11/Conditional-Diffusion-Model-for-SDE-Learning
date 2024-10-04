import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import torch.nn as nn
import numpy as np
import torch.optim as optim
import torch
from scipy import stats
import torch.distributions as td
import numpy.matlib
from scipy.stats import qmc
from sklearn.neighbors import KernelDensity
from scipy.linalg import expm
import os
from scipy.stats import expon, gaussian_kde
from matplotlib.colors import ListedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch
from matplotlib.lines import Line2D



import faiss


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('device',device)
torch.set_default_dtype(torch.float32)

torch.manual_seed(12345678)
np.random.seed(12312414)

def make_folder(folder):
    """
    Creates given folder (or path) if it doesn't exist.
    """
    if not os.path.exists(folder):
        os.makedirs(folder)

#-----------------------------------------------------
#            Setup the diffusion model
#-----------------------------------------------------
def cond_alpha(t, dt):
    # conditional information
    # alpha_t(0) = 1
    # alpha_t(1) = 0
    return 1 - t + dt


def cond_sigma2(t, dt):
    # conditional sigma^2
    # sigma2_t(0) = 0
    # sigma2_t(1) = 1
    # return (1-scalar) + t*scalar
    return t + dt

def f(t, dt):
    # f=d_(log_alpha)/dt
    alpha_t = cond_alpha(t,dt)
    f_t = -1.0/(alpha_t)
    return f_t


def g2(t, dt):
    # g = d(sigma_t^2)/dt -2f sigma_t^2
    dsigma2_dt = 1.0
    f_t = f(t,dt)
    sigma2_t = cond_sigma2(t,dt)
    g2 = dsigma2_dt - 2*f_t*sigma2_t
    return g2

def g(t,dt):
    return (g2(t,dt))**0.5


odeslover_time_steps = 10000
t_vec = torch.linspace(1.0,0.0,odeslover_time_steps +1)
def ODE_solver(zt,x_sample,z_sample,x0_test,time_steps):
    log_weight_likelihood = -1.0* torch.sum( (x0_test[:,None,:]-x_sample)**2/2 , axis = 2, keepdims= False)
    weight_likelihood =torch.exp(log_weight_likelihood)
    for j in range(time_steps): 
        t = t_vec[j+1]
        dt = t_vec[j] - t_vec[j+1]

        score_gauss = -1.0*(zt[:,None,:]-cond_alpha(t,dt)*z_sample)/cond_sigma2(t,dt)

        log_weight_gauss= -1.0* torch.sum( (zt[:,None,:]-cond_alpha(t,dt)*z_sample)**2/(2*cond_sigma2(t,dt)) , axis =2, keepdims= False)
        weight_temp = torch.exp( log_weight_gauss )
        weight_temp = weight_temp*weight_likelihood
        weight = weight_temp/ torch.sum(weight_temp,axis=1, keepdims=True)
        score = torch.sum(score_gauss*weight[:,:,None],axis=1, keepdims= False)  
        
        zt= zt - (f(t,dt)*zt-0.5*g2(t, dt)*score) *dt
    return zt



def process_chunk(it_n_index, it_size_x0train, short_size,x_sample, x0_train, train_size):
    x0_train_index_initial = np.empty((train_size, short_size ), dtype=int)
    gpu = faiss.StandardGpuResources()  # Initialize GPU resources each time
    index = faiss.IndexFlatL2(x_dim)  # Create a FAISS index for exact searches
    gpu_index = faiss.index_cpu_to_gpu(gpu, 0, index)
    gpu_index.add(x_sample)  # Add the chunk of x_sample to the index
    for jj in range(it_n_index):
        start_idx = jj * it_size_x0train
        end_idx = min((jj + 1) * it_size_x0train, train_size)
        x0_train_chunk = x0_train[start_idx:end_idx]

        # Perform the search
        _, index_initial = gpu_index.search(x0_train_chunk, short_size)
        x0_train_index_initial[start_idx:end_idx,:] = index_initial 

        if jj % 500 == 0:
            print('find indx iteration:', jj, it_size_x0train)
    # Cleanup resources
    del gpu_index
    del index
    del gpu
    return x0_train_index_initial

# Define the architecture of the  neural network
class FN_Net(nn.Module):
    
    def __init__(self, input_dim, output_dim, hid_size):
        super(FN_Net, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hid_size = hid_size
        
        self.input = nn.Linear(self.input_dim,self.hid_size)
        self.fc1 = nn.Linear(self.hid_size,self.hid_size)
        self.output = nn.Linear(self.hid_size,self.output_dim)

        self.best_input_weight = torch.clone(self.input.weight.data)
        self.best_input_bias = torch.clone(self.input.bias.data)
        self.best_fc1_weight = torch.clone(self.fc1.weight.data)
        self.best_fc1_bias = torch.clone(self.fc1.bias.data)
        self.best_output_weight = torch.clone(self.output.weight.data)
        self.best_output_bias = torch.clone(self.output.bias.data)
    
    def forward(self,x):
        x = torch.tanh(self.input(x))
        x = torch.tanh(self.fc1(x))
        x = self.output(x)
        return x

    def update_best(self):

        self.best_input_weight = torch.clone(self.input.weight.data)
        self.best_input_bias = torch.clone(self.input.bias.data)
        self.best_fc1_weight = torch.clone(self.fc1.weight.data)
        self.best_fc1_bias = torch.clone(self.fc1.bias.data)
        self.best_output_weight = torch.clone(self.output.weight.data)
        self.best_output_bias = torch.clone(self.output.bias.data)

    def final_update(self):

        self.input.weight.data = self.best_input_weight 
        self.input.bias.data = self.best_input_bias
        self.fc1.weight.data = self.best_fc1_weight
        self.fc1.bias.data = self.best_fc1_bias
        self.output.weight.data = self.best_output_weight
        self.output.bias.data = self.best_output_bias



def std_normal2(N_data, t_steps, seeds,Sig):
    # x0 and t_steps should be 1d array
    np.random.seed(seeds)
    diff = t_steps[0,1:]-t_steps[0,:-1]
    
    n_dim = np.shape(Sig)[0]
    grow = np.zeros([N_data,t_steps.shape[1]])
    grow_temp = np.zeros([N_data,t_steps.shape[1],n_dim])
    noise = np.random.normal(0.0, np.sqrt(diff[0]), [ N_data,t_steps.shape[1]-1,n_dim])
    for i in range(t_steps.shape[1]-1):
        grow_temp[:,i+1,:] = noise[:,i,:]
    grow = Sig@ grow_temp.transpose(0,2,1)
    return grow

def Ornstein_Uhlenbeck_2D(initial,N_data, T, dt, n_dim, B, Sigma, seeds,IC_):
    Nt = int(T/dt)
    t = np.linspace(0.0,T,Nt+1).reshape(1,-1)
    Ext = expm(t.T[:,None]*B)
    X_t = np.zeros((Nt+1,n_dim,N_data))
    brownian = (std_normal2(N_data, t, seeds,Sigma)).transpose(2,1,0)
    
    xIC = np.zeros(( n_dim,N_data))
    if IC_=='uniform':
        np.random.seed(1)
        xIC[0,:] = np.random.uniform(-4, 4, N_data)
        xIC[1,:] = np.random.uniform(-3, 3, N_data)
    elif IC_=='value':
        xIC = (initial.reshape(-1,1))*np.ones((n_dim,N_data))

    Neg_Ext =expm(-t.T[:,None]*B)
    
    dW = np.zeros((Nt+1,n_dim,N_data))
    dW_temp1 = np.zeros((Nt+1,n_dim,N_data))
    dW_temp2 = np.zeros((Nt+1,n_dim,N_data))
    for jj in range(Nt+1):
        dW_temp1[jj,:,:] =  Neg_Ext[jj,:,:]@ brownian[jj, :, :]
    dW_temp2=np.cumsum( dW_temp1, axis=0)
    for jj in range(Nt+1):
        dW[jj,:,:] =  Ext[jj,:,:]@ dW_temp2[jj, :, :]
    X_t =   Ext@xIC + dW 

    # print(np.shape(X_t))
    return X_t
#================================================
B = np.array([[-1, -0.5], [-1, -1]])
Sigma = np.array([[1, 0], [0, 0.5]])
x_dim = 2

sde_T = 1.0
sde_dt = 0.01
sde_Nt = int(sde_T /sde_dt)
seeds=1
true_init = np.array([0.3, 0.4])

savedir = 'C:\\Users\\yliu1\\OneDrive - Middle Tennessee State University\\cyberduck\\sde_learning\\final_code\\2D_OU\\'
make_folder(savedir)

N_sample_path = 350000
data_sample = Ornstein_Uhlenbeck_2D(true_init,N_sample_path, sde_T, sde_dt, x_dim,  B, Sigma,seeds,'uniform')
print('size of data_sample is: ', data_sample.shape )
x_sample = (data_sample[:sde_Nt,:,:]).transpose(0, 2, 1).reshape((N_sample_path*sde_Nt,x_dim))
y_sample = (data_sample[1:sde_Nt+1,:,:]).transpose(0, 2, 1).reshape((N_sample_path*sde_Nt,x_dim))

diff_scale = np.array([5,10])
xy_diff = (y_sample-x_sample)*diff_scale
print('difference of x_sample and y_sample is: ', xy_diff)
print('mean and std of xy_diff are: ',  np.mean(xy_diff, axis = 0),np.std(xy_diff, axis=0))


sample_size = x_sample.shape[0]
print('sample size:', sample_size )

train_size =  120000
selected_row_indices =  np.random.permutation(sample_size)[:train_size]
x0_train = x_sample[selected_row_indices]
print('size of x0_train is: ', x0_train.shape )


short_size = 2048

it_size_x0train = 4000
it_n_index = train_size // it_size_x0train

x_short_indx = process_chunk(it_n_index, it_size_x0train, short_size,x_sample, x0_train, train_size)
print('finish finding indx',short_size, it_size_x0train) 



x_short = x_sample[x_short_indx]
z_short = xy_diff[x_short_indx]

np.save(os.path.join(savedir, f"data_training_x_short.npy"), x_short)
np.save(os.path.join(savedir, f"data_training_z_short.npy"), z_short)
np.save(os.path.join(savedir, f"data_training_x0train.npy"), x0_train)

del x_sample, xy_diff,x_short_indx

zT = np.random.randn(train_size,x_dim)
yTrain = np.zeros((train_size,x_dim))
it_size = min(60000,train_size)
it_n = int(train_size/it_size)
for jj in range(it_n):

    start_idx = jj * it_size
    end_idx = min((jj + 1) * it_size, train_size)
    it_zt = torch.tensor(zT[start_idx: end_idx]).to(device, dtype=torch.float32)
    it_x0 =  torch.tensor(x0_train[start_idx: end_idx]).to(device, dtype=torch.float32)

    x_mini_batch =  torch.tensor(x_short[start_idx:end_idx]).to(device, dtype=torch.float32)
    z_mini_batch =  torch.tensor(z_short[start_idx: end_idx]).to(device, dtype=torch.float32)
    
    y_temp = ODE_solver( it_zt , x_mini_batch, z_mini_batch,  it_x0, odeslover_time_steps)
    yTrain[start_idx: end_idx, :x_dim] = y_temp.to('cpu').detach().numpy()

    if jj%5==0:
        print(jj)
   
xTrain = np.hstack((x0_train,zT))

np.save(os.path.join(savedir, 'xTrain.npy'), xTrain)
np.save(os.path.join(savedir, 'yTrain.npy'), yTrain)

#========================================================================================================
#   train F(x0,z)=x1_hat-x0 using labled data (x0,z, x1_hat-x0)
#========================================================================================================
# Check for infinite values in yTrain
is_finite_yTrain = np.isfinite(yTrain) & ~np.isnan(yTrain)
# Exclude rows with infinite values
xTrain_filtered = xTrain[is_finite_yTrain.all(axis=1)]
yTrain_filtered = yTrain[is_finite_yTrain.all(axis=1)]


train_size_new = xTrain_filtered.shape[0]

print('trainning data size:', train_size_new)
# Generate random indices for shuffling
indices = np.random.permutation(train_size_new)
# Shuffle xTrain and yTrain using the generated indices
xTrain_shuffled = xTrain_filtered[indices]
yTrain_shuffled = yTrain_filtered[indices]

xTrain_mean = np.mean(xTrain_filtered, axis=0, keepdims=True)
xTrain_std = np.std(xTrain_filtered, axis=0, keepdims=True)
yTrain_mean = np.mean(yTrain_filtered, axis=0, keepdims=True)
yTrain_std = np.std(yTrain_filtered, axis=0, keepdims=True)


xTrain_new = (xTrain_shuffled - xTrain_mean) / xTrain_std
yTrain_new = (yTrain_shuffled - yTrain_mean) / yTrain_std

# Convert data to a tensor
xTrain_new = torch.tensor(xTrain_new, dtype=torch.float32).to(device)
yTrain_new = torch.tensor(yTrain_new, dtype=torch.float32).to(device)
xTrain_mean = torch.tensor(xTrain_mean, dtype=torch.float32).to(device)
xTrain_std = torch.tensor(xTrain_std , dtype=torch.float32).to(device)
yTrain_mean = torch.tensor(yTrain_mean, dtype=torch.float32).to(device)
yTrain_std  = torch.tensor(yTrain_std , dtype=torch.float32).to(device)

dataname2 = os.path.join(savedir, 'data_inf.pt')
torch.save({'xTrain_mean': xTrain_mean,'xTrain_std': xTrain_std, 'yTrain_mean': yTrain_mean, 'yTrain_std': yTrain_std, 'diff_scale': diff_scale}, dataname2)

print('xTrain_mean:', xTrain_mean)
print('xTrain_std:', xTrain_std)
print( 'yTrain_mean:', yTrain_mean)
print( 'yTrain_std:', yTrain_std)


NTrain = int(train_size_new* 0.8)
NValid = int(train_size_new * 0.2)

xValid_normal = xTrain_new [NTrain:,:]
yValid_normal = yTrain_new [NTrain:,:]

xTrain_normal = xTrain_new [:NTrain,:]
yTrain_normal = yTrain_new [:NTrain,:]

learning_rate = 0.01
FN = FN_Net(x_dim*2, x_dim, 50).to(device)
FN.zero_grad()
optimizer = optim.Adam(FN.parameters(), lr = learning_rate, weight_decay=1e-6)
criterion = nn.MSELoss()

best_valid_err = 5.0
n_iter = 2000
for j in range(n_iter):
    optimizer.zero_grad()
    pred = FN(xTrain_normal)
    loss = criterion(pred,yTrain_normal)
    loss.backward()
    optimizer.step()

    pred1 = FN(xValid_normal)
    valid_loss = criterion(pred1,yValid_normal)
    if valid_loss < best_valid_err:
        FN.update_best()
        best_valid_err = valid_loss

    if j%100==0:
        print(j,loss,valid_loss)

FN.final_update()

FN_path = os.path.join(savedir, 'FN.pth')
torch.save(FN.state_dict(), FN_path)


#======================================================
# Result of NN
#====================================================== 

# sde_dt = sde_t1 - sde_t0

Npath = 500000

ode_time_steps = 500

ode_mean_pred = np.zeros((ode_time_steps,x_dim))
ode_std_pred = np.zeros((ode_time_steps,x_dim))
ode_mean_true = np.zeros((ode_time_steps,x_dim))
ode_std_true = np.zeros((ode_time_steps,x_dim))

true_init = np.array([-0.8, 1.2])
ode_path_true = Ornstein_Uhlenbeck_2D(true_init,Npath, ode_time_steps*sde_dt, sde_dt, x_dim,  B, Sigma,seeds,'value')
ode_mean_true =  (np.mean(ode_path_true,axis=2))[1:ode_time_steps+1,:]
ode_std_true  = (np.std(ode_path_true,axis=2))[1:ode_time_steps+1,:]

x_pred_new = torch.ones(Npath, x_dim).to(device,dtype=torch.float32) * torch.tensor(true_init).to(device,dtype=torch.float32)

for jj in range(ode_time_steps):
    
    prediction = FN( (torch.hstack((x_pred_new,torch.randn(Npath,x_dim).to(device,dtype= torch.float32)))-xTrain_mean)/xTrain_std  ) * yTrain_std + yTrain_mean 
    prediction = ( prediction.to('cpu').detach().numpy() /diff_scale +x_pred_new.to('cpu').detach().numpy()  )
    
    ode_mean_pred[jj,:] = np.mean(prediction,axis=0)
    ode_std_pred[jj,:] = np.std(prediction,axis=0)
    
    x_pred_new= torch.tensor( prediction  ).to(device,dtype= torch.float32)

    print(jj, ode_mean_true[jj], ode_mean_pred[jj], ode_std_true[jj], ode_std_pred[jj])


tmesh = np.linspace(sde_dt, ode_time_steps*sde_dt, ode_time_steps)

fig, ax = plt.subplots(1, 1, figsize=(10, 6))

ax.plot(tmesh, ode_mean_true[:,0])
ax.plot(tmesh, ode_mean_pred[:,0])
ax.plot(tmesh, ode_mean_true[:,0] - ode_std_true[:,0])
ax.plot(tmesh, ode_mean_true[:,0] + ode_std_true[:,0])
ax.fill_between(tmesh, ode_mean_pred[:,0] - ode_std_pred[:,0], ode_mean_pred[:,0] + ode_std_pred[:,0], alpha=0.2)
ax.legend(['True mean', 'Pred mean', 'True mean-std', 'True mean+std', 'Pred std'])

ax.set_xlabel('Time', fontsize=16)
ax.set_ylabel('Value', fontsize=16)
ax.set_title(f'$x_0$ = {true_init[0]:.2f}' , fontsize=16)

# Increase font size of tick labels
ax.tick_params(axis='both', labelsize=14)

plt.savefig(os.path.join(savedir, f'd1_final_image_{true_init[0]:.2f}_N{ode_time_steps}.png'), dpi=300, bbox_inches='tight')
plt.show()




fig, ax = plt.subplots(1, 1, figsize=(10, 6))

ax.plot(tmesh, ode_mean_true[:,1])
ax.plot(tmesh, ode_mean_pred[:,1])
ax.plot(tmesh, ode_mean_true[:,1] - ode_std_true[:,1])
ax.plot(tmesh, ode_mean_true[:,1] + ode_std_true[:,1])
ax.fill_between(tmesh, ode_mean_pred[:,1] - ode_std_pred[:,1], ode_mean_pred[:,1] + ode_std_pred[:,1], alpha=0.2)
ax.legend(['True mean', 'Pred mean', 'True mean-std', 'True mean+std', 'Pred std'])

ax.set_xlabel('Time', fontsize=16)
ax.set_ylabel('Value', fontsize=16)
ax.set_title(f'$x_0$ = {true_init[1]:.2f}' , fontsize=16)

# Increase font size of tick labels
ax.tick_params(axis='both', labelsize=14)

plt.savefig(os.path.join(savedir, f'd2_final_image_{true_init[1]:.2f}_N{ode_time_steps}.png'), dpi=300, bbox_inches='tight')
plt.show()





ode_mean_pred = np.zeros((ode_time_steps,x_dim))
ode_std_pred = np.zeros((ode_time_steps,x_dim))
ode_mean_true = np.zeros((ode_time_steps,x_dim))
ode_std_true = np.zeros((ode_time_steps,x_dim))

true_init = np.array([0.3, 0.4])
ode_path_true = Ornstein_Uhlenbeck_2D(true_init,Npath, ode_time_steps*sde_dt, sde_dt, x_dim,  B, Sigma,seeds,'value')
ode_mean_true =  (np.mean(ode_path_true,axis=2))[1:ode_time_steps+1,:]
ode_std_true  = (np.std(ode_path_true,axis=2))[1:ode_time_steps+1,:]

x_pred_new = torch.ones(Npath, x_dim).to(device,dtype=torch.float32) * torch.tensor(true_init).to(device,dtype=torch.float32)

for jj in range(ode_time_steps):
    
    prediction = FN( (torch.hstack((x_pred_new,torch.randn(Npath,x_dim).to(device,dtype= torch.float32)))-xTrain_mean)/xTrain_std  ) * yTrain_std + yTrain_mean 
    prediction = ( prediction.to('cpu').detach().numpy() /diff_scale +x_pred_new.to('cpu').detach().numpy()  )
    
    ode_mean_pred[jj,:] = np.mean(prediction,axis=0)
    ode_std_pred[jj,:] = np.std(prediction,axis=0)
    
    x_pred_new= torch.tensor( prediction  ).to(device,dtype= torch.float32)

    print(jj, ode_mean_true[jj], ode_mean_pred[jj], ode_std_true[jj], ode_std_pred[jj])


tmesh = np.linspace(sde_dt, ode_time_steps*sde_dt, ode_time_steps)


fig, ax = plt.subplots(1, 1, figsize=(10, 6))

ax.plot(tmesh, ode_mean_true[:,0])
ax.plot(tmesh, ode_mean_pred[:,0])
ax.plot(tmesh, ode_mean_true[:,0] - ode_std_true[:,0])
ax.plot(tmesh, ode_mean_true[:,0] + ode_std_true[:,0])
ax.fill_between(tmesh, ode_mean_pred[:,0] - ode_std_pred[:,0], ode_mean_pred[:,0] + ode_std_pred[:,0], alpha=0.2)
ax.legend(['True mean', 'Pred mean', 'True mean-std', 'True mean+std', 'Pred std'])

ax.set_xlabel('Time', fontsize=16)
ax.set_ylabel('Value', fontsize=16)
ax.set_title(f'$x_0$ = {true_init[0]:.2f}' , fontsize=16)

# Increase font size of tick labels
ax.tick_params(axis='both', labelsize=14)

plt.savefig(os.path.join(savedir, f'd1_final_image_{true_init[0]:.2f}_N{ode_time_steps}.png'), dpi=300, bbox_inches='tight')
plt.show()




fig, ax = plt.subplots(1, 1, figsize=(10, 6))

ax.plot(tmesh, ode_mean_true[:,1])
ax.plot(tmesh, ode_mean_pred[:,1])
ax.plot(tmesh, ode_mean_true[:,1] - ode_std_true[:,1])
ax.plot(tmesh, ode_mean_true[:,1] + ode_std_true[:,1])
ax.fill_between(tmesh, ode_mean_pred[:,1] - ode_std_pred[:,1], ode_mean_pred[:,1] + ode_std_pred[:,1], alpha=0.2)
ax.legend(['True mean', 'Pred mean', 'True mean-std', 'True mean+std', 'Pred std'])

ax.set_xlabel('Time', fontsize=16)
ax.set_ylabel('Value', fontsize=16)
ax.set_title(f'$x_0$ = {true_init[1]:.2f}' , fontsize=16)

# Increase font size of tick labels
ax.tick_params(axis='both', labelsize=14)

plt.savefig(os.path.join(savedir, f'd2_final_image_{true_init[1]:.2f}_N{ode_time_steps}.png'), dpi=300, bbox_inches='tight')
plt.show()



ode_mean_pred = np.zeros((ode_time_steps,x_dim))
ode_std_pred = np.zeros((ode_time_steps,x_dim))
ode_mean_true = np.zeros((ode_time_steps,x_dim))
ode_std_true = np.zeros((ode_time_steps,x_dim))

true_init = np.array([-0.5, -0.5])
ode_path_true = Ornstein_Uhlenbeck_2D(true_init,Npath, ode_time_steps*sde_dt, sde_dt, x_dim,  B, Sigma,seeds,'value')
ode_mean_true =  (np.mean(ode_path_true,axis=2))[1:ode_time_steps+1,:]
ode_std_true  = (np.std(ode_path_true,axis=2))[1:ode_time_steps+1,:]

x_pred_new = torch.ones(Npath, x_dim).to(device,dtype=torch.float32) * torch.tensor(true_init).to(device,dtype=torch.float32)

for jj in range(ode_time_steps):
    
    prediction = FN( (torch.hstack((x_pred_new,torch.randn(Npath,x_dim).to(device,dtype= torch.float32)))-xTrain_mean)/xTrain_std  ) * yTrain_std + yTrain_mean 
    prediction = ( prediction.to('cpu').detach().numpy() /diff_scale +x_pred_new.to('cpu').detach().numpy()  )
    
    ode_mean_pred[jj,:] = np.mean(prediction,axis=0)
    ode_std_pred[jj,:] = np.std(prediction,axis=0)
    
    x_pred_new= torch.tensor( prediction  ).to(device,dtype= torch.float32)

    print(jj, ode_mean_true[jj], ode_mean_pred[jj], ode_std_true[jj], ode_std_pred[jj])


tmesh = np.linspace(sde_dt, ode_time_steps*sde_dt, ode_time_steps)


fig, ax = plt.subplots(1, 1, figsize=(10, 6))

ax.plot(tmesh, ode_mean_true[:,0])
ax.plot(tmesh, ode_mean_pred[:,0])
ax.plot(tmesh, ode_mean_true[:,0] - ode_std_true[:,0])
ax.plot(tmesh, ode_mean_true[:,0] + ode_std_true[:,0])
ax.fill_between(tmesh, ode_mean_pred[:,0] - ode_std_pred[:,0], ode_mean_pred[:,0] + ode_std_pred[:,0], alpha=0.2)
ax.legend(['True mean', 'Pred mean', 'True mean-std', 'True mean+std', 'Pred std'])

ax.set_xlabel('Time', fontsize=16)
ax.set_ylabel('Value', fontsize=16)
ax.set_title(f'$x_0$ = {true_init[0]:.2f}' , fontsize=16)

# Increase font size of tick labels
ax.tick_params(axis='both', labelsize=14)

plt.savefig(os.path.join(savedir, f'd1_final_image_{true_init[0]:.2f}_N{ode_time_steps}.png'), dpi=300, bbox_inches='tight')
plt.show()




fig, ax = plt.subplots(1, 1, figsize=(10, 6))

ax.plot(tmesh, ode_mean_true[:,1])
ax.plot(tmesh, ode_mean_pred[:,1])
ax.plot(tmesh, ode_mean_true[:,1] - ode_std_true[:,1])
ax.plot(tmesh, ode_mean_true[:,1] + ode_std_true[:,1])
ax.fill_between(tmesh, ode_mean_pred[:,1] - ode_std_pred[:,1], ode_mean_pred[:,1] + ode_std_pred[:,1], alpha=0.2)
ax.legend(['True mean', 'Pred mean', 'True mean-std', 'True mean+std', 'Pred std'])

ax.set_xlabel('Time', fontsize=16)
ax.set_ylabel('Value', fontsize=16)
ax.set_title(f'$x_0$ = {true_init[1]:.2f}' , fontsize=16)

# Increase font size of tick labels
ax.tick_params(axis='both', labelsize=14)

plt.savefig(os.path.join(savedir, f'd2_final_image_{true_init[1]:.2f}_N{ode_time_steps}.png'), dpi=300, bbox_inches='tight')
plt.show()


# ======================================
#  check conditional distribution

Npath = 400000

ode_time_steps = 1
ode_mean_pred = np.zeros((ode_time_steps,x_dim))
ode_std_pred = np.zeros((ode_time_steps,x_dim))
ode_mean_true = np.zeros((ode_time_steps,x_dim))
ode_std_true = np.zeros((ode_time_steps,x_dim))

true_init = np.array([0.0, 0.0])
x_pred_new = torch.ones(Npath, x_dim).to(device,dtype= torch.float32) * torch.tensor(true_init).to(device,dtype= torch.float32)

ode_path_true = Ornstein_Uhlenbeck_2D(true_init,Npath, ode_time_steps*sde_dt, sde_dt, x_dim,  B, Sigma,seeds,'value')
# print(np.shape(ode_path_true))
# print(np.shape(np.mean(ode_path_true,axis=2)))
ode_mean_true =  (np.mean(ode_path_true,axis=2))[1:ode_time_steps+1,:]
ode_std_true  = (np.std(ode_path_true,axis=2))[1:ode_time_steps+1,:]
for jj in range(ode_time_steps):
    
    prediction = FN( (torch.hstack((x_pred_new,torch.randn(Npath,x_dim).to(device,dtype= torch.float32)))-xTrain_mean)/xTrain_std  ) * yTrain_std + yTrain_mean 
    prediction = ( prediction.to('cpu').detach().numpy() /diff_scale +x_pred_new.to('cpu').detach().numpy()  )
    
    ode_mean_pred[jj,:] = np.mean(prediction,axis=0)
    ode_std_pred[jj,:] = np.std(prediction,axis=0)
    
    x_pred_new= torch.tensor(prediction).to(device)

    print(jj, ode_mean_true[jj,:], ode_mean_pred[jj,:], ode_std_true[jj,:], ode_std_pred[jj,:])

empirical_samples = np.array(prediction)
true_samples = np.array(ode_path_true[1,:,:].T)


num_levels = 12

# Create figure and subplots
fig, axs = plt.subplots(1, 2, figsize=(15, 6))
handles = []  # To store legend handles
labels = []   # To store legend labels
for i, samples in enumerate([empirical_samples, true_samples]):
    ax = axs[i]
    # KDE estimation
    kde = gaussian_kde(samples.T)
    xgrid = np.linspace(-0.35, 0.35, 100)
    ygrid = np.linspace(-0.35, 0.35, 100)
    X, Y = np.meshgrid(xgrid, ygrid)
    Z = kde(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)
    if i==0: 
        levels = np.linspace(Z.min(), Z.max(), num_levels)  # Adjust the number of levels as needed
        colors = [
        (1.0, 1.0, 1.0),  # White
        (1.0, 0.75, 0.75),  # Light red
        (1.0, 0.6, 0.6),  # Lighter red
        (1.0, 0.4, 0.4),  # Lightest red
        (1.0, 0.2, 0.2),  # Lightish red
        (1.0, 0.1, 0.1),  # Lightish red (again)
        (1.0, 0.0, 0.0),  # Mid-red
        (0.9, 0.0, 0.0),  # Mid-red (again)
        (0.8, 0.0, 0.0),  # Dark red
        (0.7, 0.0, 0.0),  # Darker red
        (0.6, 0.0, 0.0),  # Darkest red
        (0.5, 0.0, 0.0),  # Darkest red (again)
        ]


        # Define the custom colormap
        cmap_lin_alpha = LinearSegmentedColormap.from_list('custom_colormap', colors, N=num_levels)
        legend = 'learned'
    else:
        levels[0] = Z.min()
        levels[-1] = Z.max()
        colors = [
        (1.0, 1.0, 1.0),  # White
        (0.65, 0.8, 1.0),  # Light blue
        (0.5, 0.7, 1.0),  # Lighter blue
        (0.4, 0.6, 1.0),  # Lightest blue
        (0.3, 0.5, 1.0),  # Lightish blue
        (0.2, 0.4, 1.0),  # Lightish blue (again)
        (0.1, 0.25, 1.0),  # Mid-blue
        (0.0, 0.15, 1.0),  # Mid-blue (again)
        (0.0, 0.05, 0.9),  # Dark blue
        (0.0, 0.0, 0.85),  # Darker blue
        (0.0, 0.0, 0.75),  # Darkest blue
        (0.0, 0.0, 0.65),  # Darkest blue (again)
        ]
        # Define the custom colormap
        cmap_lin_alpha = LinearSegmentedColormap.from_list('custom_colormap', colors, N=num_levels)
        legend = 'reference'

    # Plot contour
    contour_filled = ax.contourf(X, Y, Z, levels=levels, cmap=cmap_lin_alpha)
    # Create legend handles
    legend_handles = [Patch(color=colors[1], label=legend)  ]
    # Add legend to the subplot
    ax.legend(handles=legend_handles, loc='upper right')

    # Remove right y-axis and upper x-axis
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(right=False, top=False)

    # Add marginal histograms pdf
    divider = make_axes_locatable(ax)
    ax_histx = divider.append_axes("top", 1.02, pad=0.1, sharex=ax)
    ax_histy = divider.append_axes("right", 1.02, pad=0.1, sharey=ax)
    
    # Fill between the PDF values and the axis
    ax_histx.fill_between(xgrid, Z.sum(axis=0), color=colors[1], alpha=0.3, clip_on=True)
    ax_histx.plot(xgrid, Z.sum(axis=0), color=colors[1], clip_on=True)
    ax_histx.set_ylim(bottom=0.01)

    ax_histy.fill_betweenx(ygrid, Z.sum(axis=1),color=colors[1], alpha=0.3, clip_on=True)
    ax_histy.plot(Z.sum(axis=1), ygrid, color=colors[1], clip_on=True)
    ax_histy.set_xlim(left=0.01)
    # Ensure the fill goes to the edge of the plot by turning off clipping
    for artist in ax_histx.get_children():
        artist.set_clip_on(False)

    for artist in ax_histy.get_children():
        artist.set_clip_on(False)
    
    # Set y-limits and x-limits to match the data range
    ax_histx.set_xlim(ax.get_xlim())
    ax_histy.set_ylim(ax.get_ylim())                       
    # Remove ticks and labels
    ax_histx.tick_params(left=False, labelleft=False, bottom=True, labelbottom=False)
    ax_histy.tick_params(left=True, labelleft=False, bottom=False, labelbottom=False)


    # Customize the plot
    ax_histx.spines['right'].set_visible(False)
    ax_histx.spines['top'].set_visible(False)
    ax_histx.spines['left'].set_visible(False)
    ax_histx.tick_params(left=False, labelleft=False)
    ax_histy.spines['right'].set_visible(False)
    ax_histy.spines['top'].set_visible(False)
    ax_histy.spines['bottom'].set_visible(False)
    # ax_histy.tick_params(bottom=False, labelbottom=False)


    ax.set_xlim([-0.35, 0.35])
    ax.set_xticks(np.linspace(-0.3, 0.3, 7))
    ax.set_ylim([-0.35, 0.35])
    ax.set_yticks(np.linspace(-0.3, 0.3, 7))
    
# Save and show the plot
plt.savefig(os.path.join(savedir,'contour_plot.png'), dpi=300, bbox_inches='tight')
plt.show()