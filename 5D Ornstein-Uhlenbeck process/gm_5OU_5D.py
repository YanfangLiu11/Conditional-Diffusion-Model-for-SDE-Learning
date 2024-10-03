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
        for i in range(n_dim):
            xIC[i, :] = np.random.uniform(-4, 4, N_data)
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
B = np.array([[0.2, 1.0, 0.2, 0.4, 0.2], 
              [-1.0, 0.0, 0.2, 0.8, -1.0],
               [0.2, 0.2, -0.8, -1.2, 0.2],
               [-0.6, 0.0, 1.2, -0.2, 0.6],
               [0.2, 0.2, 0.6, 0.4, 0.0]])
Sigma = np.array([
            [0.8, 0.2, 0.1, -0.3, 0.1],
            [-0.3, 0.6, 0.1, 0.0, -0.1],
            [0.2, -0.1, 0.9, 0.1, 0.2],
            [0.1, 0.1, -0.2, 0.7, 0.0],
            [-0.1, 0.1, 0.1, -0.1, 0.5]])

x_dim = 5

sde_T = 1.0
sde_dt = 0.01
sde_Nt = int(sde_T /sde_dt)
seeds=1
true_init = np.array([0.3, -0.2, -1.7, 2.5, 1.4])

savedir = 'C:\\Users\\yliu1\\OneDrive - Middle Tennessee State University\\cyberduck\\sde_learning\\final_code\\5OU_5D\\'
make_folder(savedir)


N_sample_path = 600000
data_sample = Ornstein_Uhlenbeck_2D(true_init,N_sample_path, sde_T, sde_dt, x_dim,  B, Sigma,seeds,'uniform')
print('size of data_sample is: ', data_sample.shape )
x_sample = (data_sample[:sde_Nt,:,:]).transpose(0, 2, 1).reshape((N_sample_path*sde_Nt,x_dim))
y_sample = (data_sample[1:sde_Nt+1,:,:]).transpose(0, 2, 1).reshape((N_sample_path*sde_Nt,x_dim))
sample_size = x_sample.shape[0]
print('sample size:', sample_size )

diff_scale = 5*np.array([1, 1, 1, 1, 1])
xy_diff = (y_sample-x_sample)*diff_scale
print('difference of x_sample and y_sample is: ', xy_diff)
print('mean and std of xy_diff are: ',  np.mean(xy_diff, axis = 0),np.std(xy_diff, axis=0))

train_size =  50000
selected_row_indices =  np.random.permutation(sample_size)[:train_size]
x0_train = x_sample[selected_row_indices]
print('size of x0_train is: ', x0_train.shape )


short_size = 2048

it_size_x0train = 400
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
n_iter = 4000
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

Npath = 400000

ode_time_steps = 500

ode_mean_pred = np.zeros((ode_time_steps,x_dim))
ode_std_pred = np.zeros((ode_time_steps,x_dim))
ode_mean_true = np.zeros((ode_time_steps,x_dim))
ode_std_true = np.zeros((ode_time_steps,x_dim))

true_init = np.array([0.3, -0.2, -1.7, 2.5, 1.4])
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
# Create subplots for each dimension horizontally
fig, axs = plt.subplots(1, x_dim, figsize=(x_dim * 6, 5))
# Plot data for each dimension
for i in range(x_dim):
    ax = axs[i]
    ax.plot(tmesh, ode_mean_true[:, i])
    ax.plot(tmesh, ode_mean_pred[:, i])
    ax.plot(tmesh, ode_mean_true[:, i] - ode_std_true[:, i])
    ax.plot(tmesh, ode_mean_true[:, i] + ode_std_true[:, i])
    ax.fill_between(tmesh, ode_mean_pred[:, i] - ode_std_pred[:, i], ode_mean_pred[:, i] + ode_std_pred[:, i], alpha=0.2)
    ax.legend(['True mean', 'Pred mean', 'True mean-std', 'True mean+std', 'Pred std'])
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('Value', fontsize=12)
    # ax.set_title(f'$x_{i+1}$ = {true_init[i]:.2f}', fontsize=12)
    ax.tick_params(axis='both', labelsize=10)

# Adjust layout and spacing
plt.tight_layout()

# Save or display the figure
plt.savefig(os.path.join(savedir, f'example1_final_image_{true_init[0]:.2f}_N{ode_time_steps}.png'), dpi=300, bbox_inches='tight')
plt.show()





ode_mean_pred = np.zeros((ode_time_steps,x_dim))
ode_std_pred = np.zeros((ode_time_steps,x_dim))
ode_mean_true = np.zeros((ode_time_steps,x_dim))
ode_std_true = np.zeros((ode_time_steps,x_dim))

true_init = np.array([-0.5, 2.5, -0.2, -1.0, 1.4])
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

# Create subplots for each dimension horizontally
fig, axs = plt.subplots(1, x_dim, figsize=(x_dim * 6, 5))
# Plot data for each dimension
for i in range(x_dim):
    ax = axs[i]
    ax.plot(tmesh, ode_mean_true[:, i])
    ax.plot(tmesh, ode_mean_pred[:, i])
    ax.plot(tmesh, ode_mean_true[:, i] - ode_std_true[:, i])
    ax.plot(tmesh, ode_mean_true[:, i] + ode_std_true[:, i])
    ax.fill_between(tmesh, ode_mean_pred[:, i] - ode_std_pred[:, i], ode_mean_pred[:, i] + ode_std_pred[:, i], alpha=0.2)
    ax.legend(['True mean', 'Pred mean', 'True mean-std', 'True mean+std', 'Pred std'])
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('Value', fontsize=12)
    # ax.set_title(f'$x_{i+1}$ = {true_init[i]:.2f}', fontsize=12)
    ax.tick_params(axis='both', labelsize=10)

# Adjust layout and spacing
plt.tight_layout()

# Save or display the figure
plt.savefig(os.path.join(savedir, f'example2_final_image_{true_init[0]:.2f}_N{ode_time_steps}.png'), dpi=300, bbox_inches='tight')
plt.show()


# ======================================
#  check conditional distribution

Npath = 400000

ode_time_steps = 1

true_init = np.array([0.3, -0.2, -1.7, 2.5, 1.4])
x_pred_new = torch.ones(Npath, x_dim).to(device,dtype= torch.float32) * torch.tensor(true_init).to(device,dtype= torch.float32)
prediction = FN( (torch.hstack((x_pred_new,torch.randn(Npath,x_dim).to(device,dtype= torch.float32)))-xTrain_mean)/xTrain_std  ) * yTrain_std + yTrain_mean 
prediction = ( prediction.to('cpu').detach().numpy() /diff_scale +x_pred_new.to('cpu').detach().numpy()  )

ode_path_true = Ornstein_Uhlenbeck_2D(true_init,Npath, ode_time_steps*sde_dt, sde_dt, x_dim,  B, Sigma,seeds,'value')
true_samples =  (ode_path_true[1,:,:]).T
print('true sample shape: ',true_samples.shape)



# Create subplots for each dimension
fig, axes = plt.subplots(1, x_dim, figsize=(x_dim* 6, 5))

# Iterate over each dimension
for i in range(x_dim):
    ax = axes[i] if x_dim > 1 else axes  # Handle the case when num_dimensions is 1
    
    # Plot histogram with filled color for the current dimension
    counts, bins, _ = ax.hist(prediction[:, i], bins=50, density=True, alpha=0.5, color=(0.625, 0.625, 0.625), histtype='stepfilled', edgecolor='black')

    # Compute kernel density estimation for the current dimension
    kde = gaussian_kde(true_samples[:, i].T)
    if i ==0:
        
        x_vals = np.linspace(-0.1, 0.7, 100)
    elif  i ==1:
        
        x_vals = np.linspace(-0.5, -0.1, 100)
    elif  i ==2:
        
        x_vals = np.linspace(-2.1, -1.3, 100)
    elif  i ==3:
       
        x_vals = np.linspace(2.2, 2.8, 100)
    else:

        x_vals = np.linspace(1.2, 1.6, 100)
    pdf_vals = kde(x_vals)

    # Plot the kernel density estimate
    ax.plot(x_vals, pdf_vals, color='blue', linewidth=1.4)

    ax.set_title(r'$X^{(' + str(i+1) + ')}$', fontsize=16)
    # ax.set_ylabel('pdf', fontsize=16)
    # ax.set_title(f'Marginal Density for Dimension {i+1}', fontsize=16)
    
    # Increase font size of tick labels
    ax.tick_params(axis='both', labelsize=14)

# Adjust layout to prevent overlapping
plt.tight_layout()

# Save and show the plot
plt.savefig(os.path.join(savedir, 'marginal_density_functions.png'), dpi=300, bbox_inches='tight')
plt.show()

