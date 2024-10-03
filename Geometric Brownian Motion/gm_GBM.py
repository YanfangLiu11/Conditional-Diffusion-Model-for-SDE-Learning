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


def Gendata(dt,sde_path, interval_start,interval_end):
    dT=1.0
    N_steps = int(dT/dt)

    # cut_value = 15
    x0 = torch.log(interval_start+ (interval_end-interval_start)*torch.rand( sde_path)).reshape(-1,1).to(device)
    # plt.figure()
    # plt.hist(x0[:,0].to('cpu'),bins=400)
    # plt.savefig(os.path.join(savedir, 'x0.png'), dpi=300)
   
    raw_data = torch.zeros(sde_path,N_steps+1).to(device)
    raw_data[:,0] = x0[:,0]
    for j in range(N_steps):
        raw_data[:,j+1] = raw_data[:,j] + (mu-0.5*sigma**2)*dt+sigma*np.sqrt(dt)*torch.randn(sde_path).to(device)  
    x_start = raw_data[:,:N_steps].reshape(-1,1)
    x_end = raw_data[:,1:N_steps+1].reshape(-1,1)
    # plt.plot(t_gen.T.to('cpu'),raw_data.T.to('cpu'))
    # plt.savefig(os.path.join(savedir, 'raw_data.png'), dpi=300)
    # plt.show()
    
    return x_start,x_end
#================================================
interval_a = 0.0
interval_b = 6.0
mu = 2.0 
sigma = 1.0

sde_T = 0.5
sde_dt = 0.01
sde_Nt = int(sde_T /sde_dt)

x_dim = 1

savedir = 'C:\\Users\\yliu1\\OneDrive - Middle Tennessee State University\\cyberduck\\sde_learning\\final_code\\GBM\\'
make_folder(savedir)


N_sample_path = 100000
x_sample,y_sample = Gendata(sde_dt, N_sample_path, interval_a, interval_b)



x_select_indx =(x_sample<=3)
x_sample = x_sample[x_select_indx]
y_sample = y_sample[x_select_indx]

x_sample = x_sample.to('cpu').detach().numpy()
y_sample = y_sample.to('cpu').detach().numpy()

x_sample = x_sample.reshape(-1,1)
y_sample = y_sample.reshape(-1,1)

plt.figure()
plt.hist(x_sample,bins=400)
plt.savefig(os.path.join(savedir, 'xsample.png'), dpi=300)

diff_scale = 2
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
optimizer = optim.Adam(FN.parameters(), lr = learning_rate, weight_decay=1e-7)
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


#======================================================
# Result of NN
#====================================================== 
Npath = 400000

ode_time_steps = 100
ode_mean_pred = np.zeros(ode_time_steps)
ode_std_pred = np.zeros(ode_time_steps)
ode_mean_true = np.zeros(ode_time_steps)
ode_std_true = np.zeros(ode_time_steps)

true_init = 0.3
x_pred_new= torch.clone((np.log(true_init)*torch.ones(Npath,x_dim)).to(device))
ode_path_true = true_init*np.ones((Npath,x_dim))
for jj in range(ode_time_steps):
    
    prediction = FN( (torch.hstack((x_pred_new,torch.randn(Npath,x_dim).to(device,dtype= torch.float32)))-xTrain_mean)/xTrain_std  ) * yTrain_std + yTrain_mean 
    prediction = ( prediction/diff_scale +x_pred_new ).to('cpu').detach().numpy() 
    ode_mean_pred[jj] = np.mean(np.exp(prediction))
    ode_std_pred[jj] = np.std(np.exp(prediction))
    
    x_pred_new= torch.tensor( prediction  ).to(device,dtype= torch.float32)
    
    ode_path_true = ode_path_true*np.exp( (mu-0.5*sigma**2)*sde_dt+sigma * np.random.normal(0, np.sqrt(sde_dt), size=(Npath,1)))
    ode_mean_true[jj] = np.mean(ode_path_true)
    ode_std_true[jj] = np.std(ode_path_true)    

    print(jj, ode_mean_true[jj], ode_mean_pred[jj], ode_std_true[jj], ode_std_pred[jj])


tmesh = np.linspace(sde_dt, ode_time_steps*sde_dt, ode_time_steps)

fig, ax = plt.subplots(1, 1, figsize=(10, 6))

ax.plot(tmesh, ode_mean_true)
ax.plot(tmesh, ode_mean_pred)
ax.plot(tmesh, ode_mean_true - ode_std_true)
ax.plot(tmesh, ode_mean_true + ode_std_true)
ax.fill_between(tmesh, ode_mean_pred - ode_std_pred, ode_mean_pred + ode_std_pred, alpha=0.2)
ax.legend(['True mean', 'Pred mean', 'True mean-std', 'True mean+std', 'Pred std'])

ax.set_xlabel('Time', fontsize=16)
ax.set_ylabel('Value', fontsize=16)
ax.set_title(f'$x_0$ = {true_init:.2f}' , fontsize=16)

# Increase font size of tick labels
ax.tick_params(axis='both', labelsize=14)

plt.savefig(os.path.join(savedir, f'final_image_{true_init:.2f}_N{ode_time_steps}.png'), dpi=300, bbox_inches='tight')
plt.show()




ode_mean_pred = np.zeros(ode_time_steps)
ode_std_pred = np.zeros(ode_time_steps)
ode_mean_true = np.zeros(ode_time_steps)
ode_std_true = np.zeros(ode_time_steps)

true_init = 0.8
x_pred_new= torch.clone((np.log(true_init)*torch.ones(Npath,x_dim)).to(device))
ode_path_true = true_init*np.ones((Npath,x_dim))
for jj in range(ode_time_steps):
    
    prediction = FN( (torch.hstack((x_pred_new,torch.randn(Npath,x_dim).to(device,dtype= torch.float32)))-xTrain_mean)/xTrain_std  ) * yTrain_std + yTrain_mean 
    prediction = ( prediction/diff_scale +x_pred_new ).to('cpu').detach().numpy() 
    ode_mean_pred[jj] = np.mean(np.exp(prediction))
    ode_std_pred[jj] = np.std(np.exp(prediction))
    
    x_pred_new= torch.tensor( prediction  ).to(device,dtype= torch.float32)
    
    ode_path_true = ode_path_true*np.exp( (mu-0.5*sigma**2)*sde_dt+sigma * np.random.normal(0, np.sqrt(sde_dt), size=(Npath,1)))
    ode_mean_true[jj] = np.mean(ode_path_true)
    ode_std_true[jj] = np.std(ode_path_true)    

    print(jj, ode_mean_true[jj], ode_mean_pred[jj], ode_std_true[jj], ode_std_pred[jj])


fig, ax = plt.subplots(1, 1, figsize=(10, 6))

ax.plot(tmesh, ode_mean_true)
ax.plot(tmesh, ode_mean_pred)
ax.plot(tmesh, ode_mean_true - ode_std_true)
ax.plot(tmesh, ode_mean_true + ode_std_true)
ax.fill_between(tmesh, ode_mean_pred - ode_std_pred, ode_mean_pred + ode_std_pred, alpha=0.2)
ax.legend(['True mean', 'Pred mean', 'True mean-std', 'True mean+std', 'Pred std'])

ax.set_xlabel('Time', fontsize=16)
ax.set_ylabel('Value', fontsize=16)
ax.set_title(f'$x_0$ = {true_init:.2f}' , fontsize=16)

# Increase font size of tick labels
ax.tick_params(axis='both', labelsize=14)

plt.savefig(os.path.join(savedir, f'final_image_{true_init:.2f}_N{ode_time_steps}.png'), dpi=300, bbox_inches='tight')
plt.show()



ode_mean_pred = np.zeros(ode_time_steps)
ode_std_pred = np.zeros(ode_time_steps)
ode_mean_true = np.zeros(ode_time_steps)
ode_std_true = np.zeros(ode_time_steps)

true_init = 0.5
x_pred_new= torch.clone((np.log(true_init)*torch.ones(Npath,x_dim)).to(device))
ode_path_true = true_init*np.ones((Npath,x_dim))
for jj in range(ode_time_steps):
    
    prediction = FN( (torch.hstack((x_pred_new,torch.randn(Npath,x_dim).to(device,dtype= torch.float32)))-xTrain_mean)/xTrain_std  ) * yTrain_std + yTrain_mean 
    prediction = ( prediction/diff_scale +x_pred_new ).to('cpu').detach().numpy() 
    ode_mean_pred[jj] = np.mean(np.exp(prediction))
    ode_std_pred[jj] = np.std(np.exp(prediction))
    
    x_pred_new= torch.tensor( prediction  ).to(device,dtype= torch.float32)
    
    ode_path_true = ode_path_true*np.exp( (mu-0.5*sigma**2)*sde_dt+sigma * np.random.normal(0, np.sqrt(sde_dt), size=(Npath,1)))
    ode_mean_true[jj] = np.mean(ode_path_true)
    ode_std_true[jj] = np.std(ode_path_true)    

    print(jj, ode_mean_true[jj], ode_mean_pred[jj], ode_std_true[jj], ode_std_pred[jj])

fig, ax = plt.subplots(1, 1, figsize=(10, 6))

ax.plot(tmesh, ode_mean_true)
ax.plot(tmesh, ode_mean_pred)
ax.plot(tmesh, ode_mean_true - ode_std_true)
ax.plot(tmesh, ode_mean_true + ode_std_true)
ax.fill_between(tmesh, ode_mean_pred - ode_std_pred, ode_mean_pred + ode_std_pred, alpha=0.2)
ax.legend(['True mean', 'Pred mean', 'True mean-std', 'True mean+std', 'Pred std'])

ax.set_xlabel('Time', fontsize=16)
ax.set_ylabel('Value', fontsize=16)
ax.set_title(f'$x_0$ = {true_init:.2f}', fontsize=16)

# Increase font size of tick labels
ax.tick_params(axis='both', labelsize=14)

plt.savefig(os.path.join(savedir, f'final_image_{true_init:.2f}_N{ode_time_steps}.png'), dpi=300, bbox_inches='tight')
plt.show()





# ======================================
#  check drift and diffusion function

Npath = 500000
N_x0= 1400
x_min = 0.5
x_max = 14.5
x0_grid=np.linspace(x_min,x_max,N_x0)
bx_pred = np.zeros(N_x0)
sigmax_pred = np.zeros(N_x0)
bx_true = 2*x0_grid
sigmax_true = x0_grid

for jj in range(N_x0):
    true_init = x0_grid[jj]
    x_pred_new= torch.clone((np.log(true_init)*torch.ones(Npath,x_dim)).to(device))

    prediction = FN( (torch.hstack((x_pred_new,torch.randn(Npath,x_dim).to(device,dtype= torch.float32)))-xTrain_mean)/xTrain_std  ) * yTrain_std + yTrain_mean 
    prediction = ( prediction/diff_scale +x_pred_new ).to('cpu').detach().numpy() 
        
    bx_pred[jj] = np.mean(( np.exp(prediction)-true_init)/sde_dt)
    sigmax_pred[jj] = np.std(( np.exp(prediction)-true_init-bx_pred[jj]*sde_dt))*np.sqrt(1/sde_dt)
    

    print(jj, bx_true[jj], bx_pred[jj], sigmax_true[jj], sigmax_pred[jj])


fig, ax = plt.subplots(1, 2, figsize=(15, 6))

ax[0].plot(x0_grid, bx_true, label = 'referenece')
ax[0].plot(x0_grid, bx_pred, label = 'learned') 
ax[0].set_xlabel('X', fontsize=16)
ax[0].set_ylabel('b(x)', fontsize=16)  
ax[0].legend(fontsize=14)  
ax[0].tick_params(axis='both', labelsize=14)

ax[1].plot(x0_grid, sigmax_true, label = 'referenece')
ax[1].plot(x0_grid, sigmax_pred, label = 'learned')
ax[1].set_xlabel('X', fontsize=16)
ax[1].set_ylabel('$\sigma(x)$', fontsize=16)
ax[1].legend(fontsize=14)  
ax[1].tick_params(axis='both', labelsize=14)  
# Increase font size of tick labels


plt.savefig(os.path.join(savedir, 'drift and diffusion.png'), dpi=300, bbox_inches='tight')
plt.show()




# ======================================
#  check drift and diffusion function

Npath = 500000
N_x0= 1400
x_min = 0.5
x_max = 14.5
x0_grid=np.linspace(x_min,x_max,N_x0)
bx_pred = np.zeros(N_x0)
sigmax_pred = np.zeros(N_x0)
bx_true = 2*x0_grid
sigmax_true = x0_grid

for jj in range(N_x0):
    true_init = x0_grid[jj]
    x_pred_new= torch.clone((np.log(true_init)*torch.ones(Npath,x_dim)).to(device))

    prediction = FN( (torch.hstack((x_pred_new,torch.randn(Npath,x_dim).to(device,dtype= torch.float32)))-xTrain_mean)/xTrain_std  ) * yTrain_std + yTrain_mean 
    prediction = ( prediction/diff_scale +x_pred_new ).to('cpu').detach().numpy() 
        
    bx_pred[jj] = np.mean(( np.exp(prediction)-true_init)/sde_dt)
    sigmax_pred[jj] = np.std(( np.exp(prediction)-true_init-bx_pred[jj]*sde_dt))*np.sqrt(1/sde_dt)
    

    print(jj, bx_true[jj], bx_pred[jj], sigmax_true[jj], sigmax_pred[jj])


fig, ax = plt.subplots(1, 2, figsize=(15, 6))

ax[0].plot(x0_grid, bx_true, label = 'referenece')
ax[0].plot(x0_grid, bx_pred, label = 'learned') 
ax[0].set_xlabel('X', fontsize=16)
ax[0].set_ylabel('b(x)', fontsize=16)  
ax[0].legend(fontsize=14)  
ax[0].tick_params(axis='both', labelsize=14)

ax[1].plot(x0_grid, sigmax_true, label = 'referenece')
ax[1].plot(x0_grid, sigmax_pred, label = 'learned')
ax[1].set_xlabel('X', fontsize=16)
ax[1].set_ylabel('$\sigma(x)$', fontsize=16)
ax[1].legend(fontsize=14)  
ax[1].tick_params(axis='both', labelsize=14)  
# Increase font size of tick labels


plt.savefig(os.path.join(savedir, 'drift and diffusion.png'), dpi=300, bbox_inches='tight')
plt.show()




# ======================================
#  check conditional distribution
Npath = 500000
true_init = 5
ode_time_steps=1
x_pred_new= torch.clone((np.log(true_init)*torch.ones(Npath,x_dim)).to(device))

prediction = FN( (torch.hstack((x_pred_new,torch.randn(Npath,x_dim).to(device,dtype= torch.float32)))-xTrain_mean)/xTrain_std  ) * yTrain_std + yTrain_mean 
prediction = ( prediction/diff_scale +x_pred_new ).to('cpu').detach().numpy() 
        
true_samples = true_init*np.exp( (mu-0.5*sigma**2)*sde_dt+sigma * np.random.normal(0, np.sqrt(sde_dt), size=(Npath,1)))
print('true sample shape: ',true_samples.shape)

fig, ax = plt.subplots(1, 1, figsize=(10, 6))

# Plot histogram with filled color
counts, bins, _ = ax.hist(np.exp(prediction), bins=50, density=True, alpha=0.5, color=(0.625, 0.625, 0.625), histtype='stepfilled', edgecolor='black')
# ax.hist(true_samples, bins=60, density=True, alpha=0.5, color=(0.625, 0.625, 0.625), histtype='stepfilled', edgecolor='red')
print(counts.max())

n_vals = 800
x_min = 3
x_max = 7
x_vals = np.linspace(x_min,x_max, n_vals)
kde = gaussian_kde(true_samples.T)
pdf_vals = kde(x_vals)

ax.plot(x_vals, pdf_vals, color='blue', linewidth=1.4)

ax.legend([ 'Reference', 'Learned'])

ax.set_xlabel('X', fontsize=16)
ax.set_ylabel('pdf', fontsize=16)
ax.set_title(f'$x_0$ = {true_init:.2f}', fontsize=16)
ax.set_xlim([x_min-0.5,x_max+0.5])
ax.set_xticks(np.linspace(x_min,x_max,5))
# Increase font size of tick labels
ax.tick_params(axis='both', labelsize=14)

plt.savefig(os.path.join(savedir, f'pdf_final_image_{true_init:.2f}.png'), dpi=300, bbox_inches='tight')
plt.show()


