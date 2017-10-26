require 'hdf5'
require 'nngraph'
require 'torch'
require 'nn'
require 'cunn'
require 'optim'
require 'image'
require 'pl'
require 'paths'
require 'cudnn'
require 'stn'
require 'sys'
ok, disp = pcall(require, 'display')
if not ok then print('display not found. unable to plot') end

-- cannot get so large memory to save hdf5, so we only test every 2048 for 4 times
ntrain = 163904
num = 1024*2
inter_dataset = torch.FloatTensor(num,3,128,128):fill(0)
GPU_ID = 1
cutorch.setDevice(GPU_ID) 

function saveDataset(data, model, start)
  local N = data:size(1)

  local inputs_lr = torch.Tensor(N,3,16,16)
  
  for i = 1,N do
	inputs_lr[i]  = data[i]
  end
  -- Generate 
  
  local samples = model:forward(inputs_lr)
  samples = nn.HardTanh():forward(samples)  
  samples:add(1):mul(0.5)  
  
  --start = start+N
  for i = 1,N do
	local tmp = samples[i]:float()
	inter_dataset[start+i] = tmp:clone()
  end

end

function saveImages(data, model_TAE, model_decoder, foldername,start)
  local N = data:size(1)

  local inputs_hr = torch.Tensor(N,3,128,128)
  
  for i = 1,N do
	inputs_hr[i]  = data[i]
  end
  -- Generate 
  
  --sys.tic()
  local samples = model_TAE:forward(inputs_hr)
  --t = sys.toc()
  --print(t)  
  samples = nn.HardTanh():forward(samples)
  local samples_UR = model_decoder:forward(samples)
  samples_UR = nn.HardTanh():forward(samples_UR)  
  
  local to_plot = {}
  for i = 1,N do
    to_plot[i] = samples_UR[i]:float()
    torch.setdefaulttensortype('torch.FloatTensor')
    local GEN = image.toDisplayTensor({input=to_plot[i], nrow=1})
    --GEN:add(1):div(2):float()
    GEN = GEN:index(1,torch.LongTensor{3,2,1})
    
    filename = string.format("%05d.png",i+start)
    image.save(foldername .. filename, GEN)
  end  

  torch.setdefaulttensortype('torch.CudaTensor') 
  cutorch.setDevice(GPU_ID)  
  
  samples_UR = nn.HardTanh():forward(samples_UR)  
  samples_UR:add(1):mul(0.5)    
  for i = 1,N do
	local tmp = samples_UR[i]:float()
	inter_dataset[start+i] = tmp:clone()
  end   
   
end

torch.setdefaulttensortype('torch.CudaTensor')

--model = torch.load('/media/anu-user1/2TB/xin/Res_GAN/logs128_ytc16_URDGN/adversarial.net')
model = torch.load('CVPR_Model/adversarial.net.old_63')
--torch.setdefaulttensortype('torch.CudaTensor')  -- when using torch.CudaTensor please not use 'image', there will be some bugs
model_STUR = model.G
model_STUR:evaluate()

---------------------------------------------------------------------------------------
-------------------------- first stage             ------------------------------------
---------------------------------------------------------------------------------------

filename_hdf5 = "../dataset/YTC_LR_unalign_30.hdf5"
local lowHd5  = hdf5.open(filename_hdf5, 'r')
local data_LR = lowHd5:read('YTC'):all()
data_LR:mul(2):add(-1)
lowHd5:close()
Data_LR = data_LR[{{ntrain+1,ntrain+num}}]

save_filename = "CVPR_Model/YTC_LR_UR_CVPR.hdf5"
if not paths.filep(save_filename) then
	os.remove(save_filename)
end

num_remainder = num%100
num_loop      = (num-num_remainder)/100
for i = 1,num_loop do
	--sys.tic()
	saveDataset(Data_LR[{{(i-1)*100+1,i*100}}], model_STUR, (i-1)*100)
	--t = sys.toc()
	--print(t)
end
if num_remainder ~= 0 then
	saveDataset(Data_LR[{{num_loop*100+1,num}}], model_STUR, num_loop*100)
end

local inter_hdf5 = hdf5.open(save_filename, 'w')
inter_hdf5:write('YTC', inter_dataset)
inter_hdf5:close()

torch.setdefaulttensortype('torch.CudaTensor') 


model_STUR = nil
model = nil
collectgarbage()

----------------------------------------------------------
----------------------------------------------------------
-- Load TAE
--TAE = torch.load('/media/anu-user1/2TB/xin/Res_GAN/TAE/TAE.net_76')
TAE = torch.load('CVPR_Model/TAE.net')
--torch.setdefaulttensortype('torch.CudaTensor')  -- when using torch.CudaTensor please not use 'image', there will be some bugs
model_TAE = TAE.EN
model_TAE:evaluate()

-- Decoder
--model = torch.load('/media/anu-user1/2TB/xin/Res_GAN/logs128_ytc16_stn_D_decay_noise_free/adversarial.net_124')
--model = torch.load('/media/anu-user1/2TB/xin/Res_GAN/logs128_ytc16_stn_D_decay_v6_noise_01/adversarial.net_70')
--model = torch.load('/media/anu-user1/2TB/xin/Res_GAN/logs128_ytc16_stn_D_decay_TAE_decoder/adversarial.net_55')
model = torch.load('CVPR_Model/adversarial.net.old_151')

model_decoder = model.G
model_decoder:evaluate()
model.D = nil
collectgarbage()

folder = 'CVPR_Model/'
if not paths.dirp(folder) then
	paths.mkdir(folder)
end


filename_hdf5 = "CVPR_Model/YTC_LR_UR_CVPR.hdf5"
local Hd5  = hdf5.open(filename_hdf5, 'r')
local data_HR = Hd5:read('YTC'):all()
data_HR:mul(2):add(-1)
Hd5:close()
Data_HR = data_HR

foldername = string.format("%s/cvpr_results/",folder)
if not paths.dirp(foldername) then
	paths.mkdir(foldername)
end

save_filename = string.format("%s/cvpr_final_results.hdf5",foldername)
if not paths.filep(save_filename) then
	os.remove(save_filename)
end

num_remainder = num%100
num_loop      = (num-num_remainder)/100
for i = 1,num_loop do
	saveImages(Data_HR[{{(i-1)*100+1,i*100}}], model_TAE, model_decoder, foldername, (i-1)*100)
end
if num_remainder ~= 0 then
	saveImages(Data_HR[{{num_loop*100+1,num}}], model_TAE, model_decoder, foldername, num_loop*100)
end

torch.setdefaulttensortype('torch.CudaTensor') 

local inter_hdf5 = hdf5.open(save_filename, 'w')
inter_hdf5:write('YTC', inter_dataset)
inter_hdf5:close()

torch.setdefaulttensortype('torch.CudaTensor') 
