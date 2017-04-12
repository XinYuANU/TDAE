------------------------------------------------------------
--- This code is based on the eyescream code released at
--- https://github.com/facebook/eyescream
--- If you find it usefull consider citing
--- http://arxiv.org/abs/1506.05751
------------------------------------------------------------

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
--require 'cleanmodel'
stn_L0 = require 'stn_L0_TAE'
stn_L1 = require 'stn_L1_TAE'
TAE    = require 'TAE_org'

ok, disp = pcall(require, 'display')
if not ok then print('display not found. unable to plot') end

----------------------------------------------------------------------
-- parse command-line options
opt = lapp[[
  -s,--save          (default "TAE_org")      subdirectory to save logs
  --saveFreq         (default 1)          save every saveFreq epochs
  -n,--network       (default "")          reload pretrained network
  -p,--plot                                plot while training
  -r,--learningRate  (default 0.001)       learning rate
  -b,--batchSize     (default 32)          batch size
  -m,--momentum      (default 0)           momentum, for SGD only
  -g,--gpu           (default 1)           gpu to run on (default cpu)    
]]


if opt.gpu < 0 or opt.gpu > 3 then opt.gpu = false end

print(opt)

ntrain = 29952
nval   = 1024

local resHd5 = hdf5.open('datasets/YTC_HR_intermediate.hdf5', 'r')
local data_Res = resHd5:read('YTC'):all()
data_Res:mul(2):add(-1)
resHd5:close()
trainData_Res = data_Res[{{1, ntrain}}]
valData_Res = data_Res[{{ntrain+nval+1, 2*nval+ntrain}}]

local highHd5 = hdf5.open('datasets/YTC_HR_rotate_v5.hdf5', 'r')
local data_HR = highHd5:read('YTC'):all()
data_HR:mul(2):add(-1)
highHd5:close()
trainData_HR = data_HR[{{1, ntrain}}]
valData_HR = data_HR[{{ntrain+nval+1, 2*nval+ntrain}}]

local lowHd5 = hdf5.open('datasets/YTC_LR_front.hdf5', 'r')
local data_LR = lowHd5:read('YTC'):all()
data_LR:mul(2):add(-1)
lowHd5:close()
trainData_LR = data_LR[{{1, ntrain}}]
valData_LR = data_LR[{{ntrain+nval+1, 2*nval+ntrain}}]

if opt.gpu then
  cutorch.setDevice(opt.gpu + 1)
  print('<gpu> using device ' .. opt.gpu)
  torch.setdefaulttensortype('torch.CudaTensor')
else
  torch.setdefaulttensortype('torch.FloatTensor')
end

--[[ load UR model
tmp = torch.load('logs128_ytc16_stn_D_decay_v6_noise_01/adversarial.net')
model_G = tmp.G
tmp.D = nil
collectgarbage()
print(model_G)
--model_G = cleanmodel(model_G)
model_G:evaluate()
--local inputs = torch.Tensor(2,3,16,16):fill(0)
--inputs[1] = valData_LR[1]:clone()
--inputs[2] = valData_LR[2]:clone()
--local samples = model_G:forward(inputs:cuda())
--getSamples_v1(valData_LR, 2)
-- fix seed
torch.manualSeed(1)
--]]

torch.manualSeed(1)
input_scale  = 128

if opt.network == '' then
	model_EN = nn.Sequential()
		
	model_EN:add(cudnn.SpatialConvolution(3, 32, 5, 5, 1, 1, 2, 2))
	model_EN:add(cudnn.SpatialBatchNormalization(32))
	model_EN:add(cudnn.ReLU(true))
	model_EN:add(cudnn.SpatialMaxPooling(2,2))                    --- 64
  
	model_EN:add(cudnn.SpatialConvolution(32, 64, 5, 5, 1, 1, 2, 2))
	model_EN:add(cudnn.SpatialBatchNormalization(64))
	model_EN:add(cudnn.ReLU(true))
	model_EN:add(cudnn.SpatialMaxPooling(2,2))                     ---32
	 
	model_EN:add(cudnn.SpatialConvolution(64, 128, 3, 3, 1, 1, 1, 1))
	model_EN:add(cudnn.SpatialBatchNormalization(128))
	model_EN:add(cudnn.ReLU(true))
	model_EN:add(stn_L1)
	model_EN:add(cudnn.SpatialMaxPooling(2,2))					  ---16
	
	model_EN:add(cudnn.SpatialConvolution(128, 256, 3, 3, 1, 1, 1, 1))  
	model_EN:add(cudnn.SpatialBatchNormalization(256))
	model_EN:add(cudnn.ReLU(true))
	model_EN:add(stn_L0)
	model_EN:add(cudnn.SpatialConvolution(256, 3, 3, 3, 1, 1, 1, 1))
  
  ----------------------------------------------------------------------
else
  print('<trainer> reloading previously trained network: ' .. opt.network)
  tmp = torch.load(opt.network)
  model_EN = tmp.EN  
end

if opt.gpu then
  print('Copy model to gpu')
  model_EN:cuda()  
end

-- loss function: negative log-likelihood
criterion_EN = nn.MSECriterion()

-- retrieve parameters and gradients
parameters_EN,gradParameters_EN = model_EN:getParameters()

-- print networks
print(model_EN)

-- log results to files
trainLogger = optim.Logger(paths.concat(opt.save, 'train.log'))
--testLogger = optim.Logger(paths.concat(opt.save, 'test.log'))

-- Training parameters
sgdState_EN = {
  learningRate = opt.learningRate,
  momentum = opt.momentum,
  optimize=true,
  numUpdates = 0
}

-- Get examples to plot
function getSamples(dataset, N)
	local N = N or 10
	local dataset_HR = dataset
	local inputs   = torch.Tensor(N,3,128,128)
	for i = 1,N do 
		--idx = math.random(nval)
		--inputs[i] = image.scale(torch.squeeze(dataset_HR[i]),16,16)
		inputs[i] = dataset_HR[i]:clone()
	end
	
	local samples = model_EN:forward(inputs)
	samples = nn.HardTanh():forward(samples)
	
	local samples_UR = model_G:forward(samples)
	samples_UR = nn.HardTanh():forward(samples_UR)
	
	local to_plot = {}
	for i = 1,N do 
--		to_plot[#to_plot+1] = samples[i]:float()
		to_plot[#to_plot+1] = samples_UR[i]:float()
	end
	return to_plot
end

function getSamples_org(dataset, N)
	local N = N or 10
	local dataset_HR = dataset
	local inputs   = torch.Tensor(N,3,128,128)
	for i = 1,N do 
		--idx = math.random(nval)
		--inputs[i] = image.scale(torch.squeeze(dataset_HR[i]),16,16)
		inputs[i] = dataset_HR[i]:clone()
	end
	
	local samples = model_EN:forward(inputs)
	samples = nn.HardTanh():forward(samples)
	
	
	local to_plot = {}
	for i = 1,N do 
		to_plot[#to_plot+1] = samples[i]:float()
--		to_plot[#to_plot+1] = samples_UR[i]:float()
	end
	return to_plot
end

while true do 
--for ii = 1,68 do
	local to_plot = getSamples_org(valData_Res,100)
	torch.setdefaulttensortype('torch.FloatTensor')
	
	trainLogger:style{['MSE accuarcy TAE'] = '-'}
	trainLogger:plot()
	
	local formatted = image.toDisplayTensor({input=to_plot, nrow=10})
	formatted:float()
	formatted = formatted:index(1, torch.LongTensor{3,2,1})
	image.save(opt.save .. '/TAE_example_' .. (epoch or 0) .. '.png', formatted)
	
	IDX = torch.randperm(ntrain)
	
	if opt.gpu then 
		torch.setdefaulttensortype('torch.CudaTensor')
	else
		torch.setdefaulttensortype('torch.FloatTensor')
	end
	
	TAE.train(trainData_Res,trainData_HR,trainData_LR)
	
	sgdState_EN.momentum = math.min(sgdState_EN.momentum + 0.0008, 0.7)
    sgdState_EN.learningRate = math.max(opt.learningRate*0.99^epoch, 0.000001)
	
end

