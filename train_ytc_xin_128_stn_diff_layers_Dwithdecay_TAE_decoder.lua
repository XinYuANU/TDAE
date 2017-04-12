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
ok, disp = pcall(require, 'display')
if not ok then print('display not found. unable to plot') end
--adversarial = require 'adverserial_xin_v1_D_revise'
URnet = require 'adverserial_xin_v1_D_revise'
stn_L1 = require 'stn_L1_org'
stn_L2 = require 'stn_L2_org'

----------------------------------------------------------------------
-- parse command-line options
opt = lapp[[
  -s,--save          (default "logs128_ytc16_stn_D_decay_TAE_decoder")      subdirectory to save logs
  --saveFreq         (default 1)          save every saveFreq epochs
  -n,--network       (default "")          reload pretrained network
  -p,--plot                                plot while training
  -r,--learningRate  (default 0.001)        learning rate
  -b,--batchSize     (default 64)         batch size
  -m,--momentum      (default 0)           momentum, for SGD only
  --coefL1           (default 0)           L1 penalty on the weights
  --coefL2           (default 0)           L2 penalty on the weights
  -t,--threads       (default 4)           number of threads
  -g,--gpu           (default 0)           gpu to run on (default cpu)
  -d,--noiseDim      (default 512)         dimensionality of noise vector
  --K                (default 1)           number of iterations to optimize D for
  -w, --window       (default 3)           windsow id of sample image
  --scale            (default 128)          scale of images to train on
  --lambda           (default 0.01)       trade off D and Euclidean distance 
  --margin           (default 0.3)        trade off D and G   
]]


if opt.gpu < 0 or opt.gpu > 3 then opt.gpu = false end

print(opt)

ntrain = 29952
nval   = 1024

local highHd5 = hdf5.open('datasets/YTC_HR_rotate_v5.hdf5', 'r')
local data_HR = highHd5:read('YTC'):all()
data_HR:mul(2):add(-1)
highHd5:close()
trainData_HR = data_HR[{{1, ntrain}}]
valData_HR = data_HR[{{ntrain+nval+1, 2*nval+ntrain}}]

-- output of TAE, generate from the training dataset by TAE
local lowHd5 = hdf5.open('datasets/YTC_TAE_intermediate.hdf5', 'r')
local data_LR = lowHd5:read('YTC'):all()
data_LR:mul(2):add(-1)
lowHd5:close()
trainData_LR = data_LR[{{1, ntrain}}]
valData_LR = data_LR[{{ntrain+nval+1, 2*nval+ntrain}}]

-- fix seed
torch.manualSeed(1)

-- threads
torch.setnumthreads(opt.threads)
print('<torch> set nb of threads to ' .. torch.getnumthreads())

if opt.gpu then
  cutorch.setDevice(opt.gpu + 1)
  print('<gpu> using device ' .. opt.gpu)
  torch.setdefaulttensortype('torch.CudaTensor')
else
  torch.setdefaulttensortype('torch.FloatTensor')
end

input_scale  = 16
opt.scale = valData_HR:size(4)
print(opt.scale)
opt.geometry = {3, opt.scale, opt.scale}

local input_sz = opt.geometry[1] * opt.geometry[2] * opt.geometry[3]

if opt.network == '' then
  
  model_D = nn.Sequential()
  model_D:add(cudnn.SpatialConvolution(3, 32, 5, 5, 1, 1, 2, 2))
  model_D:add(cudnn.SpatialMaxPooling(2,2))
  model_D:add(cudnn.ReLU(true))
  model_D:add(nn.SpatialDropout(0.2))  
  model_D:add(cudnn.SpatialConvolution(32, 64, 5, 5, 1, 1, 2, 2))
  model_D:add(cudnn.SpatialMaxPooling(2,2))
  model_D:add(cudnn.ReLU(true))
  model_D:add(nn.SpatialDropout(0.2))
  model_D:add(cudnn.SpatialConvolution(64, 128, 5, 5, 1, 1, 2, 2))
  model_D:add(cudnn.SpatialMaxPooling(2,2))
  model_D:add(cudnn.ReLU(true))
  model_D:add(nn.SpatialDropout(0.2))
  model_D:add(cudnn.SpatialConvolution(128, 96, 5, 5, 1, 1, 2, 2))
  model_D:add(cudnn.ReLU(true))
  model_D:add(cudnn.SpatialMaxPooling(2,2))
  model_D:add(nn.SpatialDropout(0.2))
  model_D:add(nn.Reshape(8*8*96))
  model_D:add(nn.Linear(8*8*96, 1024))
  model_D:add(cudnn.ReLU(true))
  model_D:add(nn.Dropout())
  model_D:add(nn.Linear(1024,1))
  model_D:add(nn.Sigmoid())
  ----------------------------------------------------------------------
  model_G = nn.Sequential()
  model_G:add(cudnn.SpatialConvolution(3, 512, 3, 3, 1, 1, 1, 1))
  model_G:add(cudnn.SpatialBatchNormalization(512))
  model_G:add(cudnn.ReLU(true))  
  model_G:add(nn.SpatialUpSamplingNearest(2))  
  model_G:add(cudnn.SpatialConvolution(512,256, 3, 3, 1, 1, 1, 1))
  model_G:add(cudnn.SpatialBatchNormalization(256))
  model_G:add(cudnn.ReLU(true))  
  model_G:add(nn.SpatialUpSamplingNearest(2))
  model_G:add(cudnn.SpatialConvolution(256, 128, 5, 5, 1, 1, 2, 2))
  model_G:add(cudnn.SpatialBatchNormalization(128))
  model_G:add(cudnn.ReLU(true))
  model_G:add(nn.SpatialUpSamplingNearest(2))  
  model_G:add(cudnn.SpatialConvolution(128, 64, 5, 5, 1, 1, 2, 2))  
  model_G:add(cudnn.SpatialBatchNormalization(64))
  model_G:add(cudnn.ReLU(true))

  model_G:add(cudnn.SpatialConvolution(64, 3, 5, 5, 1, 1, 2, 2))


else
  print('<trainer> reloading previously trained network: ' .. opt.network)
  tmp = torch.load(opt.network)
  model_D = tmp.D  
  model_G = tmp.G
end

model_G:cuda()  -- convert model to CUDA

-- loss function: negative log-likelihood
criterion_D = nn.BCECriterion()
criterion_G = nn.MSECriterion()

-- retrieve parameters and gradients
parameters_D,gradParameters_D = model_D:getParameters()
parameters_G,gradParameters_G = model_G:getParameters()

-- print networks
print('Discriminator network:')
print(model_D)
print('Generator network:')
print(model_G)

-- log results to files
trainLogger = optim.Logger(paths.concat(opt.save, 'train.log'))
--testLogger = optim.Logger(paths.concat(opt.save, 'test.log'))

if opt.gpu then
  print('Copy model to gpu')
  model_D:cuda()  
  model_G:cuda()
end

-- Training parameters
sgdState_D = {
  learningRate = opt.learningRate,
  momentum = opt.momentum,
  optimize=true,
  numUpdates = 0
}
sgdState_G = {
  learningRate = opt.learningRate,
  momentum = opt.momentum,
  optimize=true,
  numUpdates=0
}

-- Get examples to plot
function getSamples(dataset, N)
	local N = N or 10
	local dataset_HR = dataset
	local inputs   = torch.Tensor(N,3,16,16)
	for i = 1,N do 
		--idx = math.random(nval)
		--inputs[i] = image.scale(torch.squeeze(dataset_HR[i]),16,16)
		inputs[i] = dataset[i]
	end
	
	local samples = model_G:forward(inputs)
	local samples = nn.HardTanh():forward(samples)
	local to_plot = {}
	for i = 1,N do 
		to_plot[#to_plot+1] = samples[i]:float()
	end
	return to_plot
end

--while true do 
for ii = 1,200 do
	local to_plot = getSamples(valData_LR,100)
	torch.setdefaulttensortype('torch.FloatTensor')
	
	trainLogger:style{['MSE accuarcy1'] = '-'}
	-- trainLogger:style{['MSE accuarcy'] = '-'}
	trainLogger:plot()
	
	local formatted = image.toDisplayTensor({input = to_plot, nrow = 10})
	formatted:float()
	formatted = formatted:index(1,torch.LongTensor{3,2,1})
	
	image.save(opt.save .. '/UR_example_' .. (epoch or 0) .. '.png', formatted)
	
	IDX = torch.randperm(ntrain)
	
	if opt.gpu then 
		torch.setdefaulttensortype('torch.CudaTensor')
	else
		torch.setdefaulttensortype('torch.FloatTensor')
	end
	
	URnet.train(trainData_LR,trainData_HR)
	
	sgdState_D.momentum = math.min(sgdState_D.momentum + 0.0008, 0.7)
    sgdState_D.learningRate = math.max(opt.learningRate*0.99^epoch, 0.000001)
	
	sgdState_G.momentum = math.min(sgdState_G.momentum + 0.0008, 0.7)
	sgdState_G.learningRate = math.max(opt.learningRate*0.99^epoch, 0.000001)
	
	opt.lambda = math.max(opt.lambda*0.99, 0.005)   -- or 0.995
end
