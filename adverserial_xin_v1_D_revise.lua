require 'torch'
require 'nn'
require 'cunn'
require 'optim'
require 'pl'

local adversarial = {}
local input_scale = 16
function rmsprop(opfunc, x, config, state)
	
    -- (0) get/update state
    local config = config or {}
    local state = state or config
    local lr = config.learningRate or 1e-2
    local alpha = config.alpha or 0.9
    local epsilon = config.epsilon or 1e-8

    -- (1) evaluate f(x) and df/dx
    local fx, dfdx = opfunc(x)
    if config.optimize == true then
        -- (2) initialize mean square values and square gradient storage
        if not state.m then
          state.m = torch.Tensor():typeAs(x):resizeAs(dfdx):zero()
          state.tmp = torch.Tensor():typeAs(x):resizeAs(dfdx)
        end

        -- (3) calculate new (leaky) mean squared values
        state.m:mul(alpha)
        state.m:addcmul(1.0-alpha, dfdx, dfdx)

        -- (4) perform update
        state.tmp:sqrt(state.m):add(epsilon)
        -- only opdate when optimize is true
        
        
	if config.numUpdates < 50 then
	      io.write(" ", lr/50.0, " ")
	      x:addcdiv(-lr/50.0, dfdx, state.tmp)
	elseif config.numUpdates < 100 then
	    io.write(" ", lr/5.0, " ")
	    x:addcdiv(-lr /5.0, dfdx, state.tmp)
	else 
	  io.write(" ", lr, " ")
	  x:addcdiv(-lr, dfdx, state.tmp)
	end
    end
    config.numUpdates = config.numUpdates +1
  

    -- return x*, f(x) before optimization
    return x, {fx}
end


function adam(opfunc, x, config, state)
    --print('ADAM')
    -- (0) get/update state
    local config = config or {}
    local state = state or config
    local lr = config.learningRate or 0.001

    local beta1 = config.beta1 or 0.9
    local beta2 = config.beta2 or 0.999
    local epsilon = config.epsilon or 1e-8

    -- (1) evaluate f(x) and df/dx
    local fx, dfdx = opfunc(x)
    if config.optimize == true then
	    -- Initialization
	    state.t = state.t or 0
	    -- Exponential moving average of gradient values
	    state.m = state.m or x.new(dfdx:size()):zero()
	    -- Exponential moving average of squared gradient values
	    state.v = state.v or x.new(dfdx:size()):zero()
	    -- A tmp tensor to hold the sqrt(v) + epsilon
	    state.denom = state.denom or x.new(dfdx:size()):zero()

	    state.t = state.t + 1
	    
	    -- Decay the first and second moment running average coefficient
	    state.m:mul(beta1):add(1-beta1, dfdx)
	    state.v:mul(beta2):addcmul(1-beta2, dfdx, dfdx)

	    state.denom:copy(state.v):sqrt():add(epsilon)

	    local biasCorrection1 = 1 - beta1^state.t
	    local biasCorrection2 = 1 - beta2^state.t
	    
		local fac = 1
		if config.numUpdates < 10 then
		    fac = 50.0
		elseif config.numUpdates < 30 then
		    fac = 5.0
		else 
		    fac = 1.0
		end
		io.write(" ", lr/fac, " ")
        local stepSize = (lr/fac) * math.sqrt(biasCorrection2)/biasCorrection1
	    -- (2) update x
	    x:addcdiv(-stepSize, state.m, state.denom)
    end
    config.numUpdates = config.numUpdates +1
    -- return x*, f(x) before optimization
    return x, {fx}
end


-- training function

function adversarial.train(dataset_LR,dataset_HR, N)

  model_G:training()
  model_D:training()
  epoch = epoch or 0
  local N = N or dataset_HR:size(1)
  local dataBatchSize = opt.batchSize / 2
  local time = sys.clock()
  local err_gen = 0
    
  -- do one epoch
  print('\n<trainer> on training set:')
  print("<trainer> online epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. ' lr = ' .. sgdState_D.learningRate .. ', momentum = ' .. sgdState_D.momentum .. ']')
  for t = 1,N,opt.batchSize do --dataBatchSize do

    local inputs = torch.Tensor(opt.batchSize, opt.geometry[1], opt.geometry[2], opt.geometry[3])
    local targets = torch.Tensor(opt.batchSize)
    -- local LR_inputs = torch.Tensor(opt.batchSize, 3, 16, 16) 
	
--	local HR_inputs = dataset_HR[{{t,math.min(t+dataBatchSize,dataset_HR:size()[1])-1}}]
--	local LR_inputs = dataset_LR[{{t,math.min(t+dataBatchSize,dataset_LR:size()[1])-1}}]
	local HR_inputs = torch.Tensor(opt.batchSize, opt.geometry[1], opt.geometry[2], opt.geometry[3])
	local LR_inputs = torch.Tensor(opt.batchSize, opt.geometry[1], input_scale, input_scale)

    ----------------------------------------------------------------------
    -- create closure to evaluate f(X) and df/dX of discriminator
    local fevalD = function(x)
      collectgarbage()
      if x ~= parameters_D then -- get new parameters
        parameters_D:copy(x)
      end

      gradParameters_D:zero() -- reset gradients

      --  forward pass
      local outputs = model_D:forward(inputs)
      -- err_F = criterion_D:forward(outputs:narrow(1, 1, opt.batchSize / 2), targets:narrow(1, 1, opt.batchSize / 2))
      -- err_R = criterion_D:forward(outputs:narrow(1, (opt.batchSize / 2) + 1, opt.batchSize / 2), targets:narrow(1, (opt.batchSize / 2) + 1, opt.batchSize / 2))
      err_R = criterion_D:forward(outputs:narrow(1, 1, opt.batchSize / 2), targets:narrow(1, 1, opt.batchSize / 2))
      err_F = criterion_D:forward(outputs:narrow(1, (opt.batchSize / 2) + 1, opt.batchSize / 2), targets:narrow(1, (opt.batchSize / 2) + 1, opt.batchSize / 2))
    
      local margin = opt.margin -- org = 0.3
      sgdState_D.optimize = true
      sgdState_G.optimize = true      
      if err_F < margin or err_R < margin then
         sgdState_D.optimize = false
      end
      if err_F > (1.0-margin) or err_R > (1.0-margin) then
         sgdState_G.optimize = false
      end
      if sgdState_G.optimize == false and sgdState_D.optimize == false then
         sgdState_G.optimize = true 
         sgdState_D.optimize = true
      end

  
      --print(monA:size(), tarA:size())
      --io.write("v1_ytc| R:", err_R,"  F:", err_F, "  ")
      local f = criterion_D:forward(outputs, targets)

      -- backward pass 
      local df_do = criterion_D:backward(outputs, targets)
      model_D:backward(inputs, df_do)

      -- penalties (L1 and L2):
      if opt.coefL1 ~= 0 or opt.coefL2 ~= 0 then
        local norm,sign= torch.norm,torch.sign
        -- Loss:
        f = f + opt.coefL1 * norm(parameters_D,1)
        f = f + opt.coefL2 * norm(parameters_D,2)^2/2
        -- Gradients:
        gradParameters_D:add( sign(parameters_D):mul(opt.coefL1) + parameters_D:clone():mul(opt.coefL2) )
      end

      --print('grad D', gradParameters_D:norm())
      return f,gradParameters_D
    end

    ----------------------------------------------------------------------
    -- create closure to evaluate f(X) and df/dX of generator 
    local fevalG = function(x)
      collectgarbage()
      if x ~= parameters_G then -- get new parameters
        parameters_G:copy(x)
      end
      
      gradParameters_G:zero() -- reset gradients

      -- forward pass
      local samples = model_G:forward(LR_inputs)
	  local g       = criterion_G:forward(samples, HR_inputs) 
	  err_gen       = err_gen + g  
      local outputs = model_D:forward(samples)
      local f       = criterion_D:forward(outputs, targets)

      --io.write("G:",f+g, " G:", tostring(sgdState_G.optimize)," D:",tostring(sgdState_D.optimize)," ", sgdState_G.numUpdates, " ", sgdState_D.numUpdates , "\n")
      --io.flush()

      --  backward pass
      local df_samples = criterion_D:backward(outputs, targets)
      model_D:backward(samples, df_samples) 
	  local df_G_samples = criterion_G:backward(samples, HR_inputs)   ---added by xin
      local df_do = model_D.modules[1].gradInput * opt.lambda + df_G_samples
      model_G:backward(LR_inputs, df_do)

--      print('gradParameters_G', gradParameters_G:norm())
      return f,gradParameters_G
    end

    ----------------------------------------------------------------------
    -- (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
    -- Get half a minibatch of real, half fake
    for k=1,opt.K do
      -- (1.1) Real data
	  --print(dataset_HR:size())  
		local k = 1

		for i = t,t+dataBatchSize-1 do
		  -- print(IDX[i])  
		  local sample    = dataset_HR[IDX[i]]
		  inputs[k] = sample:clone()
		  k = k + 1
		end  
		
		local sample = torch.Tensor(dataBatchSize, opt.geometry[1], input_scale, input_scale)
		local k = 1
		for i = t+dataBatchSize,t+opt.batchSize-1 do
		  local sample_LR = dataset_LR[IDX[i]] 
		  -- inputs[k] = image.scale(sample_LR, opt.geometry[2], opt.geometry[3])  -- another choice
		  sample[k] = sample_LR:clone()
		  k = k + 1
		end

--		local  a = model_G:forward(sample)
--	    print(a:size())
--		print(inputs:size())
		
		inputs[{{k,opt.batchSize}}] = torch.squeeze(model_G:forward(sample))
		
		targets[{{1,dataBatchSize}}]:fill(1)	
		targets[{{dataBatchSize+1, opt.batchSize}}]:fill(0)

		rmsprop(fevalD, parameters_D, sgdState_D)

    end -- end for K


    ----------------------------------------------------------------------
    -- (2) Update G network: maximize log(D(G(z)))
    -- noise_inputs:normal(0, 1)
    local k = 1
    for i = t, t+opt.batchSize-1 do
      local sample_HR = dataset_HR[IDX[i]]
      local sample_LR = dataset_LR[IDX[i]]  
      HR_inputs[k] = sample_HR:clone()
      LR_inputs[k] = sample_LR:clone()
      k = k+1  
    end
	targets:fill(1)
	rmsprop(fevalG, parameters_G, sgdState_G)

	-- display progress
	xlua.progress(t, dataset_HR:size()[1])
  end -- end for loop over dataset

    -- time taken
    time = sys.clock() - time
    time = time / dataset_HR:size()[1]
    print("<trainer> time to learn 1 sample = " .. (time*1000) .. 'ms')

    -- print confusion matrix
    --print(confusion)
    trainLogger:add{['MSE accuarcy1'] = err_gen/opt.batchSize}

    -- save/log current net
    if epoch % opt.saveFreq == 0 then
      local filename = paths.concat(opt.save, 'adversarial.net')
      os.execute('mkdir -p ' .. sys.dirname(filename))
      if paths.filep(filename) then
        os.execute('mv ' .. filename .. ' ' .. filename .. '.old')
      end
      print('<trainer> saving network to '..filename)
	  model_D:clearState()
	  model_G:clearState()  
      torch.save(filename, {D = model_D, G = model_G, opt = opt})
    end

    -- next epoch
    epoch = epoch + 1
end


-- test function
function adversarial.test(dataset_LR,dataset_HR, N)
  model_G:evaluate()
  model_D:evaluate()
  local time = sys.clock()
  local N = N or dataset_HR:size()[1]

  print('\n<trainer> on testing Set:')
  for t = 1,N,opt.batchSize do
    -- display progress
    xlua.progress(t, dataset:size()[1])

    ----------------------------------------------------------------------
    --(1) Real data
    local inputs = torch.Tensor(opt.batchSize,opt.geometry[1],opt.geometry[2], opt.geometry[3])
    local targets = torch.ones(opt.batchSize)
    local k = 1
    for i = t,t+opt.batchSize-1 do
      local idx = math.random(dataset_HR:size()[1])
      local sample = dataset_HR[idx]
      local input = sample:clone()
      inputs[k] = input
      k = k + 1
    end
    local preds = model_D:forward(inputs) -- get predictions from D
    -- add to confusion matrix
    for i = 1,opt.batchSize do
      local c
      if preds[i][1] > 0.5 then c = 2 else c = 1 end
      confusion:add(c, targets[i] + 1)
    end

    ----------------------------------------------------------------------
    -- (2) Generated data (don't need this really, since no 'validation' generations)
    ---------------------local noise_inputs = torch.Tensor(opt.batchSize, opt.noiseDim):normal(0, 1)
    local inputs_lr = torch.Tensor(opt.batchSize, 3, 16, 16)
    local k = 1
    for i = t,t+opt.batchSize-1 do
      local idx = math.random(dataset_LR:size()[1])
      local sample = dataset_LR[idx]
      local input = sample:clone()
      inputs_lr[k] = input
      k = k + 1
    end
    local inputs = model_G:forward(inputs_lr)
    local targets = torch.zeros(opt.batchSize)
    local preds = model_D:forward(inputs) -- get predictions from D
    -- add to confusion matrix
    for i = 1,opt.batchSize do
      local c
      if preds[i][1] > 0.5 then c = 2 else c = 1 end
      confusion:add(c, targets[i] + 1)
    end
  end -- end loop over dataset

  -- timing
  time = sys.clock() - time
  time = time / dataset:size()[1]
  print("<trainer> time to test 1 sample = " .. (time*1000) .. 'ms')

  -- print confusion matrix
  print(confusion)
  testLogger:add{['% mean class accuracy (test set)'] = confusion.totalValid * 100}
  confusion:zero()
end

return adversarial
