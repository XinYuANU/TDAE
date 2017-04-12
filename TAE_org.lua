require 'torch'
require 'nn'
require 'cunn'
require 'optim'
require 'pl'

local tae = {}
local input_scale = 128
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

-- training process
function tae.train(train_Res,train_HR,train_LR)
	
	model_EN:training()
	epoch = epoch or 1
	local N = N or train_HR:size(1)
	local err = 0
	print('<trainer> on training set: ')
	print("<trainer> online epoch # " ..epoch.. ' [batchsize = ' ..opt.batchSize.. ']')
	for t = 1, N, opt.batchSize do
		local inputs_res = torch.Tensor(opt.batchSize, 3, 128,128)
		local inputs_HR  = torch.Tensor(opt.batchSize, 3, 128,128)
		local inputs_LR  = torch.Tensor(opt.batchSize, 3, 16,16)
		local k = 1
		for i = t,t+opt.batchSize-1 do
			inputs_res[k] = train_Res[IDX[i]]:clone()
			inputs_HR[k]  = train_HR[IDX[i]]:clone()
			inputs_LR[k]  = train_LR[IDX[i]]:clone()
			k = k+1
		end
		
		local feval_EN = function(x)
			collectgarbage()
			if x~= parameters_EN then
				parameters_EN:copy(x)
			end
			
			gradParameters_EN:zero()
			
			-- without model_G
			---[[
			local outputs = model_EN:forward(inputs_res)
			local f = criterion_EN:forward(outputs, inputs_LR)
			err     = err+f
			local df_do = criterion_EN:backward(outputs,inputs_LR)
			model_EN:backward(inputs_res,df_do)
			--]]
			
			-- with model_G
			--[[
			local outputs = model_EN:forward(inputs_res)
			local outputs_UR = model_G:forward(outputs)
			local f = criterion_EN:forward(outputs, inputs_LR)
			local g = criterion_G:forward(outputs_UR, inputs_HR)
			err     = err+f+g
			
			local df_g  = criterion_G:backward(outputs_UR,inputs_HR)
			model_G:backward(outputs,df_g)
			local df_en = criterion_EN:backward(outputs,inputs_LR)
			local df_do = df_en + model_G.modules[1].gradInput
			model_EN:backward(inputs_res,df_do)
			--]]
			
			return f, gradParameters_EN
		end
		
		rmsprop(feval_EN,parameters_EN,sgdState_EN)
		xlua.progress(t, train_Res:size()[1])
	end
	
	trainLogger:add{['MSE accuarcy TAE'] = err/opt.batchSize}
	
    if epoch % opt.saveFreq == 0 then
      local filename = paths.concat(opt.save, 'TAE.net')
      os.execute('mkdir -p ' .. sys.dirname(filename))
      if paths.filep(filename) then
        os.execute('mv ' .. filename .. ' ' .. filename .. '.old')
      end
      print('<trainer> saving network to '..filename)
	  model_EN:clearState()   
      torch.save(filename, {EN = model_EN, opt = opt})
    end

    -- next epoch
    epoch = epoch + 1
end
return tae