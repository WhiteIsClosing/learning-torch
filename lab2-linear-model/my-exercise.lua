-- library used
require 'torch'
require 'nn'
require 'optim'

-- my training data
-- {corn, fertilizer, insecticide}
data = torch.Tensor{
   {40,  6,  4},
   {44, 10,  4},
   {46, 12,  5},
   {48, 14,  7},
   {52, 16,  9},
   {58, 18, 12},
   {60, 22, 14},
   {68, 24, 20},
   {74, 26, 21},
   {80, 32, 24}
}

-- define our model
model = nn.Sequential()
ninputs = 2; noutputs = 1;
model:add(nn.Linear(ninputs,noutputs))

-- define our loss function
criterion = nn.MSECriterion()
-- trainable parameters (x), and the gradients (dl_dx) of our loss function
x, dl_dx = model:getParameters()
print "-----------------"
print "this is x (param)"
print "-----------------"
print(x)
print "-----------------"
print "this is dl_dx (gradient)"
print "-----------------"
print(dl_dx)

-- eval
feval = function(x_new)

  -- print "-----------------"
  -- print "feval started ..."
  -- print "-----------------"

  -- get the param from the last iter
  if x ~= x_new then
     x:copy(x_new)
  end

  -- select a new training sample
  _nidx_ = (_nidx_ or 0) + 1
  -- if training idx went over limit, make it 1 again
  if _nidx_ > (#data)[1] then _nidx_ = 1 end

  -- select one sample
  local sample = data[_nidx_]
  local target = sample[{ {1} }]
  local inputs = sample[{ {2,3} }]

  -- reset gradients (gradients are always accumulated, to accomodate batch methods)
  dl_dx:zero()

  -- evaluate the loss function and its derivative wrt x, for that sample
  local loss_x = criterion:forward(model:forward(inputs), target)
  -- get us the new dl_dx
  model:backward(inputs, criterion:backward(model.output, target))

  -- return loss(x) and dloss/dx
  return loss_x, dl_dx
end

-- sgd param
sgd_params = {
   learningRate = 1e-3,
   learningRateDecay = 1e-4,
   weightDecay = 0,
   momentum = 0
}

-- training
for i = 1,1e4 do

  -- avg loss
  current_loss = 0

  for i = 1,(#data)[1] do
    -- optim contains several optimization algorithms.
    -- All of these algorithms assume the same parameters:
    --   + a closure that computes the loss, and its gradient wrt to x,
    --     given a point x
    --   + a point x
    --   + some parameters, which are algorithm-specific
    _,fs = optim.sgd(feval,x,sgd_params)
    -- "_" is the new x, the new theta
    -- fs is the loss of this step


    -- Functions in optim all return two things:
    --   + the new x, found by the optimization method (here SGD)
    --   + the value of the loss functions at all points that were used by
    --     the algorithm. SGD only estimates the function once, so
    --     that list just contains one value.
    current_loss = current_loss + fs[1]
  end

  -- report average error on epoch
  current_loss = current_loss / (#data)[1]
  print('current loss = ' .. current_loss)

end

print 'so ... at this point it should be trained!'

text = {40.32, 42.92, 45.33, 48.85, 52.37, 57, 61.82, 69.78, 72.19, 79.42}

print('id  approx   text')
for i=1,(#data)[1] do
  local myPrediction = model:forward(data[i][{{2,3}}])
  print(string.format("%2d  %6.2f %6.2f %6.2f", i, myPrediction[1], text[i], text[i]-myPrediction[1]))
end
