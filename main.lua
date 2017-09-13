require 'nn'
require 'cunn'
require 'cudnn'
require 'image'
local tnt = require('torchnet')
require 'engines/myEngine'
require 'xlua'
local tools = require 'tools/tools'
local cluster = require 'tools/clustering'
require 'optim'

-- Load cmdline options
local opts = (require 'opts').parse(arg)

-- Load model
local model
local optimState
if (opts.resume == true) then
    model = torch.load(paths.concat(opts.directory, 'model.t7'))
    optimState = torch.load(paths.concat(opts.directory .. 'optim.t7'))
else
    model = require('models/model')(opts)
end

print('Model definition:')
print(model)

-- Multi GPU support
local gpu_list = {}
if opts.nGPU == 1 then
    gpu_list[1] = opts.devid
else
    for i = 1, opts.nGPU do
        gpu_list[i] = i
    end
end
model = nn.DataParallelTable(1, true, false):add(model:cuda(), gpu_list)
print(opts.nGPU .. ' GPUs being used')
model:cuda()

-- Load train/val dataset
local train_dataset_it, class_weights = require('datasets/dataset')(opts, opts.train_mode)
local train_size = train_dataset_it:execSingle('size')

local val_dataset_it, val_size
if (opts.val) then
    val_dataset_it = require('datasets/dataset')(opts, 'val')
    val_size = val_dataset_it:execSingle('size')
end

-- Load loss functions
if (not opts.classWeighting) then
    class_weights[1] = 0
    for i = 2, class_weights:numel() do
        class_weights[i] = 1
    end
end

local segmentation_criterion = require('lossf/loss')('softmaxLoss', class_weights)
local instance_criterion = require('lossf/loss')('instanceLoss')
local depth_criterion = require('lossf/loss')('huberLoss')

local criterion = nn.ParallelCriterion()
criterion:add(segmentation_criterion)
criterion:add(instance_criterion)
criterion:add(depth_criterion)

criterion:cuda()

print('Train dataset size: ' .. train_size)
print('Val dataset size: ' .. val_size)

-- Meters + loggers
local lossMeter = tnt.AverageValueMeter()
local segmMeter = tnt.AverageValueMeter()
local instMeter = tnt.AverageValueMeter()
local depthMeter = tnt.AverageValueMeter()

local logger
local logger_seperate
if (opts.save == true) then
    logger = optim.Logger(opts.directory .. 'loss.log')
    logger_seperate = optim.Logger(opts.directory .. 'loss_seperate.log')
else
    logger = optim.Logger()
    logger_seperate = optim.Logger()
end

if (opts.val == true) then
    logger:setNames {'train loss', 'val loss'}
    logger:style {'-', '-'}

    logger_seperate:setNames {'segm(t)', 'inst(t)', 'depth(t)', 'segm(v)', 'inst(v)', 'depth(v)'}
    logger_seperate:style {'-', '-', '-', '-', '-', '-'}
else
    logger:setNames {'train loss'}
    logger:style {'-'}

    logger_seperate:setNames {'segm', 'inst', 'depth'}
    logger_seperate:style {'-', '-', '-'}
end

local best_train_loss = 1000
local best_val_loss = 1000

-- Start TNT engine

local engine = tnt.myEngine()

-- Preallocate tensors on GPU

local input = torch.CudaTensor()
local targetSegm = torch.CudaByteTensor()
local targetInst = torch.CudaByteTensor()
local targetDepth = torch.CudaTensor()

engine.hooks.onStart = function(state)
    lossMeter:reset()
    segmMeter:reset()
    instMeter:reset()
    depthMeter:reset()
end

engine.hooks.onStartEpoch =
    function(state)
    -- Shuffle dataset at start epoch
    state.iterator:exec('resample')

    -- Reset loss meter
    lossMeter:reset()
    segmMeter:reset()
    instMeter:reset()
    depthMeter:reset()

    if (opts.freezeBN) then
        model:apply(
            function(m)
                if torch.type(m):find('BatchNormalization') then
                    m:evaluate()
                end
            end
        )
        print('fixing batch norm ...')
    end

    print('Starting epoch: ' .. state.epoch)
end

engine.hooks.onSample = function(state)
    if (state.training) then
        xlua.progress(state.t, train_size)
    else
        xlua.progress(state.t, val_size)
    end

    -- copy data to containers on GPU
    input:resize(state.sample.image:size()):copy(state.sample.image)
    targetSegm:resize(state.sample.label:size()):copy(state.sample.label)
    targetDepth:resize(state.sample.depth:size()):copy(state.sample.depth)
    targetInst:resize(state.sample.instances:size()):copy(state.sample.instances)

    state.sample.input = input
    state.sample.target = {targetSegm, targetInst, targetDepth}

    collectgarbage()
end

local win1, win2, win3, win4, win5, win6, win7

engine.hooks.onForwardCriterion = function(state)
    -- accumulate loss in meter
    lossMeter:add(criterion.output)
    segmMeter:add(segmentation_criterion.output)
    instMeter:add(instance_criterion.output)
    depthMeter:add(depth_criterion.output)

    -- display results
    if (opts.display == true) then
        if ((state.t + 1) % 2 == 0) then
            win1 = image.display({image = state.sample.input[1], win = win1, legend = 'input image', zoom = 0.5})
            win2 = image.display({image = tools.to_color(state.sample.target[1][1], 21), win = win2, legend = 'labels gt', zoom = 0.5})
            win3 = image.display({image = tools.to_color(state.sample.target[2][1], 256), win = win3, legend = 'instances gt', zoom = 0.5})
            win4 = image.display({image = state.sample.target[3][1], win = win4, legend = 'depth gt', zoom = 0.5})

            local out = state.network.output[1][1]:float()
            local _, classes = torch.max(out, 1)
            win5 = image.display({image = tools.to_color(classes, 21), win = win5, legend = 'labels', zoom = 0.5})

            local out_inst = state.network.output[2][1]:float()
            local inst_clustered = cluster.clusterWithLabels(out_inst, state.sample.target[2][1]:byte(), 1.5)
            win6 = image.display({image = tools.to_color(inst_clustered, 256), win = win6, legend = 'instances', zoom = 0.5})

            local out_depth = state.network.output[3][1]:float()
            win7 = image.display({image = out_depth, win = win7, legend = 'depth', zoom = 0.5})
        end
    end
end

engine.hooks.onEndEpoch =
    function(state)
    local train_loss = lossMeter:value()
    local train_segm_loss = segmMeter:value()
    local train_inst_loss = instMeter:value()
    local train_depth_loss = depthMeter:value()
    print('average train loss: ' .. train_loss)

    if (opts.save == true) then
        print('saving model')
        torch.save(paths.concat(opts.directory, 'model.t7'), model:clearState():get(1))
        torch.save(paths.concat(opts.directory, 'optim.t7'), state.optim)

        if (train_loss < best_train_loss) then
            print('save best train model')
            best_train_loss = train_loss
            torch.save(paths.concat(opts.directory, 'best_train_model.t7'), model:clearState():get(1))
        end
    end

    if (opts.val == true) then
        state.t = 0
        engine:test {
            network = model,
            iterator = val_dataset_it,
            criterion = criterion
        }

        local val_loss = lossMeter:value()
        local val_segm_loss = segmMeter:value()
        local val_inst_loss = instMeter:value()
        local val_depth_loss = depthMeter:value()
        print('average val loss: ' .. val_loss)

        if (opts.save == true) then
            if (val_loss < best_val_loss) then
                print('save best val model')
                best_val_loss = val_loss
                torch.save(paths.concat(opts.directory, 'best_val_model.t7'), model:clearState():get(1))
            end
        end

        logger:add {train_loss, val_loss}
        logger:plot()

        logger_seperate:add {train_segm_loss, train_inst_loss, train_depth_loss, val_segm_loss, val_inst_loss, val_depth_loss}
        logger_seperate:plot()
    else
        logger:add {train_loss}
        logger:plot()

        logger_seperate:add {train_segm_loss, train_inst_loss, train_depth_loss}
        logger_seperate:plot()
    end

    state.t = 0

    -- Do lr decay
    if (opts.LRdecay > 0) then
        state.config.learningRate = opts.LR / (1 + state.epoch * opts.LRdecay)
    end
end

engine:train {
    network = model,
    iterator = train_dataset_it,
    criterion = criterion,
    optimMethod = opts.useAdam and optim.adam or optim.sgd,
    optimState = optimState,
    config = {
        learningRate = opts.LR,
        weightDecay = 2e-4
    },
    maxepoch = opts.nEpochs,
    iterSize = opts.iterSize
}
