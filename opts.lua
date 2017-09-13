local M = {}

function M.parse(arg)
    local cmd = torch.CmdLine()
    cmd:text()
    cmd:text('Torch-7 Fast Scene Understanding Training script.')
    cmd:text('Copyright (c) 2017, Neven and De Brabandere')
    cmd:text()
    cmd:text('Options:')

    ------------ Data options --------------------

    cmd:option('-dataset', 'cityscapes', 'Options: cityscapes')
    cmd:option('-data_root', '/esat/toyota/datasets')
    cmd:option('-train_mode', 'train', 'mode: train, trainval')
    cmd:option('-val', 'true', 'Do validation during training')

    ------------- Data transformation -------------

    cmd:option('-size', 1024, 'rescale longer side to size')

    ------------- Model options -----------------------

    cmd:option('-model', 'enetBranchedPretrained', 'model: enetBranchedPretrained')
    cmd:option('-nOutputs', {20, 8, 1}, 'number of output features of branches')
    cmd:option('-freezeBN', 'true')

    ------------- Loss options ------------------------

    cmd:option('-classWeighting', 'true', 'Do class weighting for softmaxloss')

    ------------- GPU opts ---------------------------

    cmd:option('-nGPU', 1, 'number of gpus')
    cmd:option('-devid', 1, 'device id')

    ------------- Training options --------------------

    cmd:option('-nEpochs', 100, 'Number of total epochs to run')
    cmd:option('-train_bs', 2, 'train batchsize')
    cmd:option('-valtest_bs', 2, 'val/test batchsize')
    cmd:option('-iterSize', 5, '#iterations before doing param update, so virtual bs = bs * itersize')
    cmd:option('-resume', 'false', 'Resume from the latest checkpoint')

    ------------- Learning options --------------------

    cmd:option('-useAdam', 'true', 'use adam or sgd')
    cmd:option('-LR', 5e-4, 'initial learning rate')
    cmd:option('-momentum', 0)
    cmd:option('-LRdecay', 0, 'do learning rate decay')

    ------------- General options ---------------------

    cmd:option('-save', 'false', 'save models')
    cmd:option('-directory', '/dir/to/save/', 'save directory')
    cmd:option('-name', 'branchedV1', 'name of folder')

    cmd:option('-display', 'true', 'display')

    cmd:text()

    local opts = cmd:parse(arg or {})

    opts.val = opts.val ~= 'false'
    opts.resume = opts.resume ~= 'false'
    opts.useAdam = opts.useAdam ~= 'false'
    opts.save = opts.save ~= 'false'
    opts.display = opts.display ~= 'false'
    opts.classWeighting = opts.classWeighting ~= 'false'
    opts.freezeBN = opts.freezeBN ~= 'false'

    opts.directory = opts.directory .. opts.name .. '/'

    if (opts.save) then
        if not paths.dirp(opts.directory) and not paths.mkdir(opts.directory) then
            cmd:error('error: unable to create save directory: ' .. opts.save .. '\n')
        end
        -- start logging
        cmd:log(opts.directory .. 'log.txt', opts)
    end

    return opts
end

return M
