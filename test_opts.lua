local M = {}

function M.parse(arg)
    local cmd = torch.CmdLine()
    cmd:text()
    cmd:text('Torch-7 Fast Scene Understanding Test script.')
    cmd:text('Copyright (c) 2017, Neven and De Brabandere')
    cmd:text()
    cmd:text('Options:')

    ------------ Data options --------------------

    cmd:option('-dataset', 'cityscapes', 'Options: cityscapes')
    cmd:option('-data_root', '/path/to/cityscapes/')
    cmd:option('-mode', 'test', 'mode: train, trainval, val, test')

    ------------- Data transformation -------------

    cmd:option('-size', 1024, 'rescale longer side to size')
    cmd:option('-original_size', 2048, 'original size to rescale to when saving')

    ------------- Model -----------------------

    cmd:option('-model', '/path/to/model/model.t7', 'path to model')

    ------------- SAVE AND DISPLAY ------------

    cmd:option('-save', 'false', 'save models')
    cmd:option('-save_dir', '/save/dir/', 'save directory')

    cmd:option('-display', 'true', 'display')

    cmd:text()

    local opts = cmd:parse(arg or {})
    opts.save = opts.save ~= 'false'
    opts.display = opts.display ~= 'false'

    if (opts.save) then
        if not paths.dirp(opts.save_dir) and not paths.mkdir(opts.save_dir) then
            cmd:error('error: unable to create save directory: ' .. opts.save_dir .. '\n')
        end
    end

    return opts
end

return M
