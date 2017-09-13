require 'nn'
require 'cunn'
require 'cudnn'
require 'image'
require 'xlua'
local optnet = require 'optnet'

local tools = require 'tools/tools'
local cluster = require 'tools/clustering'

-- Load cmdline options
local opts = (require 'test_opts').parse(arg)

-- load dataset
local dataset_it, _ = require('datasets/dataset')(opts, opts.mode)
local dataset_size = dataset_it:execSingle('size')

-- load model
local model = torch.load(opts.model)
model:evaluate()
model:cuda()
-- optimize model for inference
optnet.optimizeMemory(model, torch.CudaTensor(1, 3, opts.size / 2, opts.size), {inplace = true, mode = 'inference', removeGradParams = true})

-- iterate through dataset
local t = 1
for item in dataset_it() do
    -- load image + gt labels
    local im = item.image:cuda() -- 1 x 3 x h x w

    -- forward through model
    local outp = model:forward(im)

    -- Extract different outputs from network
    local out_segm = outp[1]:float() -- 1 x 20 x h x w
    local out_instances = outp[2]:float() -- 1 x 8 x h x w
    local out_depth = outp[3]:float() -- 1 x 1 x h x w

    -- Segm: calculate labels
    local _, labels_segm = torch.max(out_segm, 2)
    labels_segm = labels_segm:byte()

    -- Depth: set ignore values to zero for better visualization
    if (opts.mode ~= 'test') then
        out_depth[gt_labels_depth:eq(0)] = 0
    end

    -- Cluster instances
    local labels_inst = cluster.cluster(out_instances, labels_segm:eq(15), 1.5)

    if (opts.display) then
        -- display
        win1 = image.display({image = im, win = win1, min=0, max=1})
        win2 = image.display({image = torch.add(im:float(), tools.to_color(labels_segm, 21)), win = win2})
        win3 = image.display({image = out_depth, win = win3})
        win4 = image.display({image = torch.add(im:float(), 0.5*tools.to_color(labels_inst, 256)), win = win4, min=0, max=1})

        print('Enter to continue ...')
        io.read()
    end

    -- opts.save images
    if (opts.save) then
        local name = item.name[1]

        labels_segm = image.scale(labels_segm:squeeze(), opts.original_size, opts.original_size / 2, 'simple')
        out_depth = image.scale(out_depth:squeeze(), opts.original_size, opts.original_size / 2, 'simple')
        labels_inst = image.scale(labels_inst:squeeze(), opts.original_size, opts.original_size / 2, 'simple')

        image.save(paths.concat(opts.save_dir, string.format('%s_segm.png', name)), labels_segm)
        image.save(paths.concat(opts.save_dir, string.format('%s_disp.png', name)), out_depth)
        image.save(paths.concat(opts.save_dir, string.format('%s_inst.png', name)), labels_inst)

        xlua.progress(t, dataset_size)
        t = t + 1
    end
end
