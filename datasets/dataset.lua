local tnt = require('torchnet')
require('datasets/Cityscapes')
require 'paths'

local function getTransformations(opts, mode)
    local transforms = require 'tools/transforms'
    local transf = {}

    -- rescale image
    if (mode == 'test') then
        table.insert(transf, transforms.Scale(opts.size, {'image'}, {'bicubic'}, false))
    else
        table.insert(transf, transforms.Scale(opts.size, {'image', 'label', 'depth', 'instances'}, {'bicubic', 'simple', 'simple', 'simple'}, false))
    end

    -- add other transformations, as random crop/scale/rotation ...

    return transforms.Compose(transf)
end

local function getDatasetIterator(opts, mode)
    local it, weights
    local batchSize = mode == 'train' and opts.train_bs or opts.valtest_bs
    local transf = getTransformations(opts, mode)

    if (opts.dataset == 'cityscapes') then
        print('creating cityscapes dataset')
        local parenth_path = paths.concat(opts.data_root, 'cityscapes')

        if (mode == 'test') then
            it = tnt.Cityscapes(parenth_path, mode):transform(transf):batch(batchSize, 'skip-last'):iterator()
        elseif (mode == 'trainval') then
            local train_d = tnt.Cityscapes(parenth_path, 'train')
            local val_d = tnt.Cityscapes(parenth_path, 'val')
            local train_val = tnt.ConcatDataset({datasets = {train_d, val_d}})

            it =
                train_val:transform(transf):shuffle():batch(batchSize, 'skip-last'):parallel(
                {
                    nthread = 4,
                    init = function()
                        require 'torchnet'
                        require 'datasets/Cityscapes'
                    end
                }
            )
        else
            it =
                tnt.Cityscapes(parenth_path, mode):transform(transf):shuffle():batch(batchSize, 'skip-last'):parallel(
                {
                    nthread = 4,
                    init = function()
                        require 'torchnet'
                        require 'datasets/Cityscapes'
                    end
                }
            )
        end
    else
        assert(false, 'Cannot load dataset ' .. opts.dataset)
    end

    weights = tnt.Cityscapes.getClassWeights()

    return it, weights
end

local function unitTest()
    local tools = require('tools/tools')
    local it,
        weights =
        getDatasetIterator(
        {
            dataset = 'cityscapes',
            data_root = '/esat/toyota/datasets',
            train_bs = 1,
            valtest_bs = 1,
            size = 768
        },
        'train'
    )

    print('weights: ', weights)

    local win1, win2, win3, win4
    for item in it() do
        print('image type: ' .. item.image:type())
        print('image size: ', item.image:size())

        print('label type: ' .. item.label:type())
        print('label size: ', item.label:size())
        tools.unique(item.label)

        print('depth type: ' .. item.depth:type())
        print('depth size: ', item.depth:size())

        print('instance type: ' .. item.instances:type())
        print('instance size: ', item.instances:size())
        print(tools.unique(item.instances))

        win1 = image.display({image = item.image, win = win1, zoom = 0.25})
        win2 = image.display({image = tools.to_color(item.label, 256), win = win2, zoom = 0.25})
        win3 = image.display({image = item.depth, win = win3, zoom = 0.25})
        win4 = image.display({image = tools.to_color(item.instances, 256), win = win4, zoom = 0.25})

        io.read()
    end
end

--unitTest()

return getDatasetIterator
