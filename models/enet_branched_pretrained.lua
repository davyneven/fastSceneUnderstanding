require 'nn'
require 'cunn'
require 'cudnn'

local function bottleneck(input, output, upsample, reverse_module)
    local internal = output / 4
    local input_stride = upsample and 2 or 1

    local module = nn.Sequential()
    local sum = nn.ConcatTable()
    local main = nn.Sequential()
    local other = nn.Sequential()
    sum:add(main):add(other)

    main:add(cudnn.SpatialConvolution(input, internal, 1, 1, 1, 1, 0, 0):noBias())
    main:add(nn.SpatialBatchNormalization(internal, 1e-3))
    main:add(cudnn.ReLU(true))
    if not upsample then
        main:add(cudnn.SpatialConvolution(internal, internal, 3, 3, 1, 1, 1, 1))
    else
        main:add(nn.SpatialFullConvolution(internal, internal, 3, 3, 2, 2, 1, 1, 1, 1))
    end
    main:add(nn.SpatialBatchNormalization(internal, 1e-3))
    main:add(cudnn.ReLU(true))
    main:add(cudnn.SpatialConvolution(internal, output, 1, 1, 1, 1, 0, 0):noBias())
    main:add(nn.SpatialBatchNormalization(output, 1e-3))

    other:add(nn.Identity())
    if input ~= output or upsample then
        other:add(cudnn.SpatialConvolution(input, output, 1, 1, 1, 1, 0, 0):noBias())
        other:add(nn.SpatialBatchNormalization(output, 1e-3))
        if upsample and reverse_module then
            other:add(nn.SpatialMaxUnpooling(reverse_module))
        end
    end

    if upsample and not reverse_module then
        main:remove(#main.modules) -- remove BN
        return main
    end
    return module:add(sum):add(nn.CAddTable()):add(cudnn.ReLU(true))
end

local function createModel(nClasses)
    local pretrained_model = torch.load('models/cityscapesSegmentation.t7')
    local model = nn.Sequential()

    -- add shared encoder part
    for i = 1, 18 do
        local module = pretrained_model:get(i):clone()
        model:add(module)
    end

    -- create branches and add non-shared encoder part
    local split = nn.ConcatTable()
    local branches = {}
    for j = 1, #nClasses do
        local branch = nn.Sequential()
        table.insert(branches, branch)
        for i = 19, 26 do
            local module = pretrained_model:get(i):clone()
            branch:add(module)
        end
        split:add(branch)
    end

    model:add(split)

    -- find pooling modules
    local pooling_modules = {}
    model:apply(
        function(module)
            if torch.typename(module):match('nn.SpatialMaxPooling') then
                table.insert(pooling_modules, module)
            end
        end
    )
    assert(#pooling_modules == 3, 'There should be 3 pooling modules')

    -- add decoder part
    for i = 1, 3 do
        local branch = branches[i]
        branch:add(pretrained_model:get(27):clone())
        branch:add(pretrained_model:get(28):clone())
        branch:add(pretrained_model:get(29):clone())
        branch:add(pretrained_model:get(30):clone())
        branch:add(pretrained_model:get(31):clone())
        if (nClasses[i] == 20) then
            branch:add(pretrained_model:get(32):clone())
        else
            branch:add(nn.SpatialFullConvolution(16, nClasses[i], 2, 2, 2, 2))
        end

        -- relink maxunpooling to correct pooling layer
        local counter = 3
        branch:apply(
            function(module)
                if torch.typename(module):match('nn.SpatialMaxUnpooling') then
                    module.pooling = pooling_modules[counter]
                    counter = counter - 1
                end
            end
        )
    end

    pretrained_model = nil
    return model
end

return createModel
