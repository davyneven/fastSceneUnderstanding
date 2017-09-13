local getModel = function(opts)
    local model

    if (opts.model == 'enetBranchedPretrained') then
        model = require('models/enet_branched_pretrained')(opts.nOutputs)
    else
        assert(false, 'unknown model: ' .. opts.model)
    end

    return model
end

return getModel
