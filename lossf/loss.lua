local grad = require 'autograd'

local getLoss = function(lossf_name, weights)
    local lossfunction

    if (lossf_name == 'softmaxLoss') then
        lossfunction = cudnn.SpatialCrossEntropyCriterion(weights)
    elseif (lossf_name == 'huberLoss') then
        lossfunction = grad.nn.AutoCriterion('depthLoss_huber')(require 'lossf/myHuberLoss')
    elseif (lossf_name == 'instanceLoss') then
        lossfunction = grad.nn.AutoCriterion('instance_loss')(require 'lossf/myInstanceLoss')
    else
        assert(false, 'Cannot load lossfunction ' .. opts.lossf)
    end

    return lossfunction
end

return getLoss
