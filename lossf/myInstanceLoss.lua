local M = {}
local tools = require 'tools/tools'

local in_margin = 0.5
local out_margin = 1.5
local Lnorm = 2

local function norm(inp, L)
    local n
    if (L == 1) then
        n = torch.sum(torch.abs(inp), 1)
    else
        n = torch.sqrt(torch.sum(torch.pow(inp, 2), 1) + 1e-8)
    end
    return n
end

-- prediction: batchsize x nDim x h x w
-- labels: batchsize x classes x h x w

local lossf =
    function(prediction, labels)
    local batchsize = prediction:size(1)
    local c = prediction:size(2)
    local height = prediction:size(3)
    local width = prediction:size(4)
    local nInstanceMaps = labels:size(2)
    local loss = 0

    M.loss_dist = 0
    M.loss_var = 0

    for b = 1, batchsize do
        local pred = prediction[b] -- c x h x w
        local loss_var = 0
        local loss_dist = 0

        for h = 1, nInstanceMaps do
            local label = labels[b][h]:view(1, height, width) -- 1 x h x w
            local means = {}
            local loss_v = 0
            local loss_d = 0

            -- center pull force
            for j = 1, label:max() do
                local mask = label:eq(j)
                local mask_sum = mask:sum()
                if (mask_sum > 1) then
                    local inst = pred[mask:expandAs(pred)]:view(c, -1, 1) -- c x -1 x 1

                    -- Calculate mean of instance
                    local mean = torch.mean(inst, 2) -- c x 1 x 1
                    table.insert(means, mean)

                    -- Calculate variance of instance
                    local var = norm((inst - mean:expandAs(inst)), 2) -- 1 x -1 x 1
                    var = torch.cmax(var - (in_margin), 0)
                    local not_hinged = torch.sum(torch.gt(var, 0))

                    var = torch.pow(var, 2)
                    var = var:view(-1)

                    var = torch.mean(var)
                    loss_v = loss_v + var
                end
            end

            loss_var = loss_var + loss_v

            -- center push force
            if (#means > 1) then
                for j = 1, #means do
                    local mean_A = means[j] -- c x 1 x 1
                    for k = j + 1, #means do
                        local mean_B = means[k] -- c x 1 x 1
                        local d = norm(mean_A - mean_B, Lnorm) -- 1 x 1 x 1
                        d = torch.pow(torch.cmax(-(d - 2 * out_margin), 0), 2)
                        loss_d = loss_d + d[1][1][1]
                    end
                end

                loss_dist = loss_dist + loss_d / ((#means - 1) + 1e-8)
            end
        end

        loss = loss + (loss_dist + loss_var)
    end

    loss = loss / batchsize + torch.sum(prediction) * 0

    return loss
end

return lossf
