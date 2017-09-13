local tools = require 'tools/tools'

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

local clusterWithLabels =
    function(out, targets, bandwidth)
    if (out:dim() == 4) then
        out = out:squeeze(1)
    end
    if (targets:dim() == 4) then
        targets = targets:squeeze(1)
    end
    local c = out:size(1)
    local height = out:size(2)
    local width = out:size(3)
    local maps = torch.ByteTensor(targets:size(1), height, width):zero()
    for h = 1, targets:size(1) do
        local target = targets[h]:view(1, height, width)

        local segmented = torch.ByteTensor():resizeAs(target):zero()

        for i = 1, target:max() do
            local target_mask = target:eq(i):expandAs(out)
            if (target_mask:sum() > 0) then
                local inst_pixels = out[target_mask]:view(c, -1, 1) -- c x -1 x 1
                local th_value = torch.mean(inst_pixels, 2)

                local d_map = norm(out - th_value:expandAs(out), Lnorm):squeeze() -- 1 x h x w
                local threshold_mask = torch.lt(d_map, bandwidth)

                segmented[threshold_mask] = i
            end
        end
        segmented[target:eq(0)] = 0
        -- remap labels
        local l = 1
        for j = 1, torch.max(segmented) do
            if (segmented:eq(j):sum() > 1) then
                segmented[segmented:eq(j)] = l
                l = l + 1
            else
                segmented[segmented:eq(j)] = 0
            end
        end
        maps[h] = segmented
    end
    return maps
end

local meanshift = function(samples, mean, bandwidth)
    local ndim = samples:size(1)
    -- calculate norm map
    local norm_map = samples - mean:expandAs(samples)
    norm_map = norm(norm_map, 2)
    -- threshold
    local mask = torch.lt(norm_map, bandwidth):expandAs(samples)
    -- calculate new mean
    local new_mean
    if (mask:sum() > 0) then
        new_mean = torch.mean(samples[mask]:view(ndim, -1), 2)
    else
        new_mean = mean
    end
    return new_mean
end

-- unoptimized clustering
local cluster =
    function(out, label, bandwidth)
    if (out:dim() == 4) then
        out = out:squeeze(1)
    end
    if (label:dim() == 4) then
        label = label:squeeze(1)
    end

    local ndim = out:size(1)
    local h, w = out:size(2), out:size(3)

    local mask = label:ne(0):view(-1) -- h*w
    if (mask:sum() == 0) then
        return torch.ByteTensor(1, h, w):zero()
    end

    out = out:view(ndim, -1) -- ndim x h*w

    local unclustered = torch.ones(h * w):byte():cmul(mask)
    local instance_map = torch.zeros(h * w):int()

    local l = 1
    local counter = 0
    while (unclustered:sum() > 100 and counter < 20) do
        -- Mask out
        local out_masked = out[unclustered:view(1, -1):expandAs(out)]:view(ndim, -1)

        -- Take random unclustered pixel
        local index = math.random(out_masked:size(2))
        local mean = out_masked[{{}, {index}}] -- ndim x 1

        -- Do meanshift until convergence
        local new_mean = meanshift(out_masked, mean, bandwidth)
        local it = 0
        while (torch.norm(mean - new_mean) > 0.0001 and it < 100) do
            mean = new_mean
            new_mean = meanshift(out_masked, mean, bandwidth)
            it = it + 1
        end

        -- Threshold around mean
        if (it < 100) then
            -- Mask out pixels
            local norm_map = norm(out - new_mean:expandAs(out), 2)

            -- threshold
            local th_mask = torch.lt(norm_map, bandwidth):view(-1)

            -- calculate intersection
            local inter = torch.cmul(instance_map:gt(0), th_mask):sum()
            local iop = inter / torch.sum(th_mask)

            if (iop < 0.5) then
                -- Don't overwrite previous found pixels
                th_mask = torch.cmul(th_mask, unclustered)
                -- Do erosion
                local th_mask_tmp = image.erode(th_mask:view(h, w))
                -- Do dilation
                local th_mask_tmp = image.dilate(th_mask_tmp)
                instance_map[th_mask_tmp:view(-1)] = l
                l = l + 1
            end

            -- Mask out clustered pixels
            unclustered[th_mask:view(-1)] = 0
            counter = 0
        else
            counter = counter + 1
        end
    end

    --relabel
    local tmp = torch.ByteTensor(h * w):zero()
    local l = 1
    for j = 1, torch.max(instance_map) do
        if (instance_map:eq(j):sum() > 10) then
            tmp[instance_map:eq(j)] = l
            l = l + 1
        end
    end

    instance_map = tmp:view(1, h, w)

    return instance_map
end

local M = {}
M.clusterWithLabels = clusterWithLabels
M.cluster = cluster

return M
