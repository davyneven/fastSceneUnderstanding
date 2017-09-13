require 'image'

local function loss(input, label)
    local mask = label:gt(0)
    local d = input[mask] - label[mask]
    local ds = d:size(1)

    local da = torch.abs(d)
    local d2 = torch.pow(d, 2)

    local th = 1 / 5 * torch.max(da)
    local mask2 = torch.gt(da, th)
    da[mask2] = (d2[mask2] + (th * th)) / (2 * th)

    return 1 / ds * torch.sum(da)
end

return loss
