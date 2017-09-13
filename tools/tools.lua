local function unique(input)
    input = input:view(-1)
    local b = {}
    for i = 1, input:numel() do
        b[input[i]] = true
    end
    local out = {}
    for i in pairs(b) do
        table.insert(out, i)
    end
    return out
end

local function to_color(input, dim, map)
    require 'imgraph'
    local ncolors = dim or input:max() + 1
    local colormap = map or image.colormap(ncolors)
    input = imgraph.colorize(input:squeeze():float(), colormap:float())
    return input
end

local M = {}

M.unique = unique
M.to_color = to_color

return M
