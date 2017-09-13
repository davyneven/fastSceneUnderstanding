--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--  Image transforms for data augmentation and input normalization
--
--  Adapted to only work on inputs specified by keys
--

require 'image'

local M = {}

function M.Compose(transforms)
    return function(input)
        for _, transform in ipairs(transforms) do
            input = transform(input)
        end
        return input
    end
end

-- Scales the smaller edge to size
function M.Scale(size, keys, interpolation, shorter_side)
    interpolation = interpolation or {}
    return function(input)
        if (keys == nil) then
            return input
        else
            assert(#keys == #interpolation, 'Need an equal number interpolations as keys!')
            local comp = shorter_side and math.min or math.max
            for i, k in pairs(keys) do
                local sample = input[k]
                local w, h = sample:dim() == 3 and sample:size(3) or sample:size(2), sample:dim() == 3 and sample:size(2) or sample:size(1)
                if (comp(w, h) == w) then
                    sample = image.scale(sample, size, h / w * size, interpolation[i])
                else
                    sample = image.scale(sample, w / h * size, size, interpolation[i])
                end
                input[k] = sample
            end
            return input
        end
    end
end

return M
