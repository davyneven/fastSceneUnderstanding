local tnt = require 'torchnet'
local pl = (require 'pl.import_into')()
require 'paths'
require 'image'

--------------------------------------------------------------------
-- CLASS DEFINITION
-- -----------------------------------------------------------------

local Cityscapes = torch.class('tnt.Cityscapes', 'tnt.Dataset', tnt)

Cityscapes.classes = {
    'Unlabeled',
    'Road',
    'Sidewalk',
    'Building',
    'Wall',
    'Fence',
    'Pole',
    'TrafficLight',
    'TrafficSign',
    'Vegetation',
    'Terrain',
    'Sky',
    'Person',
    'Rider',
    'Car',
    'Truck',
    'Bus',
    'Train',
    'Motorcycle',
    'Bicycle'
}

Cityscapes.nClasses = #Cityscapes.classes

function Cityscapes.__init(self, parent_path, mode)
    assert(mode == 'train' or mode == 'val' or mode == 'test', 'Cannot load data in mode ' .. mode)

    self.csv_images = pl.data.read(paths.concat(parent_path, mode .. 'Images.txt'), {fieldnames = ''})

    if (mode ~= 'test') then
        self.csv_labels = pl.data.read(paths.concat(parent_path, mode .. 'Labels.txt'), {fieldnames = ''})
        self.csv_depth = pl.data.read(paths.concat(parent_path, mode .. 'Depth.txt'), {fieldnames = ''})
        self.csv_instances = pl.data.read(paths.concat(parent_path, mode .. 'Instances.txt'), {fieldnames = ''})

        assert((#self.csv_images == #self.csv_labels and #self.csv_labels == #self.csv_instances and #self.csv_instances == #self.csv_depth), 'Size of csv-files or not equal. (' .. #self.csv_images .. ',' .. #self.csv_labels .. ',' .. #self.csv_instances .. ',' .. #self.csv_depth .. ')')
    end

    self.parent_path = parent_path
    self.mode = mode
end

function Cityscapes.get(self, idx)
    assert(idx > 0 and idx <= #self.csv_images)

    -- Loading image
    local imgFile = self.csv_images[idx][1]
    local im = image.load(paths.concat(self.parent_path, imgFile), 3, 'float')
    local name = pl.utils.split(paths.basename(self.csv_images[idx][1], '.png'), '_leftImg8bit')[1]

    if (self.mode ~= 'test') then
        -- Loading segmentation map
        local labelFile = self.csv_labels[idx][1]
        local label = image.load(paths.concat(self.parent_path, labelFile), 1, 'byte')
        label = label + 2

        -- squeeze label to dim hxw
        label = label:squeeze()

        -- Loading depth map
        local depthFile = self.csv_depth[idx][1]
        local depth = image.load(paths.concat(self.parent_path, depthFile), 1, 'float')
        depth = depth:squeeze()

        -- Loading instance map
        local instanceFile = self.csv_instances[idx][1]
        local instances = image.load(paths.concat(self.parent_path, instanceFile), 1, 'float')
        instances = instances * (2 ^ 16 - 1) + 1
        instances = instances % 1000
        -- take only car instances for now
        instances[label:ne(15)] = 0
        instances = instances:byte()

        return {image = im, label = label, depth = depth, instances = instances, name = name}
    else
        return {image = im, name = name}
    end
end

function Cityscapes.size(self)
    return #self.csv_images
end

----------------------------------------------------------
-- LOCAL FUNCTIONS
----------------------------------------------------------

local getLabelHistogram = function(parent_path)
    local dataset = tnt.Cityscapes(parent_path, 'train')
    local histogram = torch.FloatTensor(Cityscapes.nClasses):zero()
    for i = 1, dataset:size() do
        histogram = histogram + torch.histc(dataset:get(i).label:float(), Cityscapes.nClasses, 1, Cityscapes.nClasses)
        print(i)
    end

    return histogram
end

Cityscapes.getClassWeights = function(parent_path)
    local weights
    if (paths.filep('datasets/weightsCityscapes.t7')) then
        weights = torch.load('datasets/weightsCityscapes.t7')
    else
        local histogram = getLabelHistogram(parent_path)
        -- set ignore class to zero
        histogram[1] = 0
        local normHist = histogram / histogram:sum()
        weights = torch.Tensor(Cityscapes.nClasses):fill(1)
        for i = 1, Cityscapes.nClasses do
            -- Ignore unlabeled and egoVehicle
            if histogram[i] < 1 then
                print('Class ' .. tostring(i) .. ' not found')
                weights[i] = 0
            else
                weights[i] = 1 / (torch.log(1.02 + normHist[i]))
            end
        end
        torch.save('datasets/weightsCityscapes.t7', weights)
    end
    return weights
end

local function unitTest()
    local tools = require('tools/tools')
    local dataset = tnt.Cityscapes('/esat/toyota/datasets/cityscapes', 'train')

    local win1, win2, win3, win4
    for i = 1, dataset:size() do
        local item = dataset:get(i)

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
