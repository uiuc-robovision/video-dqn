#! /bin/ruby

# To compute the navmeshes for different sized agents
# I modified the agentHeight and agentMaxClimb values in the  'setDefaults()' function in 
# src/esp/nav/PathFinder.h in the habitat-sim repo
# (see https://github.com/facebookresearch/habitat-sim/blob/v0.1.4/src/esp/nav/PathFinder.h line 93)
# after that you rebuild the project using 
# `python setup.py --headless --build-datatool` 
# which creates the datatool used in this script below
# see src/utils/datatool/datatool.cpp (https://github.com/facebookresearch/habitat-sim/blob/master/src/utils/datatool/datatool.cpp)
# in the habitat-sim repo for details about what else the tool can do
# We used agent height 1.25 to allow it to get under some doorways
# and max climb 0.05 to disallow stairs

folder = '/data/gibson'

files =  `ls #{folder} | grep .glb`.split()
names = files.map {|e|  e.match(/(.*).glb$/)[1] }
puts names
names.each do |e|
  system("/data/python_packages/habitat-sim/build/utils/datatool/datatool create_navmesh #{folder}/#{e}.glb #{folder}/#{e}.navmesh")
end
