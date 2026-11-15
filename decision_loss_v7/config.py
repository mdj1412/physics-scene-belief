N_WORLDS=48
REPEATS=32
CONDITIONS=[('native','view'),('precise','view'),('native','left'),('native','right'),('native','center')]
ACTIONS=['view','left','right','center']
AUDIT=[0,1,16,17,32,33]
RASTER=[0,1,2,3,16,17,18,19,32,33,34,35]
COSTS={'equal':[0,0,0,0],'view_cheaper':[0,.002,.002,.002],'view_costlier':[.002,0,0,0]}
