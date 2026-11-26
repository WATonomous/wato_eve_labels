"""One class ontology for the whole pipeline.

The BEV map head has six classes, but only four are meaningful as *pixels* in a
camera image: you cannot reliably see a stop line or a lane divider at 320x256,
and carpark is drivable as far as a planner is concerned. So the 2D branch works
in four classes and the BEV stage expands them.
"""
CLASSES = ["other", "drivable", "walkway", "crossing"]
NUM_CLASSES = len(CLASSES)
IGNORE = 255

# BEV channel order used by the BEVFusion map head (do not reorder)
BEV_CLASSES = ["drivable_area", "ped_crossing", "walkway", "stop_line",
               "carpark_area", "divider"]

# 2D class -> BEV channels it fills
BEV_EXPAND = {
    "drivable": ["drivable_area", "carpark_area"],
    "walkway": ["walkway"],
    "crossing": ["ped_crossing"],
}
