import numpy as np

SYNTHSEG_CODEMAP = {
    0:  "background",
    2:  "left_cerebral_white_matter", 
    3:  "left_cerebral_cortex",
    4:  "left_lateral_ventricle",
    5:  "left_inferior_lateral_ventricle",
    7:  "left_cerebellum_white_matter",
    8:  "left_cerebellum_cortex",
    10: "left_thalamus",
    11: "left_caudate",
    12: "left_putamen", 
    13: "left_pallidum", 
    14: "third_ventricle", 
    15: "fourth_ventricle", 
    16: "brain_stem", 
    17: "left_hippocampus",
    18: "left_amygdala",
    24: "csf", 
    26: "left_accumbens_area", 
    28: "left_ventral_dc", 
    41: "right_cerebral_white_matter",
    42: "right_cerebral_cortex",
    43: "right_lateral_ventricle",
    44: "right_inferior_lateral_ventricle",
    46: "right_cerebellum_white_matter", 
    47: "right_cerebellum_cortex", 
    49: "right_thalamus", 
    50: "right_caudate", 
    51: "right_putamen", 
    52: "right_pallidum", 
    53: "right_hippocampus",
    54: "right_amygdala",
    58: "right_accumbens_area", 
    60: "right_ventral_dc"
}


CONDITIONING_BIO = [
    "cerebral_cortex", 
    "hippocampus", 
    "amygdala",
    "cerebral_white_matter",
    "lateral_ventricle"
]

CONDITIONING_DG = [
    "AGE",
    "APOE4",
    "sex",
]

CONDITIONING_DX = [
    "diagnosis"
]

CONDITIONING_VARIABLES = CONDITIONING_DG + CONDITIONING_DX + CONDITIONING_BIO

LATENT_SHAPE_DM = (3, 16, 20, 16)

AGE_MIN, AGE_MAX, AGE_DELTA = 0, 100, 100
SEX_MIN, SEX_MAX, SEX_DELTA = 1, 2, 1
DIA_MIN, DIA_MAX, DIA_DELTA = 1, 3, 2

# cerebral_cortex 490000 5750000
# hippocampus 5000 14000
# amygdala 1700 5600
# cerebral_white_matter 449000 751000
# lateral_ventricle 313000 179000
