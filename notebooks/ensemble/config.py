N_SEGMENTS = 50

LABEL_MAP = {
    'Not-Engaged': 0,
    'Barely-engaged': 1,
    'Engaged': 2,
    'Highly-Engaged': 3
}
SNP = 'SNP(Subject Not Present)'

MARLIN = 'marlin_features_large'    # MARLIN Only
GAZE_HP_AU = 'engage_gaze+hp+au'    # OpenFace Only
MEDIAPIPE = 'engage_bodypose'       # Mediapipe Only

FUSION = 'engage_gaze+hp+au_marlin' # OpenFace + MARLIN