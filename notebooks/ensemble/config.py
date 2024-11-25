N_SEGMENTS = 50

LABEL_MAP = {
    'Not-Engaged': 0,
    'Barely-engaged': 1,
    'Engaged': 2,
    'Highly-Engaged': 3
}
SNP = 'SNP(Subject Not Present)'

# Single data
GAZE_HP_AU = 'engage_gaze+hp+au'    # OpenFace Only
MARLIN = 'marlin_features_large'    # MARLIN Only
MEDIAPIPE = 'engage_bodypose'       # Mediapipe Only
VIDEOLLAVA = 'video_llava'          # VideoLlava Only

# Double data
FUSION = 'engage_gaze+hp+au_marlin'     # OpenFace + MARLIN
FUSION_2 = 'engage_gaze+hp+au_bodypose' # OpenFace + Mediapipe
FUSION_3 = 'engage_bodypose_marlin'     # MARLIN + Mediapipe

# Not Used Yet, Triple Combine
FUSION_TRI = 'engage_gaze+hp+au_marlin_bodypose' # OpenFace + Mediapipe + MARLIN
