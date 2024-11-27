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

# Triple Combine
FUSION_TRI = 'engage_gaze+hp+au_marlin_bodypose' # OpenFace + Mediapipe + MARLIN
FUSION_TRI_2 = 'engage_gaze+hp+au_bodypose_videollava' # OpenFace + Mediapipe + Videollava
FUSION_TRI_3 = 'engage_gaze+hp+au_marlin_videollava' # OpenFace + MARLIN + Videollava
FUSION_TRI_4 = 'engage_bodypose_marlin_videollava' # Mediapipe + MARLIN + Videollava

# Quad Combine
FUSION_QUAD = 'engage_gaze+hp+au_marlin_bodypose_videollava' # OpenFace + Mediapipe + MARLIN + Videollava