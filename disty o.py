import cv2
import numpy as np
from time import sleep

leftCamera = cv2.VideoCapture(0)
rightCamera = cv2.VideoCapture(1)

# Use StereoSGBM for better disparity computation
min_disp = -32  # Allow for negative disparity to detect closer objects
num_disp = 16 * 12  # Higher numDisparities for more depth levels (must be divisible by 16)
block_size = 15  # Block size for matching features

stereo = cv2.StereoSGBM_create(
    minDisparity=min_disp,
    numDisparities=num_disp,
    blockSize=block_size,
    P1=8 * 3 * block_size**2,
    P2=32 * 3 * block_size**2,
    disp12MaxDiff=1,
    uniquenessRatio=10,
    speckleWindowSize=200,  # Larger window to remove speckles more effectively
    speckleRange=32,
    preFilterCap=63,
    mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
)

# WLS filter for smoothing disparity map
wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=stereo)
wls_filter.setLambda(8000)
wls_filter.setSigmaColor(1.5)

while True:
    sleep(1/60)
    retL, leftFrame = leftCamera.read()
    retR, rightFrame = rightCamera.read()

    if not retL or not retR:
        break

    # Convert to grayscale
    leftFrame_gray = cv2.cvtColor(leftFrame, cv2.COLOR_RGB2GRAY)
    rightFrame_gray = cv2.cvtColor(rightFrame, cv2.COLOR_RGB2GRAY)

    # Compute disparity maps for both left and right frames
    disparity_left = stereo.compute(leftFrame_gray, rightFrame_gray).astype(np.float32) / 16.0
    right_matcher = cv2.ximgproc.createRightMatcher(stereo)
    disparity_right = right_matcher.compute(rightFrame_gray, leftFrame_gray).astype(np.float32) / 16.0

    # Apply WLS filter to smooth the disparity map
    disparity_filtered = wls_filter.filter(disparity_left, leftFrame_gray, disparity_map_right=disparity_right)

    # Replace NaN and Inf values with 0 before normalization
    disparity_filtered = np.nan_to_num(disparity_filtered, nan=0.0, posinf=0.0, neginf=0.0)

    # Normalize the disparity map for visualization
    disparity_filtered[disparity_filtered < 0] = 0  # Remove negative values
    disparity_visual = cv2.normalize(disparity_filtered, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

    # Clip and convert to uint8
    disparity_visual = np.clip(disparity_visual, 0, 255)
    disparity_visual = np.uint8(disparity_visual)

    # Apply a color map for better depth visualization
    disparity_color = cv2.applyColorMap(disparity_visual, cv2.COLORMAP_JET)

    # Display the depth map and the left frame for comparison
    cv2.imshow("Left Frame", leftFrame_gray)
    cv2.imshow("Disparity Map", disparity_color)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
leftCamera.release()
rightCamera.release()
cv2.destroyAllWindows()
