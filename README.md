Vehicle Counting and Detection Using OpenCV

Project Overview

This project implements vehicle detection and counting from a video feed using OpenCV. The program uses background subtraction techniques to detect moving vehicles and counts them as they cross a defined line. The project is optimized for detecting vehicles on a road while minimizing false positives (non-vehicle objects).

Features

-Vehicle Detection: Identifies moving vehicles from a video feed using background subtraction.

-Vehicle Counting: Counts vehicles that cross a predefined line in the video.

-Background Subtraction: Uses the cv2.bgsegm or cv2.createBackgroundSubtractorMOG2() algorithm for detecting motion.

-Noise Reduction: Applies morphological operations and size filtering to minimize false detections.

-Dynamic Tracking: Tracks detected vehicles and avoids recounting.
