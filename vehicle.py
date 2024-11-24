import cv2
import cv2.bgsegm
import numpy as np

# Video source/ web camera
cap = cv2.VideoCapture('video.mp4')

# Minimum dimensions for bounding boxes
min_width_rect = 80
min_height_rect = 80
min_area = 300  # Minimum area of contours

# Line position for vehicle counting
count_line_position = 550

# Background subtractor using bgsegm
algo = cv2.bgsegm.createBackgroundSubtractorMOG()

# Function to calculate the center of a bounding box
def center_handle(x, y, w, h):
    x1 = int(w / 2)
    y1 = int(h / 2)
    cx = x + x1
    cy = y + y1
    return cx, cy

detect = []
offset = 6  # Allowable error in pixels
counter = 0

while True:
    ret, frame1 = cap.read()
    if not ret:
        break

    # Convert frame to grayscale and apply Gaussian blur
    grey = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(grey, (3, 3), 5)

    # Apply background subtraction on each frame
    img_sub = algo.apply(blur)
    
    # Apply morphological operations to clean the image
    dilat = cv2.dilate(img_sub, np.ones((5, 5)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilatada = cv2.morphologyEx(dilat, cv2.MORPH_CLOSE, kernel)
    dilatada = cv2.morphologyEx(dilatada, cv2.MORPH_CLOSE, kernel)

    # Find contours in the processed image
    counterShape, h = cv2.findContours(dilatada, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Draw the counting line
    cv2.line(frame1, (25, count_line_position), (1200, count_line_position), (255, 127, 0), 3)
    
    # to draw rectangle
    for (i, c) in enumerate(counterShape):
        (x, y, w, h) = cv2.boundingRect(c)
        
        # Filter based on size and area
        area = w * h
        if area < min_area:
            continue

        validate_counter = (w >= min_width_rect) and (h >= min_height_rect)
        if not validate_counter:
            continue

        # Draw rectangle around valid vehicles
        cv2.rectangle(frame1, (x, y), (x + w, y + h), (0, 255, 0), 2)
        #cv2.putText(frame1 , "Vehicle"+str(counter) , (x,y-20) , cv2.FONT_HERSHEY_TRIPLEX, 1,(255,244,0),2)
        # Calculate the center of the detected vehicle
        center = center_handle(x, y, w, h)
        detect.append(center)

        # Draw a circle at the center
        cv2.circle(frame1, center, 4, (0, 0, 255), -1)

    # Vehicle counting logic
    for (x, y) in detect:
        if y < (count_line_position + offset) and y > (count_line_position - offset):
            counter += 1
            cv2.line(frame1, (25, count_line_position), (1200, count_line_position), (0, 127, 255), 3)
            detect.remove((x, y))
            print("Vehicle Counter: " + str(counter))

    # Display the vehicle count on the video frame
    cv2.putText(frame1, "VEHICLE COUNTER: " + str(counter), (450, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)

    # Show the processed frames
    cv2.imshow('Detector', dilatada)
    cv2.imshow('Video Original', frame1)

    # Break the loop if 'Enter' key is pressed
    if cv2.waitKey(1) == 13:
        break

# Release resources
cv2.destroyAllWindows()
cap.release()



