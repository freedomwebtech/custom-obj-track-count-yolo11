import cv2
from ultralytics import solutions

# Define the mouse callback function
def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:  # Check for mouse movement
        point = [x, y]
        print(f"Mouse moved to: {point}")

# Load the video
cap = cv2.VideoCapture("vid2.mp4")



# Define region points for object counting
region_points = [(308, 289), (439, 289)]

# Set up the object counter
counter = solutions.ObjectCounter(
    region=region_points,  # Pass region points
    model="best.pt",  # Model for object counting
    show_in=True,  # Display in counts
    show_out=True,  # Display out counts
    line_width=2,  # Adjust line width for display
)
# Set up the OpenCV window and mouse callback
cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)
count = 0
while True:
    # Read a frame from the video
    ret, frame = cap.read()
    if not ret:
        break

    count += 1
    if count % 2 != 0:  # Skip odd frames
        continue

    # Resize the frame for display
    frame = cv2.resize(frame, (1020, 500))

    # Process the frame with the object counter
    frame = counter.count(frame)

    # Show the frame
    cv2.imshow("RGB", frame)

    # Use cv2.waitKey(1) for real-time updates
    if cv2.waitKey(1) & 0xFF == ord("q"):  # Press 'q' to quit
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()
