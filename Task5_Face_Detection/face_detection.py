import cv2

# Load Haar Cascade face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Load image
image = cv2.imread("Task5_Face_Detection/sample.jpg")

# Convert image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces
faces = face_cascade.detectMultiScale(
    gray,
    scaleFactor=1.2,
    minNeighbors=7,
    minSize=(80, 80)
)

# Print number of faces
print("Faces detected:", len(faces))

# Draw rectangles around faces
for (x, y, w, h) in faces:
    cv2.rectangle(
        image,
        (x, y),
        (x + w, y + h),
        (0, 255, 0),
        2
    )

# Show output image
cv2.imshow("Face Detection", image)

# Wait until key pressed
cv2.waitKey(0)

# Close window
cv2.destroyAllWindows()