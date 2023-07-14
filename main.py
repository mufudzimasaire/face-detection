import cv2 as cv

# FaceDetector
# 
# This is a simple Python3 program that demonstrates
# how openCV can be used for face detection. 
class FaceDetector:
  def __init__(self, classifier: cv.CascadeClassifier) -> None:
    self.classifier = classifier

  def setClassifier(self, classifier: cv.CascadeClassifier) -> None:
    self.classifier = classifier

  def getClassifier(self) -> cv.CascadeClassifier:
    return self.classifier

  def run(self) -> None:
    # Attach webcam
    camera = cv.VideoCapture(0)

    # Iterate through frames
    while True:
      # Read the current frame
      isRead, frame = camera.read()

      if isRead:
        self.__detectFaces(frame)

        # Display frame in a window
        cv.imshow('Face detection', frame)

        # Close window when 'q' is pressed
        if cv.waitKey(1) == ord('q'):
          break
      else:
        print('Could not retrieve the frame')
        break

    # Clean up post-exit
    self.stop(camera)

  def stop(self, camera: cv.VideoCapture) -> None:
    camera.release()
    cv.destroyAllWindows()

  def __detectFaces(self, frame: cv.UMat) -> None:
    # Convert frame to grayscale
    grayscale_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # Detect faces using grayscale frame
    face_coordinates = self.classifier.detectMultiScale(grayscale_frame)

    # Draw rectangles around detected faces
    for (x, y, w, h) in face_coordinates:
      cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 5)


FaceDetector(
  # Use openCV pre-trained face frontals classifier
  cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')
).run()
