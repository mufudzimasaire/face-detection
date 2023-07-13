import cv2 as cv

def main(classifier: cv.CascadeClassifier) -> None:
  # Attach webcam
  camera = cv.VideoCapture(0)

  # Iterate through frames
  while True:
    # Read the current frame
    isRead, frame = camera.read()

    if isRead:
      detect_faces(classifier, frame)

      # Display frame in a window
      cv.imshow('Face detection', frame)

      # Close window when 'q' is pressed
      if cv.waitKey(1) == ord('q'):
        break
    else:
      print('Could not retrieve the frame')
      break

  # Clean up post-exit
  exit_clean_up(camera)


def detect_faces(classifier: cv.CascadeClassifier, frame: cv.UMat) -> None:
  # Convert frame to grayscale
  grayscale_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

  # Detect faces using grayscale frame
  face_coordinates = classifier.detectMultiScale(grayscale_frame)

  # Draw rectangles around detected faces
  for (x, y, w, h) in face_coordinates:
    cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 5)


def exit_clean_up(camera: cv.VideoCapture) -> None:
  camera.release()
  cv.destroyAllWindows()


main(
  # Use openCV pre-trained face frontals classifier
  cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')
)
