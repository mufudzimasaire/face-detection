import cv2

def main(algorithm):
  # Attach webcam
  camera = cv2.VideoCapture(0)

  # Iterate through frames
  while True:
    # Read the current frame
    frame_read, frame = camera.read()

    if frame_read:
      # Convert frame to grayscale
      grayscale_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

      # Detect faces using grayscale frame
      face_coordinates = algorithm.detectMultiScale(grayscale_frame)

      # Draw rectangles around detected faces
      for (x, y, w, h) in face_coordinates:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 5)

      # Display frame in a window
      cv2.imshow('Face detection', frame)

      # Close window when 'q' is pressed
      if cv2.waitKey(1) == ord('q'):
        break
    else:
      print('Could not retrieve the frame')
      break

  # Clean up post-exit
  camera.release()
  cv2.destroyAllWindows()


main(
  # Use openCV pre-trained face frontals classifier
  cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
)
