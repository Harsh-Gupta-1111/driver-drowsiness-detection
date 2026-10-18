import cv2

class Camera:
    def __init__(self, camera_index: int = 0, width: int = 640,height: int = 540):
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.cap = None
    
    def open(self):
        self.cap = cv2.VideoCapture(self.camera_index)

        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open the camera at index {self.camera_index}.")
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
    
    def read(self):
        if self.cap is None:
            raise RuntimeError('Camera not is opened. Call open() first')
        
        ret, frame = self.cap.read()

        if not ret or frame is None:
            raise RuntimeError("Failed to read frame from the webcam")

        return frame
    
    def release(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None