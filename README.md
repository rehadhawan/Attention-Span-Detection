# Attention-Span-Detetction

## Head Pose Estimation Chatbot

This is a Python program that uses the MediaPipe Face Mesh model to estimate the head pose of a person captured through a webcam. The program calculates the head's yaw and pitch angles to determine the user's attentiveness level during the conversation. The chatbot classifies users into three attentiveness categories: "Attentive," "Partially Attentive," and "Not Attentive."

### Prerequisites

Before running the program, make sure you have the following libraries installed:

- OpenCV (`cv2`)
- Mediapipe (`mediapipe`)
- NumPy (`numpy`)
- Pandas (`pandas`)

You can install these libraries using `pip`

### Usage
Clone the repository or download the code.

Open a terminal or command prompt and navigate to the project directory.

Run the Python script:

``` python head_pose_estimation.py ```

The webcam feed will open, and the chatbot will analyze the user's head pose and attentiveness level in real-time. The chatbot's response will be displayed on the screen.

Press the 'q' key to quit the program and save the attentiveness data to an Excel file (koordinat.xlsx).

#### Note
The program is for demonstration purposes and may not be accurate for all scenarios. The attentiveness assessment is based on simple head pose estimation and may not capture all aspects of attentiveness.
