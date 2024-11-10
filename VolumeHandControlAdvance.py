import cv2
import time
import numpy as np
import HandTrackingModule as htm
import math
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

# Camera Settings
wCam, hCam = 640, 480

# Initialize Video Capture
cap = cv2.VideoCapture(0)  # Try 0 first, if that doesn't work try 1
if not cap.isOpened():
    cap = cv2.VideoCapture(1)  # Try alternative camera index
if not cap.isOpened():
    raise IOError("Cannot open webcam")

cap.set(cv2.CAP_PROP_FRAME_WIDTH, wCam)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, hCam)

# Initialize Variables
pTime = 0
detector = htm.handDetector(detectionCon=0.7, maxHands=1)

# Audio Setup
try:
    devices = AudioUtilities.GetSpeakers()
    interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    volume = cast(interface, POINTER(IAudioEndpointVolume))
    volRange = volume.GetVolumeRange()
    minVol, maxVol = volRange[0], volRange[1]
except Exception as e:
    print(f"Error initializing audio: {e}")
    exit(1)

# Initialize Volume Variables
vol = 0
volBar = 400
volPer = 0
area = 0
colorVol = (255, 0, 0)
smoothness = 10


def draw_volume_bar(img, volBar, volPer, colorVol, cVol, fps):
    # Draw volume bar
    cv2.rectangle(img, (50, 150), (85, 400), (255, 0, 0), 3)
    cv2.rectangle(img, (50, int(volBar)), (85, 400), (255, 0, 0), cv2.FILLED)

    # Draw volume percentage
    cv2.putText(img, f'{int(volPer)} %', (40, 450), cv2.FONT_HERSHEY_COMPLEX,
                1, (255, 0, 0), 3)

    # Draw current volume
    cv2.putText(img, f'Vol Set: {int(cVol)}', (400, 50), cv2.FONT_HERSHEY_COMPLEX,
                1, colorVol, 3)

    # Draw FPS
    cv2.putText(img, f'FPS: {int(fps)}', (40, 50), cv2.FONT_HERSHEY_COMPLEX,
                1, (255, 0, 0), 3)


try:
    while True:
        success, img = cap.read()
        if not success or img is None:
            print("Error: Could not read frame")
            break

        # Find Hand
        img = detector.findHands(img)
        lmList, bbox = detector.findPosition(img, draw=True)

        if len(lmList) != 0:
            # Filter based on size
            area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) // 100

            if 250 < area < 1000:
                # Find Distance between index and Thumb
                try:
                    length, img, lineInfo = detector.findDistance(4, 8, img)

                    # Convert Volume
                    volBar = np.interp(length, [50, 200], [400, 150])
                    volPer = np.interp(length, [50, 200], [0, 100])

                    # Reduce Resolution to make it smoother
                    volPer = smoothness * round(volPer / smoothness)

                    # Check fingers up
                    fingers = detector.fingersUp()

                    # If pinky is down set volume
                    if not fingers[4]:
                        try:
                            volume.SetMasterVolumeLevelScalar(volPer / 100, None)
                            cv2.circle(img, (lineInfo[4], lineInfo[5]), 15,
                                       (0, 255, 0), cv2.FILLED)
                            colorVol = (0, 255, 0)
                        except Exception as e:
                            print(f"Error setting volume: {e}")
                    else:
                        colorVol = (255, 0, 0)
                except Exception as e:
                    print(f"Error in hand tracking: {e}")

        # Calculate FPS
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        # Get current volume
        try:
            cVol = int(volume.GetMasterVolumeLevelScalar() * 100)
        except:
            cVol = 0

        # Draw all elements
        draw_volume_bar(img, volBar, volPer, colorVol, cVol, fps)

        # Show image
        cv2.imshow("Volume Control", img)

        # Break loop with 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except Exception as e:
    print(f"An error occurred: {e}")

finally:
    # Clean up
    cap.release()
    cv2.destroyAllWindows()