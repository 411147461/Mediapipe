import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0) # 讀取視訊鏡頭，cv2.VideoCapture(鏡頭編號)
mpHands = mp.solutions.hands # 使用mediapipe手部模型
hands = mpHands.Hands() # mediapipe手部模型參數default: static_image_mode=False, max_num_hands=2, model_complexity=1, min_detection_confidence=0.5, min_tracking_confidence=0.5
mpDraw = mp.solutions.drawing_utils # 畫出點座標的function

# 手部辨識點&線的樣式(BGR)
handLmsStyle = mpDraw.DrawingSpec(color=(0, 0, 255), thickness = 5)
handConStyle = mpDraw.DrawingSpec(color=(0, 255, 0), thickness = 5)

pTime = 0
cTime = 0

while True:
    ret, img = cap.read() # ret(retval):是否擷取成功，img:擷取的影像
    if ret:
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # cv2 BGR轉RGB
        result = hands.process(imgRGB) # 偵測手的21個關鍵點
        # print(result.multi_hand_landmarks)
        """
        # 抓視窗大小
        imgHeight = img.shape[0]
        imgWidth = img.shape[1]
        """
        # 畫出21個關鍵點和連線
        if result.multi_hand_landmarks: # 如果偵測到手
            for handLms in result.multi_hand_landmarks: # for迴圈跑過偵測到的所有的手
                mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS, handLmsStyle, handConStyle) # 畫出關鍵點和連線
                """
                # 印出21個關鍵點編號
                for i, lm in enumerate(handLms.landmark):
                    xPos = int(lm.x * imgWidth)
                    yPos = int(lm.y * imgHeight)

                    cv2.putText(img, str(i), (xPos-24, yPos+5), cv2.FONT_ITALIC, 0.4, (0, 0, 255), 2)
                    print(i, xPos, yPos) # result.multi_hand_landMarks 手部21個點座標
                """

        # 印出fps
        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime
        cv2.putText(img, f"FPS:{int(fps)}", (30, 50), cv2.FONT_ITALIC, 1, (0, 0, 0), 2)

        cv2.imshow('Webcam', img) # 顯示圖片，Webcam是視窗名稱

    # 按下q跳出while
    if cv2.waitKey(1) == ord('q'):
        break