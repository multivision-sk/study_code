import cv2, dlib
import numpy as np

cap = cv2.VideoCapture('./video/rara.mp4')  # 동영상 파일 로드
#cap = cv2.VideoCapture(0)  # 은 웹캠이 켜지고 본인얼굴로 테스트 가능

# load overlap image
overlay = cv2.imread('./image/ryan.jpg', cv2.IMREAD_UNCHANGED)


scaler = 0.3

# 얼굴 detector 모듈 초기화
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

while True:
    ret, img = cap.read()  # 동영상 파일에서 frame 단위로 읽기
    if not ret:
        break

    img = cv2.resize(img, (int(img.shape[1] * scaler), int(img.shape[0] * scaler)))
    ori = img.copy()

    # detect faces
    faces = detector(img)
    face = faces[0]  # 찾은 모두 얼굴에서 첫번째 얼굴만 가져오기

    # 얼굴 특징점
    dlib_shape = predictor(img, face)
    shape_2d = np.array([[p.x, p.y] for p in dlib_shape.parts()])  # 연산 편리성을 위해 행렬화

    # 얼굴 size 계산
    top_left = np.min(shape_2d, axis=0)
    bottom_right = np.max(shape_2d, axis=0)


    center_x, center_y = np.mean(shape_2d, axis=0).astype(np.int)


    # visualize
    img = cv2.rectangle(img, pt1=(face.left(), face.top()), pt2=(face.right(), face.bottom()), color=(255, 255, 255),
                        thickness=2, lineType=cv2.LINE_AA)

    # 특징점 그리기 , 68개점
    for s in shape_2d:
        cv2.circle(img, center=tuple(s), radius=1, color=(0, 0, 255), thickness=2, lineType=cv2.LINE_AA)

    cv2.circle(img, center=tuple(top_left), radius=1, color=(255, 0, 0), thickness=2, lineType=cv2.LINE_AA) #얼굴 좌상단
    cv2.circle(img, center=tuple(bottom_right), radius=1, color=(255, 0, 0), thickness=2, lineType=cv2.LINE_AA) #얼굴 우하단
    cv2.circle(img, center=tuple((center_x, center_y)), radius=1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA) #얼굴 중심

    cv2.imshow('img', img)
    cv2.waitKey(1)  # 대기


