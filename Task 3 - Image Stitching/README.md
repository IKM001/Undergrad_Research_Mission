# Undergrad_Research_Mission
학부연구생 과제 저장소입니다.

# Homography 변환을 이용해 파노라마 이미지 만들기 

### 1. 제공된 코드 분석

과제 해결을 위해 `main.py`에 정의된 핵심 함수와 코드 구조를 분석했습니다.

#### A. `trim(frame)` 함수 분석

`warpPerspective` 적용 후 이미지 외곽에 생기는 검은 여백(빈 영역)을 제거하기 위한 재귀 함수입니다.

* **동작 원리**
  - 이미지의 상하좌우 가장자리 행/열의 픽셀 합(`np.sum`)이 0인지 확인합니다.
  - 픽셀 합이 0이라는 것은 해당 행/열이 전부 검정(빈 영역)임을 의미합니다.
  - 빈 행/열이 발견되면 해당 부분을 슬라이싱으로 제거하고 재귀 호출하여, 유효한 픽셀이 있는 영역만 남을 때까지 반복합니다.

#### B. 베이스라인 코드 구조 분석

##### B-1. 이미지 로드
```python
l_img = cv2.imread('1.jpg', -1)
l_img_gray = ___

r_img = cv2.imread('2.jpg', -1)
r_img_gray = ___
```
* `-1` 플래그로 원본 컬러 이미지를 그대로 읽습니다.
* `l_img_gray`, `r_img_gray`는 이후 ORB 특징점 추출을 위해 Grayscale로 변환한 이미지를 담을 변수입니다. ORB 알고리즘은 Grayscale 입력만을 처리할 수 있기 때문에 변환이 필요합니다.
* 최종 스티칭 결과는 컬러여야 하므로 원본 컬러 이미지(`l_img`, `r_img`)도 따로 보관합니다.

##### B-2. ORB 키포인트 & 디스크립터 추출
```python
orb = cv2.ORB_create()
kp1, desc1 = ___
kp2, desc2 = ___
```
* ORB(Oriented FAST and Rotated BRIEF)는 FAST로 키포인트를 검출하고, BRIEF로 디스크립터를 생성하는 알고리즘입니다. SIFT/SURF 대비 특허 문제가 없고 속도가 빠릅니다.
* ORB 객체는 생성되어 있으나, 두 이미지 각각에 대해 키포인트(`kp`)와 디스크립터(`desc`)를 동시에 추출하는 함수 호출이 비어있습니다.

##### B-3. BFMatcher를 이용한 특징점 매칭
```python
matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
matches = ___
```
* BFMatcher(Brute-Force Matcher)는 한쪽 이미지의 모든 디스크립터를 다른 쪽 이미지의 모든 디스크립터와 전수 비교하여 가장 유사한 쌍을 찾습니다.
* ORB는 이진(Binary) 디스크립터이므로 유클리드 거리 대신 `NORM_HAMMING`(해밍 거리)을 사용합니다.
* `matcher` 객체는 생성되어 있으나, 실제 두 디스크립터(`desc1`, `desc2`) 간의 매칭을 수행하는 함수 호출이 비어있습니다.

##### B-4. RANSAC을 이용한 Homography 행렬 추정
```python
left_pts = ___
right_pts = ___
r2l_H, _ = cv2.findHomography( , , cv2.RANSAC, 5.0)
```
* 매칭 결과(`matches`)에서 각 매칭 쌍의 `queryIdx`(왼쪽 이미지 키포인트 인덱스)와 `trainIdx`(오른쪽 이미지 키포인트 인덱스)를 이용해 실제 픽셀 좌표를 추출하는 부분이 비어있습니다.
* `findHomography`는 두 이미지 간의 Homography 행렬(자유도 8, 최소 4개 매칭쌍 필요)을 추정합니다. `cv2.RANSAC`으로 잘못된 매칭점(outlier)을 자동 제거하고, `5.0`은 outlier 판별 임계값(픽셀 단위)입니다.
* `r2l_H`라는 변수명에서 알 수 있듯이 **오른쪽->왼쪽** 방향의 변환 행렬을 구해야 하므로 인자 순서가 중요합니다.

##### B-5. 오른쪽 이미지 투영 (warpPerspective)
```python
i_size = (l_img.shape[1]+r_img.shape[1], l_img.shape[0])
stitched_image = ___
```
* `i_size`는 두 이미지의 너비를 합산한 크기의 캔버스로, 오른쪽 이미지가 투영될 공간을 확보합니다.
* 추정된 Homography 행렬(`r2l_H`)을 이용해 오른쪽 이미지를 왼쪽 이미지의 좌표계로 원근 투영하는 함수 호출이 비어있습니다. 이 시점에서 캔버스 왼쪽 절반은 비어있는 상태입니다.

##### B-6. 왼쪽 이미지 합성
```python
stitched_image[0:l_img_gray.shape[0], 0:l_img_gray.shape[1]] = ___
```
* 투영 후 비어있는 캔버스 왼쪽 영역에 왼쪽 원본 이미지를 직접 대입하여 최종 파노라마를 완성합니다.
* 컬러 결과물을 위해 grayscale이 아닌 원본 컬러 이미지(`l_img`)를 채워야 합니다.