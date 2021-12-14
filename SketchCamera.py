import cv2
import numpy as np
from subprocess import call
from multiprocessing import Pool, Value, cpu_count 
import gui
import os

try:
    num_cpu = cpu_count()
except:
    num_cpu = os.cpu_count()

"""constant"""
k = 0.1
o = 0.8

def animation(target, original, sketch):
    target = (1-o)*original + o*sketch
    return clipping(target)

def clipping(target):
    if target > 255:
        target = 255
    elif target < 0:
        target = 0
    return target

def compute_edge(img):
    pool = Pool(num_cpu-1)
    e = img.copy()
    w = img.shape[0]
    h = img.shape[1]
    alpha = pool.map(compute_alpha, iterable=[img])[0]
    for x in range(w):
        for y in range(h):
            if e[x, y] < alpha:
                e[x, y] = 0
            else:
                e[x, y] = 255
    return e
def compute_histogram(img):
    diction = {}
    w = img.shape[0]
    h = img.shape[1]
    for x in range(256):
        diction[x]=0 
    for x in range(w):
        for y in range(h):
            v = img[x ,y]
            diction[v] = diction.get(v) + 1
            
    return diction

def compute_alpha(img):
    histogram = compute_histogram(img)
    n = img.shape[0] * img.shape[1]
    val = 0.0
    for x in range(256):
        val = val + histogram[x]/n
        if val >= k:
            return x


def sketch_filter(img):
    w = img.shape[0]
    h = img.shape[1]
    ker = img.copy()
    for x in range(w):
        for y in range(h):
            lis = []
            if (x>0 and y>0) and (x<w-1 and y<h-1):
                for a in range(-1, 2, 1):
                    for b in range(-1, 2, 1):
                        lis.append(ker[x-a,y-b])
                img[x, y] =  animation(img[x, y], img[x, y], 255 * ker[x,y] / max(lis))
    return img

'''카메라 구동함수'''
def video_capture():
    try:
        print("카메라를 구동합니다.")
        cap=cv2.VideoCapture(0)
    except:
        print("카메라 구동실패.")
        return
    cap.set(3, 480)
    cap.set(4, 320)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print('비디오 읽기 오류')
            break
        frame = cv2.flip(frame,1)

        cv2.imshow("INPUT ESC", frame)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            return frame
            break
        
    cap.release()
    cv2.destroyWindow("frame1")



def main():
    
    frame = video_capture()
    num, value = gui.main()
    pool = Pool(num_cpu-1)
    '''bold color Pencil'''
    if num ==1:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        
        v = pool.map(func = sketch_filter, iterable = [v] )[0]
        pool.close()
        pool.join()

        img = cv2.merge((h, s, v))
        img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
        cv2.imshow("Any KEY", img)
        cv2.waitKey(0) & 0xFF
        cv2.destroyWindow("Any KEY")
    
    '''color Pencil'''
    # pecil_img = pencil_filter(frame)
    if num == 0:
        b, g, r = cv2.split(frame)
        b, g, r = pool.map(func  = sketch_filter, iterable = (b, g, r))
        pool.close()
        pool.join()
        bold_color_pencil = cv2.merge((b, g, r))
        cv2.imshow("Any KEY", bold_color_pencil)
        cv2.waitKey(0) & 0xFF
        cv2.destroyWindow("Any KEY")
   
    '''gray sketch filter'''
    if num == 2:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = pool.map(func = sketch_filter, iterable=[gray])[0]
        gray = compute_edge(gray)
        pool.close()
        pool.join()
        cv2.imshow("Any KEY", gray)

        k = cv2.waitKey(0) & 0xFF
        cv2.destroyAllWindows()
    try:
        targetDirectory = os.path.expanduser('~')+"/Desktop"
        call(["open", targetDirectory])
        if num ==1:
            cv2.imwrite(os.path.join(targetDirectory, "bold_color_pencil.jpeg"), img)
        elif num == 0:
            cv2.imwrite(os.path.join(targetDirectory, "color_pencil.jpeg"), bold_color_pencil)
        elif num == 2:
            cv2.imwrite(os.path.join(targetDirectory, "pencil_sketch.jpeg"), gray)
    except:
        print("사진저장이 실패하였습니다.")
if __name__ == "__main__":
    main()