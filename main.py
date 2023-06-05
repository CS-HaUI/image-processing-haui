


from process.function import FaceRecognition
obj = FaceRecognition()
obj.val()



# obj.val()
# import cv2
# import numpy as np
# video = cv2.VideoCapture(0)


# obj1 = cv2.resize(cv2.imread("data\\obj1.png"), (150, 50))
# obj2 = cv2.resize(cv2.imread("data\\obj2.png"), (150, 50))
# obj3 = cv2.resize(cv2.imread("data\\obj3.png"), (150, 50))
# obj4 = cv2.resize(cv2.imread("data\\obj4.png"), (150, 50))
# obj5 = cv2.resize(cv2.imread("data\\obj5.png"), (150, 50))
# obj6 = cv2.resize(cv2.imread("data\\obj6.png"), (150, 50))

# objs = [obj1, obj2, obj3, obj4, obj5, obj6]


# num_obj = int(open('data\\objects\\log.txt', 'r').readline())
# print(num_obj)




# index = 0
# while 1:
#     _, frame = video.read()
#     if _ is not None:
#         frame = cv2.resize(frame, (1280, 720))
#         origin = frame.copy()
#         rows, cols, channels = frame.shape 
        
#         mask = np.zeros_like(frame)
#         x,y = cols//2, rows//2
        
#         cv2.circle(mask, (x,y), 300, (255,255,255), -1)
        
#         ROI = cv2.bitwise_and(frame, mask)
#         cv2.circle(mask, (x,y), 300, (255,0,0), 10)
        
#         mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
#         x, y, w, h = cv2.boundingRect(mask)
#         result = ROI[y-20:y+h+20,x-20:x+w+20]
#         mask = mask[y-20:y+h+20,x-20:x+w+20]
#         result[mask==0] = (255,255,255)
        
        
#         result[0:50, 0:150] = objs[index]
        
#         crop = origin[y:y+h, x:x+w]
#         cv2.imshow('a', result)
#         if cv2.waitKey(1) & 0xff == ord('s'):
#             if index == 5:
#                 index = 0
#                 break
#             cv2.imwrite(f"{index}.png", crop)
#             index+=1
#             if index == 6:
#                 index = 0
#     else:
#         break
    
# video.release()
# cv2.destroyAllWindows()
