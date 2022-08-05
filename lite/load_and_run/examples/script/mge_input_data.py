#!/usr/bin/env python3
import numpy as np
import cv2
import megengine.data.transform as T
import megengine.functional as F
import json
import urllib
url, filename = ("https://data.megengine.org.cn/images/cat.jpg", "input_data/cat.jpg")
try: urllib.URLopener().retrieve(url, filename)
except: urllib.request.urlretrieve(url, filename)

# numpy data
data=np.random.rand(1,3,224,224)
np.save("input_data/resnet50_input_uint8.npy",data.astype(np.uint8))
np.save("input_data/resnet50_input.npy",data.astype(np.float32))

#ppm data
image = cv2.imread("input_data/cat.jpg")
transform = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
])
processed_img = transform.apply(image)
cv2.imwrite("input_data/cat.ppm",processed_img)

#json 
data_obj = {
    "shape": [1,3],
    "type": "float32",
    "raw": [2,3,4]
}
with open("input_data/add_demo_input.json", "w") as f:
     json.dump({"data":data_obj},f)