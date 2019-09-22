# Copyright 2018 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""A client that performs inferences on a ResNet model using the REST API.

The client downloads a test image of a cat, queries the server over the REST API
with the test image repeatedly and measures how long it takes to respond.

The client expects a TensorFlow Serving ModelServer running a ResNet SavedModel
from:

https://github.com/tensorflow/models/tree/master/official/resnet#pre-trained-model

The SavedModel must be one that can take JPEG images as inputs.

Typical usage example:

    resnet_client.py
"""

from __future__ import print_function

import base64
import requests
import numpy as np
#import cv2
import json
import time
from PIL import Image



# The server URL specifies the endpoint of your server running the ResNet
# model with the name "resnet" and using the predict interface.
SERVER_URL = 'http://34.69.138.197:8501/v1/models/my_model:predict'

# The image URL is the location of the image we should send to the server


def main(image,output_img):
  # Download the image
  #img=cv2.imread(image)
  #img = Image.open(image)
  
  img=image
  print(type(img))
  img = np.array(img)
  print(img.shape)
  X=np.zeros((1,img.shape[0],img.shape[1],3))
  X[0,:,:,:]=1.0*img[:,:,:3]/255
  prediction=manageImage(X)
  prediction=prediction[0]

  #cv2.imwrite(output_img,prediction*255)   
  return prediction*255  


def singleRequest(X):
  data = json.dumps({"signature_name": "serving_default", "instances": X.tolist()})
  print("waiting for server to respond")
  t1=time.time()
  response = requests.post(SERVER_URL, data=data)
  print("done! it took ",time.time()-t1,"seconds")
  response.raise_for_status()
  prediction = response.json()['predictions']

  #print('Prediction class:{}'.format(prediction))
  prediction=np.asarray(prediction)
  return prediction


def manageImage(X):
    final_X=X
    num_columns=X.shape[2]
    block_size=(400000/num_columns)
    print(block_size)
    num_rows=X.shape[1]
    print("number of requests:",int(num_rows/block_size))
    if num_rows>block_size:
       num_full_blocks=int(num_rows/block_size)
       for block_start in range(num_full_blocks):
           print("done for: ",block_start)
           final_X[:,block_start*block_size:block_start*block_size+block_size,:,:]=\
           singleRequest(X[:,block_start*block_size:block_start*block_size+block_size,:,:])
    else:
      final_X=singleRequest(X)
    return final_X
           
           

   







