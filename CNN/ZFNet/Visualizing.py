from __future__ import print_function
import time
import os
os.chdir("d:/assignment")
import json
import matplotlib.pyplot as plt
import tensorflow as tf
from utils.classifiers.squeezenet import SqueezeNet
from utils.data_utils import load_tiny_imagenet
from utils.image_utils import SQUEEZENET_MEAN, SQUEEZENET_STD
from utils.data_utils import load_imagenet_val
from scipy.ndimage.filters import gaussian_filter1d
import numpy as np
from utils.image_utils import preprocess_image, deprocess_image
import PIL.Image as pilimg

plt.rcParams['figure.figsize'] = (10.0, 8.0)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

def get_session():
    '''
    ## fuction ##
    session return, 설정
    '''
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = False
    session = tf.Session(config=config)
    return session

tf.reset_default_graph() # 기존 존재하던 graph 삭제 (ram)
sess = get_session() # session 실행
SAVE_PATH = 'utils/datasets/squeezenet.ckpt' # pre-trained model
model = SqueezeNet(save_path=SAVE_PATH, sess=sess)

### load imagenet ###

X_raw, y, class_names = load_imagenet_val(num=5) # image net value 5장

def imageNetVis():
    '''
    function:
        5장 imagenet 시각화
        
    '''
    plt.figure(figsize=(12,6))
    for i in range(5):
        plt.subplot(1, 5, i +1)
        plt.imshow(X_raw[i])
        plt.title(class_names[y[i]])
        plt.axis('off')
    plt.gcf().tight_layout()
    
def blurImage(X, sigma=1):
    '''
    function:
        using gaussian filter, blur images
    
    input:
        - X: image input
        - sigma: 흐리는 정도
    
    return:
        - X: image blurred
    '''
    X = gaussian_filter1d(X, sigma, axis=1)
    X = gaussian_filter1d(X, sigma, axis=2)
    return X

def computeMapping(X, y, model):
    '''
    function:
        X와 labely에서 작용하는 feature를 계산
    
    input:
        - X: input images (N, H, W, 3)
        - y: Labels shape (N, )
        - model: squeeze Net
    
    Return:
        - map: (N, H, W)
    '''
    mapping = None
    correct_scores = tf.gather_nd(model.classifier, 
                                  tf.stack((tf.range(X.shape[0]), model.labels), axis=1))
    dX = tf.gradients(correct_scores, model.image) # 역 그라디언트 계산
    mapping_abs = tf.abs(dX) # 절댓값 --> 음수는 활성화되지 않은 값
    mapping_max = tf.reduce_max(mapping_abs, axis=4) # rgb중 가장 활성화된 값
    mapping_squeeze = tf.squeeze(mapping_max)
    mapping = sess.run(mapping_squeeze, feed_dict={model.image:X, model.labels:y})
    return mapping
    
def visualizeMapping(X, y, mask):
    '''
    function:
        시각화 함수
    
    input:
        - X: input image
        - y: input class
        - mask: 자극받은 해당 idx
    '''
    mask = np.asarray(mask)
    Xm = X[mask]
    ym = y[mask]
    mapping = computeMapping(Xm, ym, model)
#    mapping = mapping.reshape(1, 224, 224)
    print(mapping.shape)
    for i in range(mask.size):
        plt.subplot(2, mask.size, i+1)
        plt.imshow(deprocess_image(Xm[i]))
        plt.axis('off')
        plt.title(class_names[ym[i]])
        plt.subplot(2, mask.size, mask.size + i +1)
        plt.title(mask[i])
        plt.imshow(mapping[i])
        plt.axis('off')
        plt.gcf().set_size_inches(10,4)
    plt.show()
    return mapping
    
X = np.array([preprocess_image(img) for img in X_raw])
mask = np.arange(5)
mapping = visualizeMapping(X, y, mask)

test_image = np.array(pilimg.open("test.jpg")).reshape(1,224, 224, 3)
test_image = preprocess_image(test_image)
y_hat = sess.run(model.classifier, feed_dict={model.image:test_image})
np.argmax(y_hat)
idx = np.array([664])
mask = np.arange(1)
mapping = visualizeMapping(test_image, idx, mask)

def createClassVisualization(target_y, model, **kwargs):
    '''
    function:
     학습된 모델에서 해당 클래스에 최대로 점수화하는 이미지를 생성   
    inputs:
        - target_y: 해당 클래스 one-hot
        - model: pretrained model (squeezeNet)
    returns:
    
    
    '''
    l2_reg = kwargs.pop('l2_reg', 1e-3) # numpy, 변수 저장
    learning_Rate = kwargs.pop('learning_rate', 25) # 굉장히 큰수로 학습
    num_iterations = kwargs.pop('num_iteration', 100)
    blur_every = kwargs.pop('blur_every', 10)
    max_jitter = kwargs.pop('max_jitter', 16)
    show_every = kwargs.pop('show_every', 25)
    
    X = 255 * np.random.rand(224, 244, 3) # 학습 image (random init in pixel range)
    X = preprocess_image(X)[None] 

































