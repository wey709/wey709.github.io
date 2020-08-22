---
layout: post
title: "논문 공부: Deep learning for event-driven stock prediction"
date: 2020-07-15
excerpt: cnn, transfer learning, cut-out
tags: [deep learning, CNN, practice]
comments: true
---


```python
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import InceptionResNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras import backend as K
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import SGD, Adam
```


```python
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
```


```python
import random
import cv2
import os
```


```python
# os.environ['CUDA_VISIBLE_DEVICES']='1,2'
os.environ['CUDA_VISIBLE_DEVICES']='2,3'
```


```python
with tf.device('/GPU:2'):
    data = np.load('image_array.npy')
    transformed_label = np.load('label_array.npy')
```


```python
with tf.device('/GPU:2'):
    encoder = LabelBinarizer()
    transformed_label = encoder.fit_transform(transformed_label)
#print(transformed_label)
```


```python
with tf.device('/GPU:2'):
    (trainX, testX, trainY, testY) = train_test_split(data, transformed_label, test_size=0.25, random_state=42)
```


```python
trainX = trainX / 255.0 
testX = testX / 255.0
```


```python
# source: https://github.com/yu4u/cutout-random-erasing

def get_random_eraser(p=0.5, s_l=0.02, s_h=0.4, r_1=0.3, r_2=1/0.3, v_l=0, v_h=255, pixel_level=False):
    def eraser(input_img):
        img_h, img_w, img_c = input_img.shape
        p_1 = np.random.rand()

        if p_1 > p:
            return input_img

        while True:
            s = np.random.uniform(s_l, s_h) * img_h * img_w
            r = np.random.uniform(r_1, r_2)
            w = int(np.sqrt(s / r))
            h = int(np.sqrt(s * r))
            left = np.random.randint(0, img_w)
            top = np.random.randint(0, img_h)

            if left + w <= img_w and top + h <= img_h:
                break

        if pixel_level:
            c = np.random.uniform(v_l, v_h, (h, w, img_c))
        else:
            c = np.random.uniform(v_l, v_h)

        input_img[top:top + h, left:left + w, :] = c

        return input_img

    return eraser
```


```python
datagen = tf.keras.preprocessing.image.ImageDataGenerator(preprocessing_function=get_random_eraser(v_l=0, v_h=1))
```


```python
incept_res = InceptionResNetV2(weights='imagenet', include_top=False)
```


```python
inputs = tf.keras.Input(shape=(150, 150, 3))
x = incept_res(inputs)
x = GlobalAveragePooling2D()(x)
x = Dropout(0.2)(x)
```


```python
predictions = Dense(150,kernel_regularizer=regularizers.l2(0.005), activation='softmax')(x)
```


```python
model = Model(inputs=inputs, outputs=predictions)
```


```python
model.compile(optimizer=Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
```


```python
model.summary()
```

    Model: "functional_5"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    input_6 (InputLayer)         [(None, 150, 150, 3)]     0         
    _________________________________________________________________
    inception_resnet_v2 (Functio (None, None, None, 1536)  54336736  
    _________________________________________________________________
    global_average_pooling2d_2 ( (None, 1536)              0         
    _________________________________________________________________
    dropout_2 (Dropout)          (None, 1536)              0         
    _________________________________________________________________
    dense_4 (Dense)              (None, 150)               230550    
    =================================================================
    Total params: 54,567,286
    Trainable params: 54,506,742
    Non-trainable params: 60,544
    _________________________________________________________________



```python
early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=1)
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath='chkpoint-trial04-cutout',
    save_weights_only=True,
    monitor='val_acc',
    mode='max',
    save_best_only=True)

```


```python
with tf.device('/GPU:2'):
    history = model.fit(datagen.flow(trainX, trainY, batch_size=32), epochs=50, callbacks=[early_stopping_callback,model_checkpoint_callback], validation_data=(testX, testY))
```

    Epoch 1/50
    3523/3523 [==============================] - ETA: 0s - loss: 2.8135 - accuracy: 0.5056WARNING:tensorflow:Can save best model only with val_acc available, skipping.
    3523/3523 [==============================] - 533s 151ms/step - loss: 2.8135 - accuracy: 0.5056 - val_loss: 1.7278 - val_accuracy: 0.6683
    Epoch 2/50
    3523/3523 [==============================] - ETA: 0s - loss: 1.5222 - accuracy: 0.6915WARNING:tensorflow:Can save best model only with val_acc available, skipping.
    3523/3523 [==============================] - 517s 147ms/step - loss: 1.5222 - accuracy: 0.6915 - val_loss: 1.6863 - val_accuracy: 0.6968
    Epoch 3/50
    3523/3523 [==============================] - ETA: 0s - loss: 1.1357 - accuracy: 0.7591WARNING:tensorflow:Can save best model only with val_acc available, skipping.
    3523/3523 [==============================] - 518s 147ms/step - loss: 1.1357 - accuracy: 0.7591 - val_loss: 1.1786 - val_accuracy: 0.7378
    Epoch 4/50
    3523/3523 [==============================] - ETA: 0s - loss: 0.9260 - accuracy: 0.8024WARNING:tensorflow:Can save best model only with val_acc available, skipping.
    3523/3523 [==============================] - 516s 147ms/step - loss: 0.9260 - accuracy: 0.8024 - val_loss: 1.5460 - val_accuracy: 0.6513
    Epoch 5/50
    3523/3523 [==============================] - ETA: 0s - loss: 0.7836 - accuracy: 0.8339WARNING:tensorflow:Can save best model only with val_acc available, skipping.
    3523/3523 [==============================] - 519s 147ms/step - loss: 0.7836 - accuracy: 0.8339 - val_loss: 1.1877 - val_accuracy: 0.7406
    Epoch 6/50
    3523/3523 [==============================] - ETA: 0s - loss: 0.6937 - accuracy: 0.8580WARNING:tensorflow:Can save best model only with val_acc available, skipping.
    3523/3523 [==============================] - 516s 147ms/step - loss: 0.6937 - accuracy: 0.8580 - val_loss: 1.1886 - val_accuracy: 0.7457
    Epoch 7/50
    3523/3523 [==============================] - ETA: 0s - loss: 0.6221 - accuracy: 0.8736WARNING:tensorflow:Can save best model only with val_acc available, skipping.
    3523/3523 [==============================] - 518s 147ms/step - loss: 0.6221 - accuracy: 0.8736 - val_loss: 1.1890 - val_accuracy: 0.7484
    Epoch 8/50
    3523/3523 [==============================] - ETA: 0s - loss: 0.5675 - accuracy: 0.8868WARNING:tensorflow:Can save best model only with val_acc available, skipping.
    3523/3523 [==============================] - 517s 147ms/step - loss: 0.5675 - accuracy: 0.8868 - val_loss: 1.1681 - val_accuracy: 0.7555
    Epoch 9/50
    3523/3523 [==============================] - ETA: 0s - loss: 0.5281 - accuracy: 0.8957WARNING:tensorflow:Can save best model only with val_acc available, skipping.
    3523/3523 [==============================] - 517s 147ms/step - loss: 0.5281 - accuracy: 0.8957 - val_loss: 1.2206 - val_accuracy: 0.7492
    Epoch 10/50
    3523/3523 [==============================] - ETA: 0s - loss: 0.4972 - accuracy: 0.9030WARNING:tensorflow:Can save best model only with val_acc available, skipping.
    3523/3523 [==============================] - 518s 147ms/step - loss: 0.4972 - accuracy: 0.9030 - val_loss: 1.2040 - val_accuracy: 0.7548
    Epoch 11/50
    3523/3523 [==============================] - ETA: 0s - loss: 0.4672 - accuracy: 0.9101WARNING:tensorflow:Can save best model only with val_acc available, skipping.
    3523/3523 [==============================] - 517s 147ms/step - loss: 0.4672 - accuracy: 0.9101 - val_loss: 1.1618 - val_accuracy: 0.7628
    Epoch 12/50
    3523/3523 [==============================] - ETA: 0s - loss: 0.4459 - accuracy: 0.9150WARNING:tensorflow:Can save best model only with val_acc available, skipping.
    3523/3523 [==============================] - 517s 147ms/step - loss: 0.4459 - accuracy: 0.9150 - val_loss: 1.2195 - val_accuracy: 0.7552
    Epoch 13/50
    3523/3523 [==============================] - ETA: 0s - loss: 0.4251 - accuracy: 0.9194WARNING:tensorflow:Can save best model only with val_acc available, skipping.
    3523/3523 [==============================] - 518s 147ms/step - loss: 0.4251 - accuracy: 0.9194 - val_loss: 1.2375 - val_accuracy: 0.7569
    Epoch 14/50
    3523/3523 [==============================] - ETA: 0s - loss: 0.4157 - accuracy: 0.9215WARNING:tensorflow:Can save best model only with val_acc available, skipping.
    3523/3523 [==============================] - 518s 147ms/step - loss: 0.4157 - accuracy: 0.9215 - val_loss: 1.2135 - val_accuracy: 0.7588
    Epoch 15/50
    3523/3523 [==============================] - ETA: 0s - loss: 0.3923 - accuracy: 0.9274WARNING:tensorflow:Can save best model only with val_acc available, skipping.
    3523/3523 [==============================] - 518s 147ms/step - loss: 0.3923 - accuracy: 0.9274 - val_loss: 1.2507 - val_accuracy: 0.7551
    Epoch 16/50
    3523/3523 [==============================] - ETA: 0s - loss: 0.3818 - accuracy: 0.9294WARNING:tensorflow:Can save best model only with val_acc available, skipping.
    3523/3523 [==============================] - 517s 147ms/step - loss: 0.3818 - accuracy: 0.9294 - val_loss: 1.2964 - val_accuracy: 0.7527
    Epoch 17/50
    3523/3523 [==============================] - ETA: 0s - loss: 0.3735 - accuracy: 0.9312WARNING:tensorflow:Can save best model only with val_acc available, skipping.
    3523/3523 [==============================] - 514s 146ms/step - loss: 0.3735 - accuracy: 0.9312 - val_loss: 1.2748 - val_accuracy: 0.7541
    Epoch 18/50
    3523/3523 [==============================] - ETA: 0s - loss: 0.3641 - accuracy: 0.9332WARNING:tensorflow:Can save best model only with val_acc available, skipping.
    3523/3523 [==============================] - 517s 147ms/step - loss: 0.3641 - accuracy: 0.9332 - val_loss: 1.2075 - val_accuracy: 0.7654
    Epoch 19/50
    3523/3523 [==============================] - ETA: 0s - loss: 0.3604 - accuracy: 0.9338WARNING:tensorflow:Can save best model only with val_acc available, skipping.
    3523/3523 [==============================] - 517s 147ms/step - loss: 0.3604 - accuracy: 0.9338 - val_loss: 1.2774 - val_accuracy: 0.7607
    Epoch 20/50
    3523/3523 [==============================] - ETA: 0s - loss: 0.3445 - accuracy: 0.9371WARNING:tensorflow:Can save best model only with val_acc available, skipping.
    3523/3523 [==============================] - 519s 147ms/step - loss: 0.3445 - accuracy: 0.9371 - val_loss: 1.2628 - val_accuracy: 0.7601
    Epoch 21/50
    3523/3523 [==============================] - ETA: 0s - loss: 0.3369 - accuracy: 0.9387WARNING:tensorflow:Can save best model only with val_acc available, skipping.
    3523/3523 [==============================] - 515s 146ms/step - loss: 0.3369 - accuracy: 0.9387 - val_loss: 1.2311 - val_accuracy: 0.7642
    Epoch 22/50
    3523/3523 [==============================] - ETA: 0s - loss: 0.3332 - accuracy: 0.9395WARNING:tensorflow:Can save best model only with val_acc available, skipping.
    3523/3523 [==============================] - 518s 147ms/step - loss: 0.3332 - accuracy: 0.9395 - val_loss: 1.2764 - val_accuracy: 0.7572
    Epoch 23/50
    3523/3523 [==============================] - ETA: 0s - loss: 0.3260 - accuracy: 0.9417WARNING:tensorflow:Can save best model only with val_acc available, skipping.
    3523/3523 [==============================] - 519s 147ms/step - loss: 0.3260 - accuracy: 0.9417 - val_loss: 1.2509 - val_accuracy: 0.7624
    Epoch 24/50
    3523/3523 [==============================] - ETA: 0s - loss: 0.3123 - accuracy: 0.9449WARNING:tensorflow:Can save best model only with val_acc available, skipping.
    3523/3523 [==============================] - 515s 146ms/step - loss: 0.3123 - accuracy: 0.9449 - val_loss: 1.3004 - val_accuracy: 0.7604
    Epoch 25/50
    3523/3523 [==============================] - ETA: 0s - loss: 0.3069 - accuracy: 0.9459WARNING:tensorflow:Can save best model only with val_acc available, skipping.
    3523/3523 [==============================] - 517s 147ms/step - loss: 0.3069 - accuracy: 0.9459 - val_loss: 1.3211 - val_accuracy: 0.7586
    Epoch 26/50
    3523/3523 [==============================] - ETA: 0s - loss: 0.2986 - accuracy: 0.9482WARNING:tensorflow:Can save best model only with val_acc available, skipping.
    3523/3523 [==============================] - 518s 147ms/step - loss: 0.2986 - accuracy: 0.9482 - val_loss: 1.3037 - val_accuracy: 0.7608
    Epoch 27/50
    3523/3523 [==============================] - ETA: 0s - loss: 0.2971 - accuracy: 0.9478WARNING:tensorflow:Can save best model only with val_acc available, skipping.
    3523/3523 [==============================] - 519s 147ms/step - loss: 0.2971 - accuracy: 0.9478 - val_loss: 1.2757 - val_accuracy: 0.7633
    Epoch 28/50
    3523/3523 [==============================] - ETA: 0s - loss: 0.2919 - accuracy: 0.9490WARNING:tensorflow:Can save best model only with val_acc available, skipping.
    3523/3523 [==============================] - 519s 147ms/step - loss: 0.2919 - accuracy: 0.9490 - val_loss: 1.2524 - val_accuracy: 0.7650
    Epoch 29/50
    3523/3523 [==============================] - ETA: 0s - loss: 0.2842 - accuracy: 0.9510WARNING:tensorflow:Can save best model only with val_acc available, skipping.
    3523/3523 [==============================] - 517s 147ms/step - loss: 0.2842 - accuracy: 0.9510 - val_loss: 1.2655 - val_accuracy: 0.7673
    Epoch 30/50
    3523/3523 [==============================] - ETA: 0s - loss: 0.2840 - accuracy: 0.9508WARNING:tensorflow:Can save best model only with val_acc available, skipping.
    3523/3523 [==============================] - 518s 147ms/step - loss: 0.2840 - accuracy: 0.9508 - val_loss: 1.3067 - val_accuracy: 0.7662
    Epoch 31/50
    3523/3523 [==============================] - ETA: 0s - loss: 0.2813 - accuracy: 0.9511WARNING:tensorflow:Can save best model only with val_acc available, skipping.
    3523/3523 [==============================] - 519s 147ms/step - loss: 0.2813 - accuracy: 0.9511 - val_loss: 1.2415 - val_accuracy: 0.7681
    Epoch 32/50
    3523/3523 [==============================] - ETA: 0s - loss: 0.2755 - accuracy: 0.9529WARNING:tensorflow:Can save best model only with val_acc available, skipping.
    3523/3523 [==============================] - 512s 145ms/step - loss: 0.2755 - accuracy: 0.9529 - val_loss: 1.2498 - val_accuracy: 0.7695
    Epoch 33/50
    3523/3523 [==============================] - ETA: 0s - loss: 0.2706 - accuracy: 0.9536WARNING:tensorflow:Can save best model only with val_acc available, skipping.
    3523/3523 [==============================] - 517s 147ms/step - loss: 0.2706 - accuracy: 0.9536 - val_loss: 1.3054 - val_accuracy: 0.7640
    Epoch 34/50
    3523/3523 [==============================] - ETA: 0s - loss: 0.2670 - accuracy: 0.9543WARNING:tensorflow:Can save best model only with val_acc available, skipping.
    3523/3523 [==============================] - 517s 147ms/step - loss: 0.2670 - accuracy: 0.9543 - val_loss: 1.3106 - val_accuracy: 0.7593
    Epoch 35/50
    3523/3523 [==============================] - ETA: 0s - loss: 0.2630 - accuracy: 0.9552WARNING:tensorflow:Can save best model only with val_acc available, skipping.
    3523/3523 [==============================] - 518s 147ms/step - loss: 0.2630 - accuracy: 0.9552 - val_loss: 1.3393 - val_accuracy: 0.7587
    Epoch 36/50
    3523/3523 [==============================] - ETA: 0s - loss: 0.2631 - accuracy: 0.9551WARNING:tensorflow:Can save best model only with val_acc available, skipping.
    3523/3523 [==============================] - 518s 147ms/step - loss: 0.2631 - accuracy: 0.9551 - val_loss: 1.3105 - val_accuracy: 0.7592
    Epoch 37/50
    3523/3523 [==============================] - ETA: 0s - loss: 0.2579 - accuracy: 0.9569WARNING:tensorflow:Can save best model only with val_acc available, skipping.
    3523/3523 [==============================] - 517s 147ms/step - loss: 0.2579 - accuracy: 0.9569 - val_loss: 1.2353 - val_accuracy: 0.7744
    Epoch 38/50
    3523/3523 [==============================] - ETA: 0s - loss: 0.2511 - accuracy: 0.9577WARNING:tensorflow:Can save best model only with val_acc available, skipping.
    3523/3523 [==============================] - 518s 147ms/step - loss: 0.2511 - accuracy: 0.9577 - val_loss: 1.3210 - val_accuracy: 0.7633
    Epoch 39/50
    3523/3523 [==============================] - ETA: 0s - loss: 0.2524 - accuracy: 0.9576WARNING:tensorflow:Can save best model only with val_acc available, skipping.
    3523/3523 [==============================] - 518s 147ms/step - loss: 0.2524 - accuracy: 0.9576 - val_loss: 1.3136 - val_accuracy: 0.7694
    Epoch 40/50
    3523/3523 [==============================] - ETA: 0s - loss: 0.2517 - accuracy: 0.9579WARNING:tensorflow:Can save best model only with val_acc available, skipping.
    3523/3523 [==============================] - 519s 147ms/step - loss: 0.2517 - accuracy: 0.9579 - val_loss: 1.2850 - val_accuracy: 0.7713
    Epoch 41/50
    3523/3523 [==============================] - ETA: 0s - loss: 0.2465 - accuracy: 0.9592WARNING:tensorflow:Can save best model only with val_acc available, skipping.
    3523/3523 [==============================] - 517s 147ms/step - loss: 0.2465 - accuracy: 0.9592 - val_loss: 1.3448 - val_accuracy: 0.7620
    Epoch 42/50
    3523/3523 [==============================] - ETA: 0s - loss: 0.2470 - accuracy: 0.9582WARNING:tensorflow:Can save best model only with val_acc available, skipping.
    3523/3523 [==============================] - 518s 147ms/step - loss: 0.2470 - accuracy: 0.9582 - val_loss: 1.3024 - val_accuracy: 0.7636
    Epoch 43/50
    3523/3523 [==============================] - ETA: 0s - loss: 0.2363 - accuracy: 0.9614WARNING:tensorflow:Can save best model only with val_acc available, skipping.
    3523/3523 [==============================] - 512s 145ms/step - loss: 0.2363 - accuracy: 0.9614 - val_loss: 1.3247 - val_accuracy: 0.7692
    Epoch 44/50
    3523/3523 [==============================] - ETA: 0s - loss: 0.2437 - accuracy: 0.9594WARNING:tensorflow:Can save best model only with val_acc available, skipping.
    3523/3523 [==============================] - 518s 147ms/step - loss: 0.2437 - accuracy: 0.9594 - val_loss: 1.3046 - val_accuracy: 0.7664
    Epoch 45/50
    3523/3523 [==============================] - ETA: 0s - loss: 0.2396 - accuracy: 0.9601WARNING:tensorflow:Can save best model only with val_acc available, skipping.
    3523/3523 [==============================] - 518s 147ms/step - loss: 0.2396 - accuracy: 0.9601 - val_loss: 1.2439 - val_accuracy: 0.7767
    Epoch 46/50
    3523/3523 [==============================] - ETA: 0s - loss: 0.2401 - accuracy: 0.9603WARNING:tensorflow:Can save best model only with val_acc available, skipping.
    3523/3523 [==============================] - 557s 158ms/step - loss: 0.2401 - accuracy: 0.9603 - val_loss: 1.2777 - val_accuracy: 0.7704



```python
model.save('trial04')
```

    WARNING:tensorflow:From /home/eun0709/.local/share/virtualenvs/cnn-practice-00---gx0hUu/lib/python3.6/site-packages/tensorflow/python/training/tracking/tracking.py:111: Model.state_updates (from tensorflow.python.keras.engine.training) is deprecated and will be removed in a future version.
    Instructions for updating:
    This property should not be used in TensorFlow 2.0, as updates are applied automatically.
    WARNING:tensorflow:From /home/eun0709/.local/share/virtualenvs/cnn-practice-00---gx0hUu/lib/python3.6/site-packages/tensorflow/python/training/tracking/tracking.py:111: Layer.updates (from tensorflow.python.keras.engine.base_layer) is deprecated and will be removed in a future version.
    Instructions for updating:
    This property should not be used in TensorFlow 2.0, as updates are applied automatically.
    INFO:tensorflow:Assets written to: trial04/assets



```python
import matplotlib.pyplot as plt
```


```python
plt.plot(history.history['loss'], label='MAE (training data)')
plt.plot(history.history['val_loss'], label='MAE (validation data)')
plt.ylabel('MAE value')
plt.xlabel('No. epoch')
# plt.legend(loc="upper left")
plt.show()
```


![png](output_20_0.png)



```python
plt.plot(history.history['accuracy'], label='acc (training data)')
plt.plot(history.history['val_accuracy'], label='acc (validation data)')
plt.ylabel('Accuracy')
plt.xlabel('No. epoch')
# plt.legend(loc="upper left")
plt.show()
```


![png](output_21_0.png)
