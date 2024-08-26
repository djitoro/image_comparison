# Small projects and algorithms for working with images
The code was written for use in future works.
___
## List of topics: 
1. Ideas for the future
2. MSE - pixel-by-pixel comparison
3. Autoencoders
4. Retrained models 

___
## Description: 
The pros and cons of each method, as well as a detailed implementation, can be found in the file at the very bottom of the page

### 1.Ideas for the future
1. To study methods for solving segmentation problems
2. Read about: the Kalman filter, and the Hungarian algorithm as a target distribution
3. Use algorithms like: YOLO, SSD

### 2. MSE
```python
# Compute Mean Squared Error (MSE)
mse = ((image1 - image2) ** 2).mean()
```

### 3. Autoencoders
Important stages:
1. Data preprocessing
```python
def image2array(filelist):
    image_array = []
    for image in filelist[:200]:
        img = io.imread(image)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224,224))
        image_array.append(img)
    image_array = np.array(image_array)
    image_array = image_array.reshape(image_array.shape[0], 224, 224, 3)
    image_array = image_array.astype('float32')
    image_array /= 255
    return np.array(image_array)
```
2. Architecture
```python
def build_deep_autoencoder(img_shape, code_size): # построение архитектуры
    H,W,C = img_shape
    # encoder - сжимает исходное изображение в кодированное представление
    encoder = tf.keras.models.Sequential() # инициализация модели
    encoder.add(L.InputLayer(img_shape)) # добавление входного слоя, размер равен размеру изображения
    encoder.add(L.Conv2D(filters=32, kernel_size=(3, 3), activation='elu', padding='same'))
    encoder.add(L.MaxPooling2D(pool_size=(2, 2)))
    encoder.add(L.Conv2D(filters=64, kernel_size=(3, 3), activation='elu', padding='same'))
    encoder.add(L.MaxPooling2D(pool_size=(2, 2)))
    encoder.add(L.Conv2D(filters=128, kernel_size=(3, 3), activation='elu', padding='same'))
    encoder.add(L.MaxPooling2D(pool_size=(2, 2)))
    encoder.add(L.Conv2D(filters=256, kernel_size=(3, 3), activation='elu', padding='same'))
    encoder.add(L.MaxPooling2D(pool_size=(2, 2)))
    encoder.add(L.Flatten())
    encoder.add(L.Dense(code_size))

    # decoder - востановление изображения из кодированного представления
    decoder = tf.keras.models.Sequential()
    decoder.add(L.InputLayer((code_size,)))
    decoder.add(L.Dense(14*14*256))
    decoder.add(L.Reshape((14, 14, 256)))
    decoder.add(L.Conv2DTranspose(filters=128, kernel_size=(3, 3), strides=2, activation='elu', padding='same'))
    decoder.add(L.Conv2DTranspose(filters=64, kernel_size=(3, 3), strides=2, activation='elu', padding='same'))
    decoder.add(L.Conv2DTranspose(filters=32, kernel_size=(3, 3), strides=2, activation='elu', padding='same'))
    decoder.add(L.Conv2DTranspose(filters=3, kernel_size=(3, 3), strides=2, activation=None, padding='same'))

    return encoder, decoder
```
3. Vector comparison and output of results
```python
def get_similar(image, n_neighbors=5):
    assert image.ndim==3,"image must be [batch,height,width,3]"
    code = encoder.predict(image[None])
    (distances,),(idx,) = nei_clf.kneighbors(code,n_neighbors=n_neighbors)
    return distances,images[idx]
def show_similar(image):
    distances,neighbors = get_similar(image,n_neighbors=3)
    plt.figure(figsize=[8,7])
    plt.subplot(1,4,1)
    plt.imshow(image)
    plt.title("Original image")
```

### 4. Further training of the model
Important stages:
1. Let's take a ready-made model
```python
# возьмём модель VGG16 - свёрточную сеть с 13 слоями
model = keras.applications.vgg16.VGG16(weights='imagenet', include_top=True)
model.summary()
```
2. Let's remove the last layer responsible for classification
```python
feat_extractor = Model(inputs=model.input, outputs=model.get_layer("fc2").output)
feat_extractor.summary()
```
3. Applying the model to your data 
```python
# применем модель на наших данных 
import time
tic = time.perf_counter()
features = []
for i, image_path in enumerate(filelist[:200]):
    if i % 500 == 0:
        toc = time.perf_counter()
        elap = toc-tic;
        print("analyzing image %d / %d. Time: %4.4f seconds." % (i, len(images),elap))
        tic = time.perf_counter()
    img, x = load_image(path);
    feat = feat_extractor.predict(x)[0]
    features.append(feat)
print('finished extracting features for %d images' % len(images))

# понизам размерность данных при помощи PCA 
from sklearn.decomposition import PCA
features = np.array(features)
pca = PCA(n_components=100)
pca.fit(features)

pca_features = pca.transform(features)
```


[Decision](https://colab.research.google.com/drive/1Y1HfrvvNE6y5QbKBH6Cx2WUyXkEGkFxB?usp=sharing)
