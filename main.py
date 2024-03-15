
import matplotlib.pyplot as plt
import tensorflow as tf
import keras.layers as tfl
import keras.callbacks as tfc
from keras.models import Model

#Variaveis

height= 256         #Altura da imagem
width = 256         #Largura da imagem
chanels = 3         #Numero de canais da imagem
seed = 13           #Seed aleatoria
batch = 32          #Tamanho do batch
epocas = 50         #Numero de epocas


path = "C:/Users/tulio/OneDrive/Documentos/GitHub/brain_tumor_CNN/images/Training"   #Caminho para o diretorio com as imagens 


#Criando os datasets

#Dataset para o treinamento
train_ds = tf.keras.utils.image_dataset_from_directory(

    path,                                    #Caminnho para o diretorio com as imagens
    validation_split=0.2,                    #Fração das imagens para este dataset
    subset="training",                       #Subset a ser retornado
    seed=seed,                               #Seed aleatoria
    image_size=(height,width),               #Redimensionar a imagem
    batch_size= batch,                       #Tamanho do batch
    label_mode="int",                        #Sparse categorical crossentropy
    labels= "inferred",                      #Labels gerados do diretorio
    color_mode="rgb"                         #Tipo de imagem/quantidade de canais
)

#Dataset para a validação
val_ds = tf.keras.utils.image_dataset_from_directory(

    path,                                    #Caminnho para o diretorio com as imagens
    validation_split=0.2,                    #Fração das imagens para este dataset
    subset="validation",                     #Subset a ser retornado
    seed=seed,                               #Seed aleatoria
    image_size=(height,width),               #Redimensionar a imagem
    batch_size= batch,                       #Tamanho do batch
    label_mode="int",                        #Sparse categorical crossentropy
    labels= "inferred",                      #Labels gerados do diretorio
    color_mode="rgb"                         #Tipo de imagem/quantidade de canais
)

class_names = train_ds.class_names          #Nomes das classes

#Criando o modelo da rede
#ResNet-50 e um modelo de rede neural convolucional com 50 camadas
modelo_base = tf.keras.applications.resnet50.ResNet50(weights = "imagenet",
                                                 include_top = False,
                                                 input_shape = (height,width,chanels)
                                                 )

#Freezing layer
for layer in modelo_base.layers:
    layer.trainable = False

#Adicionando camadas ao output da rede
x = modelo_base.output 
x = tfl.Flatten()(x)
x = tfl.Dropout(0.2)(x)
dl1 = tfl.Dense(512, activation='relu')(x)
output = tfl.Dense(4, activation='softmax')(dl1)

model = Model(inputs=[modelo_base.input], outputs=[output])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])

#model.summary()

#Criando o checkpoint, para salvar os melhores pesos
callback = tfc.ModelCheckpoint("C:/Users/tulio/OneDrive/Documentos/GitHub/brain_tumor_CNN/best.h5",save_best_only=True)

#Criando uma condicao para que a rede pare de treinar se nao houver melhoras, ajuda a evitar overfitting
early_stopping_callback = tfc.EarlyStopping(patience=5,restore_best_weights=True)         

#Treinando o modelo
history = model.fit(train_ds,validation_data=val_ds,epochs=epocas,callbacks=[early_stopping_callback, callback])

#Carregando os melhores pesos
model.load_weights("C:/Users/tulio/OneDrive/Documentos/GitHub/brain_tumor_CNN/best.h5")

#Salvando o modelo
model.save("C:/Users/tulio/OneDrive/Documentos/GitHub/brain_tumor_CNN/model.h5")

#Salvando os pesos
model.save_weights("C:/Users/tulio/OneDrive/Documentos/GitHub/brain_tumor_CNN/ModelWeights.h5")

#Plotando o grafico de acuracia
plt.plot(history.history['accuracy'],color='red',label='training accuracy')
plt.plot(history.history['val_accuracy'],color='blue',label='validation accuracy')
plt.legend()
plt.show()

#Plotando o grafico de loss
plt.plot(history.history['loss'],color='red',label='training loss')
plt.plot(history.history['val_loss'],color='blue',label='validation loss')
plt.legend()
plt.show()