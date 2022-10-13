from keras.preprocessing.image import ImageDataGenerator
import ready

train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
                ready.train_dir, # 目标目录
                target_size=(150, 150), # 所有图像调整为150x150
                batch_size=20,
                class_mode='binary') # 二进制标签
validation_generator = test_datagen.flow_from_directory(
                ready.validation_dir,
                target_size=(150, 150),
                batch_size=20,
                class_mode='binary')
