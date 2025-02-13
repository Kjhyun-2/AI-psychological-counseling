from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from load_new import load_fer2013, preprocess_input
from models.cnn import mini_XCEPTION
from sklearn.model_selection import train_test_split
import os

def train_model():
    # parameters
    batch_size = 32
    num_epochs = 10000
    input_shape = (48, 48, 1)
    validation_split = .2
    verbose = 1
    num_classes = 7
    patience = 50
    base_path = 'models/'
    best_model_path = os.path.join(base_path, 'best_model.hdf5')  # 고정된 파일 이름

    # data generator
    data_generator = ImageDataGenerator(
        featurewise_center=False,
        featurewise_std_normalization=False,
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=.1,
        horizontal_flip=True
    )

    # model parameters/compilation
    model = mini_XCEPTION(input_shape, num_classes)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    # callbacks
    log_file_path = base_path + '_emotion_training.log'
    csv_logger = CSVLogger(log_file_path, append=False)
    early_stop = EarlyStopping('val_loss', patience=patience)
    reduce_lr = ReduceLROnPlateau('val_loss', factor=0.1, patience=int(patience / 4), verbose=1)

    model_checkpoint = ModelCheckpoint(
        filepath=best_model_path,
        monitor='val_accuracy',
        verbose=1,
        save_best_only=True,
        mode='max'
    )
    callbacks = [model_checkpoint, csv_logger, early_stop, reduce_lr]

    # loading dataset
    faces, emotions = load_fer2013()

    faces = preprocess_input(faces)
    num_samples, num_classes = emotions.shape
    xtrain, xtest, ytrain, ytest = train_test_split(faces, emotions, test_size=0.2, shuffle=True)
    model.fit(data_generator.flow(xtrain, ytrain, batch_size),
              steps_per_epoch=len(xtrain) / batch_size,
              epochs=num_epochs,
              verbose=1,
              callbacks=callbacks,
              validation_data=(xtest, ytest))

    # Save the final best model with val_accuracy in the filename
    val_accuracy = model.evaluate(xtest, ytest, verbose=0)[1]
    final_model_path = os.path.join(base_path, f'best_model-{val_accuracy:.4f}.hdf5')
    model.save(final_model_path)
    print(f'Final best model saved as: {final_model_path}')

if __name__ == '__main__':
    train_model()
