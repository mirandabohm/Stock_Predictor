import os
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from math import ceil
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler 

# =============================================================================
# Dataset V1.0: Scalar Closing Values as Labels 
# =============================================================================

# Load data 
file = r'DJIA_all_time' + '.csv'

dirname = os.path.dirname(__file__)
filename = os.path.join((dirname), 'data/' +file)
df = pd.read_csv(filename) # 2D (examples, 7 features)

#Extract close data and dates
close_data = df['Close'] # 1D (examples, )
dates = df['Date'] # 1D (examples, )
adj_dates = mdates.datestr2num(dates)

# Visualize whole dataset
plt.plot_date(adj_dates, close_data, '-')    
plt.title('DJIA Close vs. Date, 1985-2020')
plt.xlabel('Date', fontsize = 18)
plt.ylabel('Close value', fontsize = 16)
plt.show()

# Check for, and handle, missing values 
if close_data.isna().sum():
    print('Alert! Missing data')
    # do something to handle empty data rows 
    
# Define the training set
percent_training: float = 0.80
num_training_samples = ceil(percent_training*len(df)) # int
training_set = df.iloc[:num_training_samples, 5:6].values # (7135, 1)

# Scale training data
scaler = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = scaler.fit_transform(training_set) #2D (num_training_samples, 1)

# Some parameters
sequence_length: int = 90

labels_are_sequences = False

# =============================================================================
# Dataset V1.0: Scalar Closing Values as Labels 
# =============================================================================

# Build 3D training set (examples, sequence_length, 1) 
x_train = np.array([training_set_scaled[i - sequence_length:i, 0] for i in range(sequence_length, len(training_set_scaled))]) 
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# Build test sets
num_testing_samples: int = len(df) - x_train.shape[0] # Scalar 
testing_set = df.iloc[-num_testing_samples:, 5:6].values # 2D (examples, 1)
testing_set_scaled = scaler.fit_transform(testing_set) # 2D ndarray (examples, 1)

x_test = np.array([testing_set_scaled[i - sequence_length:i, 0] for i in range(sequence_length, len(testing_set_scaled))])
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1)) #3D (examples-sequence_length, sequence_length, 1)

if not labels_are_sequences:
    # Build 1D training labels (examples, )
    y_train = np.array([training_set_scaled[i, 0] for i in range(sequence_length, len(training_set_scaled))])
    y_test = np.array([testing_set_scaled[i, 0] for i in range(sequence_length, len(testing_set_scaled))]) # (examples-sequence_length, 1)
    y_test = np.reshape(y_test, (y_test.shape[0])) #1D (examples, )

# =============================================================================
# Dataset V2.0: Vectors of Closing Values as Labels 
# =============================================================================

if labels_are_sequences:
    y_train, y_test = [], []
    
    for i in range(sequence_length, len(training_set_scaled)):
        y_train.append(training_set_scaled[i + sequence_length: sequence_length*2 + i, 0])
        y_test.append(testing_set_scaled[i + sequence_length: sequence_length*2 + i, 0])
    
    y_train = np.array(list(y_item for y_item in y_train))
    y_test = np.array(list(y_item for y_item in y_test))
    # y_train= [training_set_scaled[i + sequence_length: sequence_length*2 + i, 0] for i in range(sequence_length, len(training_set_scaled))]
    # y_test = [testing_set_scaled[i + sequence_length: sequence_length*2 + i, 0] for i in range(sequence_length, len(testing_set_scaled))] # (examples-sequence_length, 1)
    # y_test = np.reshape(y_test, (y_test.shape[0])) #1D (examples, )


# =============================================================================
# Define and build model 
# =============================================================================

epochs: int = 150
batch_size: int = 32

LSTM_1 = LSTM(
    units = 5, 
    input_shape = (x_train.shape[1], 1),
    return_sequences = False,
    )

LSTM_2 = LSTM(
    units = 10
    )

model = Sequential()
model.add(LSTM_1) # Output shape: (batch_size, sequence_length, units)
model.add(Dropout(0.4))
# model.add(LSTM_2) # Output shape: 
# model.add(Dropout(0.2))

model.add(Dense(1))
model.compile(loss = 'mean_squared_error', 
             optimizer = 'adam', 
             )

early_stopping = EarlyStopping(monitor='val_loss', 
                               mode='min', 
                               verbose = 1, 
                               patience = 9,
                               restore_best_weights = False
                               )

history = model.fit(x_train,
          y_train,
          epochs = epochs, 
          batch_size = batch_size,
          verbose = 2, 
          validation_split = 0.20,
          # validation_data = (x_test, y_test),
          callbacks = [early_stopping],
          )

# Evaluate performance 
model.summary()

model.save("model.h5")
print("Model saved.")

loss = model.evaluate(x_test, y_test, batch_size = batch_size)

# early_stopping.stopped_epoch returns 0 if training didn't stop early. 
print('Training stopped after',early_stopping.stopped_epoch,'epochs.')

# =============================================================================
# plt.plot(history.history['accuracy'])
# plt.title('Model Accuracy vs. Epoch')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['Train', 'Test'], loc='upper left')
# plt.show()
# =============================================================================

plt.plot(history.history['loss'])
plt.title('Model Loss vs. Epoch')
plt.ylabel('loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

prediction = model.predict(x_test)
prediction = scaler.inverse_transform(prediction)

y_test2 = np.reshape(y_test, (y_test.shape[0], 1))
y_test = scaler.inverse_transform(y_test2)

test_dates = adj_dates[-x_test.shape[0]:]

# Visualizing the results
plt.plot_date(test_dates, y_test, '-', linewidth = 2, color = 'red', label = 'Real DJIA Close')
plt.plot(test_dates, prediction, color = 'blue', label = 'Predicted Close')
plt.title('Close Prediction')
plt.xlabel('Time')
plt.ylabel('DJIA Close')
plt.legend()
plt.show()

# Generate future data 
time_horizon = sequence_length
# future_lookback = adj_dates[-time_horizon:]

last_n = x_test[-time_horizon:,:,:] # Find last n number of days
future_prediction = model.predict(last_n)
future_prediction2 = np.reshape(future_prediction, (future_prediction.shape[0], 1))
future_prediction3 = scaler.inverse_transform(future_prediction2)
future_prediction3 = np.reshape(future_prediction3, (future_prediction3.shape[0]))
 
full_dataset_numpy = np.array(close_data)
all_data = np.append(full_dataset_numpy, future_prediction3)
plt.plot(all_data, color = 'blue', label = 'All data')
plt.title('All data including predictions')
plt.xlabel('Time')
plt.ylabel('DJIA Close')
plt.legend()
plt.show()

# Generate dates for future predictions
# Begin at the last date in the dataset, then add 'time_horizon' many new dates
last_date = dates.iloc[-1] # String
timestamp_list = pd.date_range(last_date, periods = time_horizon).tolist() #List of timestamps

# Convert list of timestamps to list of strings 
datestring_list = [i.strftime("%Y-%m-%d") for i in timestamp_list] #List of strings

# Clip first value, which is already included in the dataset
datestring2 = mdates.datestr2num(datestring_list)

plt.plot_date(datestring2, future_prediction3, '-', color = 'blue', label = 'Predicted Close')
plt.title('DJIA Close Prediction')
plt.xlabel('Date')
plt.ylabel('Predicted Close')
plt.xticks(rotation = 45)
plt.legend()
plt.show()













