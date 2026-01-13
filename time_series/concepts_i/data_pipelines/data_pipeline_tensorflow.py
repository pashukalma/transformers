
'''
tf.data

Create pipelines
  - in-memory dictionary and lists of tensors
  - out-of-memory sharded data files
preprocess data in parallel
configure data fed into a model with chaining methods
embedding with feature columns like function layers
formats from tf.data.Dataset: TextLineDataset, TFRecordDataset, FixedLengthRecordDataset
'''

''' create a dataset from in-memory tensors '''
def create_dataset(X, y, epochs, batch_size, shuffle=True):
  dataset = tf.data.Dataset.from_tensor_slices((X, y))
  dataset = dataset.repeat(epochs).batch(batch_size, drop_remainder=True)
  if shuffle:
    dataset = dataset.shuffle(10000)
  return dataset
#

''' from_tensors() and from_tensor_slices() '''

''' read and process csv files '''
def parse_row(records):
  columns = tf.decode_csv(records, record_defaults=DEFAULTS)
  features = dict(zip(FEATURE_NAMES, columns))
  labels = columns[2]
  return features, labels

def create_dataset(filename, epochs, batch_size, shuffle=True):
  dataset = tf.data.TextLineDataset(filename)
  dataset = dataset.map(parse_row)
  dataset = dataset.repeat(epochs).batch(batch_size, drop_remainder=True)
  if shuffle:
    dataset = dataset.shuffle(10000)
  return dataset

''' read and process sharded csv files '''
def parse_row(records):
  columns = tf.decode_csv(records, record_defaults=DEFAULTS)
  features = dict(zip(FEATURE_NAMES, columns))
  labels = columns[2]
  return features, labels

def create_dataset(filename, epochs, batch_size, shuffle=True):
  dataset = tf.data.Dataset.list_files(filename) \
            .flat_map(lambda filename: tf.data.TextLineDataset(filename)) \
            .map(parse_row)

  dataset = dataset.repeat(epochs).batch(batch_size, drop_remainder=True)
  if shuffle:
    dataset = dataset.shuffle(10000)
  return dataset

''' configurations
      - without prefetching (cpu/gpu)
      - with prefethcing (cpu/gpu)
      - with prefetching + multithreading load and process (cpu/cpu)
'''
dataset = tf.data.TextLineDataset(filename) \
            .skip(num_header_lines) \
            .map(add_key) \
            .map(decode_csv) \
            .map(lambda_feats, labels: preproc(feats), labels)
            .filter(is_valid) \
            .cache()

''' training inout with required disctionary of features and labels '''

''' create the input pipeline '''
def create_dataset(pattern, batch_size, mode=tf.estimator.ModeKeys.Eval):
  dataset = tf.data.experimental.make_csv_dataset(
    pattern, batch_size, csv_columns, defaults)
  dataset = dataset.map(lambda features, labels: (dict(features), labels))
  if mode == tf.estimator.ModeKeys.TRAIN:
    dataset = dataset.shuffle(1000).repeat()
  dataset = dataset.prefetch(1)
  return dataset

''' use dense features layer to input feature columns to the model '''
feature_columns = [DEFAULT_FEATURES_LIST]
feature_layer = tf.keras.layers.DenseFeatures(feature_columns)
model = tf.keras.Sequential([
  feature_layer,
  layers.Dense(128, activation='relu'),
  layers.Dense(128, activation='relu'),
  layers.Dense(1, activation='sigmoid')
])

model.fit()