# Import Libraries
from __future__ import print_function
import os
import keras
from keras.models import Model, Sequential
from keras.layers import Input, LSTM, Dense, Embedding, Bidirectional, TimeDistributed, Dropout
from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from pickle import dump
from keras.utils import plot_model
from data_utils import Utility
import pandas as pd
import numpy as np
import os
import difflib
import nltk
batch_size = 256 # Batch size for training.
epochs = 247

"""Medical Visual Question Answering (MVQA)"""

#Fetch required inputs from the datasets
timages, tquestions, tanswers, name, timage_count, tques_count, tans_unqcount = Utility.read_dataset("Train")
vimages, vquestions, val_answers, name, vimage_count, vques_count, vans_unqcount  = Utility.read_dataset("Valid")
timages, tquestions, name, timage_count, tques_count = Utility.read_dataset("Test")

"""Improvement Of QA-Pairs"""

num_unique_labels = tans_unqcount
labels = [tquestions[i][1] for i in range(len(num_unique_labels))]
# Count occurrences of each label
label_counts = Counter(labels)
# Computation of the number of questions under each label
for label, count in label_counts.items():
    print(f"Label {label}: {count} questions")   
# Finding the hardsample threshold value
average_questions_per_label = sum(label_counts.values()) / len(label_counts)
print(f"Average number of questions per label: {average_questions_per_label:.2f}")
valid_labels = label_counts[label_counts > average_questions_per_label].index
# Filter the DataFrame to keep only questions with valid labels
filtered_df = df[df['label'].isin(valid_labels)]
print("Original DataFrame:")
print(df)
print("\nFiltered DataFrame:")
print(filtered_df)

num_unique_filteredlabels = filtered_df["answer"].nunique()
filteredquestions = filtered_df["question"]
filt_labels = [filteredquestions[i][1] for i in range(len(num_unique_filteredlabels))]
# Count occurrences of each label
filt_label_counts = Counter(filt_labels)
# Print the number of questions per label
for filt_label, filt_count in filt_label_counts.items():
    print(f"Label {filt_label}: {filt_count} questions")  
# Selecting appropriate average value
average_questions_per_label_afterreduction = sum(filt_label_counts.values()) / len(filt_label_counts)  
print(f"Average number of questions per label after reduction : {average_questions_per_label_afterreduction:.2f}")
qai_labels = filt_label_counts[filt_label_counts < average_questions_per_label_afterreduction].index

# Function to generate new questions
def generate_samples(label, num_questions, feature_columns):
    np.random.seed(42)  # For reproducibility
    new_features = {col: np.random.uniform(filtered_df[col].min(), filtered_df[col].max(), num_questions) for col in feature_columns}
    new_features['label'] = [label] * num_questions
    return pd.DataFrame(new_features)

# Augment dataset
feature_columns = df.columns[:-1]
new_questions= []
for label in qai_labels:
    num_to_generate = int(average_questions_per_label_afterreduction - filt_label_counts [label])
    new_questions.append(generate_samples(label, num_to_generate, feature_columns))
# Concatenate the new questions with the original dataset
if new_questions:
    augmented_df = pd.concat([filtered_df] + new_questions, ignore_index=True)
else:
    augmented_df = filtered_df
    
    
    
"""Improvement of Medical Images"""

labels = [timages[i][1] for i in range(len(num_unique_labels))]
# Count occurrences of each label
label_counts = Counter(labels)
# Count the number of images under each label
for label, count in label_counts.items():
    print(f"Label {label}: {count} images")
average_images_per_label = sum(label_counts.values()) / len(label_counts)
print(f"Average number of images per label: {average_images_per_label:.2f}")
# Hardsample threshold value
valid_labels = label_counts[label_counts > average_images_per_label].index
# Filter the DataFrame to keep only images with valid labels
filtered_df = df[df['label'].isin(valid_labels)]
num_unique_filteredlabels = filtered_df["answer"].nunique()
filteredimages = filtered_df["tanswers"]
filt_labels = [filteredimages[i][1] for i in range(len(num_unique_filteredlabels))]
# Count occurrences of each label
filt_label_counts = Counter(filt_labels)
# Print the number of images per label
for filt_label, filt_count in filt_label_counts.items():
    print(f"Label {filt_label}: {filt_count} images")  
# Appropriate Average Value
average_images_per_label_afterreduction = sum(filt_label_counts.values()) / len(filt_label_counts) 
print(f"Average number of images per label after reduction : {average_images_per_label_afterreduction:.2f}")
mii_labels = filt_label_counts[filt_label_counts < average_images_per_label_afterreduction].index
 

# Function to generate new images - Mixup and label smoothing
def mixup_data(x, y, alpha=1.0):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    batch_size = x.size(0)
    index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# x, y are batches of images and labels
x, y = filteredimages, mii_labels
y = torch.nn.functional.one_hot(y, filt_label_counts=filt_label_counts).float()  # Convert labels to one-hot
alpha = 0.4  # Adjust Mixup parameter as needed
# Apply Mixup
mixed_x, y_a, y_b, lam = mixup_data(x, y, alpha)
# Pass through the model
outputs = model(mixed_x)
# Compute loss
criterion = torch.nn.CrossEntropyLoss()
loss = mixup_criterion(criterion, outputs, y_a, y_b, lam)
smoothing = 0.1  # Smoothing factor

# Convert labels to one-hot encoding with label smoothing
def smooth_labels(mixed_x, lab_counts, smoothing):
    one_hot = np.eye(lab_counts)[mixed_x]
    smoothed = (1 - smoothing) * one_hot + smoothing / lab_counts
    return smoothed
lab_counts = Counter(y_a)
df['smoothed_label'] = list(smooth_labels(df['y_a'], lab_counts, smoothing))
# Combine samples with the same features by averaging smoothed labels
reduced_df = (
    df.groupby([mixed_x])
    .agg({'smoothed_label': lambda x: np.mean(list(x), axis=0)})
    .reset_index()
)
print("Original DataFrame:")
print(df[['mixed_x', 'y_a']])
print("\nReduced DataFrame with Smoothed Labels:")
print(reduced_df)

# Based on the image_id cocatenate image and QA-pairs
improved_dataset = pd.merge(reduced_df, augmented_df, on='name', how='inner')  
train_images, train_questions, train_answers, name = improved_dataset['timages'], improved_dataset['tquestions'], improved_dataset['tanswers'], improved_dataset['name']

"""MVQA Model Creation"""

# Extract features from each images in the directory
features = dict()
def extract_features(image_list):
       # load the model
        model = VGG16()
       # re-structure the model
        model.layers.pop()
        model = Model(inputs=model.inputs, outputs=model.layers[-1].output)
       # summarize
        print(model.summary())
        c =1
       # extract features from each photo
        for filename in image_list:
              # load an image from file
                #filename = directory + '/' + name
                image = load_img(filename, target_size=(224, 224))
              # convert the image pixels to a numpy array
                image = img_to_array(image)
              # reshape data for the model
                image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
              # prepare the image for the VGG model
                image = preprocess_input(image)
              # get features
                feature = model.predict(image, verbose=0)
              # get image id
                image_id = filename.split('/')[-1].split('.')[0]
              # store feature
                features[image_id] = feature
                print(str(c), '>%s' % filename, "\t", image_id)
                c+=1
        return features
print('extract train images features')
train_features = extract_features(train_images)
dump(train_features, open('train_features.pkl', 'wb'))


print('extract valid images features')
valid_features = extract_features(val_images)
dump(valid_features, open('valid_features.pkl', 'wb'))

print('extract test images features')
test_features = extract_features(test_images)
dump(test_features, open('test_features.pkl', 'wb'))
print('Extracted Features: %d' % len(features))

from pickle import load
def load_photo_features(filename, dataset):
        # load all features
        all_features = load(open(filename, 'rb'))
        # filter features
        features = {k.split('/')[-1].split('.')[0]: all_features[k.split('/')[-1].split('.')[0]] for k in dataset}
        return features

print("load train features")
train_features = load_photo_features('train_features.pkl', train_images)
print("load val features")
valid_features = load_photo_features('valid_features.pkl', val_images)
print("load test features")
test_features = load_photo_features('test_features.pkl', test_images)

lines = pd.DataFrame({'eng':train_questions, 'fr':train_answers})
eng = lines.eng.tolist() + val_questions.tolist()
fr = lines.fr.tolist() + val_questions.tolist()
eng = lines.eng.tolist() + test_questions.tolist()
fr = lines.fr.tolist() + test_questions.tolist()
lines = pd.DataFrame({'eng':eng, 'fr':fr})
lines.fr = lines.fr.apply(lambda x : 'START_ '+ x + ' _END')
lines1 = pd.DataFrame({'eng':eng, 'fr':fr})
lines1.fr = lines1.fr.apply(lambda x : 'START_ '+ x + ' _END')

import pdb; pdb.set_trace()
all_eng_words=set()
all_eng_words1=set()

for eng in lines.eng:
    for word in eng.split():
        if word not in all_eng_words:
            all_eng_words.add(word)

for eng in val_questions:
    for word in eng.split():
        if word not in all_eng_words:
            all_eng_words.add(word)

for eng in lines1.eng:
    for word1 in eng.split():
        if word1 not in all_eng_words1:
            all_eng_words1.add(word1)

for eng in test_questions:
    for word1 in eng.split():
        if word1 not in all_eng_words1:
            all_eng_words1.add(word1)



all_french_words=set()
for fr in lines.fr:
    for word in fr.split():
        if word not in all_french_words:
            all_french_words.add(word)

for eng in val_answers:
    for word in eng.split():
        if word not in all_eng_words:
            all_french_words.add(word)
# Answers
lenght_list=[]
for l in lines.fr:
    lenght_list.append(len(l.split(' ')))

# Questions
lenght_list=[]
for l in lines.eng:
    lenght_list.append(len(l.split(' ')))

input_words = sorted(list(all_eng_words))
target_words = sorted(list(all_french_words))
num_encoder_tokens = len(all_eng_words)
num_decoder_tokens = len(all_french_words)

input_token_index = dict(
    [(word, i) for i, word in enumerate(input_words)])
target_token_index = dict(
    [(word, i) for i, word in enumerate(target_words)])

encoder_input_data = np.zeros(
    (len(lines.eng), 11),
    dtype='float32')
decoder_input_data = np.zeros(
    (len(lines.fr), 21),
    dtype='float32')
decoder_target_data = np.zeros(
    (len(lines.fr), 21, num_decoder_tokens),
    dtype='float32')


for i, (input_text, target_text) in enumerate(zip(lines.eng , lines.fr)):
    for t, word in enumerate(input_text.split()):
        encoder_input_data[i, t] = input_token_index[word]
    for t, word in enumerate(target_text.split()):
        # decoder_target_data is ahead of decoder_input_data by one timestep
        decoder_input_data[i, t] = target_token_index[word]
        if t > 0:
            # decoder_target_data will be ahead by one timestep
            # and will not include the start character.
            decoder_target_data[i, t - 1, target_token_index[word]] = 1.


all_french_words1=set()
for fr in lines1.fr:
    for word1 in fr.split():
        if word1 not in all_french_words1:
            all_french_words1.add(word1)

# Answers
lenght_list1=[]
for l1 in lines1.fr:
    lenght_list1.append(len(l1.split(' ')))

# Questions
lenght_list1=[]
for l1 in lines1.eng:
    lenght_list1.append(len(l1.split(' ')))

input_words1 = sorted(list(all_eng_words1))
target_words1 = sorted(list(all_french_words1))
num_encoder_tokens1 = len(all_eng_words1)
num_decoder_tokens1 = len(all_french_words1)

input_token_index1 = dict(
    [(word1, i) for i, word1 in enumerate(input_words1)])
target_token_index1 = dict(
    [(word1, i) for i, word1 in enumerate(target_words1)])

encoder_input_data1 = np.zeros(
    (len(lines1.eng), 11),
    dtype='float32')
decoder_input_data1 = np.zeros(
    (len(lines1.fr), 21),
    dtype='float32')
decoder_target_data1 = np.zeros(
    (len(lines1.fr), 21, num_decoder_tokens1),
    dtype='float32')


for i, (input_text1, target_text1) in enumerate(zip(lines1.eng , lines1.fr)):
    for t1, word1 in enumerate(input_text1.split()):
        encoder_input_data1[i, t1] = input_token_index1[word1]
    for t1, word1 in enumerate(target_text1.split()):
        # decoder_target_data is ahead of decoder_input_data by one timestep
        decoder_input_data1[i, t1] = target_token_index1[word1]
        if t1 > 0:
            # decoder_target_data will be ahead by one timestep
            # and will not include the start character.
            decoder_target_data1[i, t1 - 1, target_token_index1[word1]] = 1.


print ("Load pretrained embeddings ...")
GLOVE_DIR = "glove"
embeddings_index = {}
f = open(os.path.join(GLOVE_DIR, 'glove.6B.300d.txt'))
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()
print('Found %s word vectors.' % len(embeddings_index))

emb_size = 300
hidden_nodes = 1024
embedding_size = emb_size
embedding_matrix = np.zeros((len(input_token_index) , emb_size))
for word, i in input_token_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector

embedding_matrix_2 = np.zeros((len(target_token_index) , emb_size))
for word, i in target_token_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix_2[i] = embedding_vector

# Image Model
# Feature extractor model
inputs1 = Input(shape=(4096,))
fe11 = Dense(2500, activation='relu')(inputs1)
fe2 = Dense(hidden_nodes, activation='relu')(fe11)

# Encoder model
encoder_inputs = Input(shape=(None,))
en_x=  Embedding(num_encoder_tokens, embedding_size, weights=[embedding_matrix], trainable=True)(encoder_inputs)
encoder = LSTM(hidden_nodes, return_state=True, return_sequences=True)
encoder_outputs, state_h, state_c = encoder(en_x)
encoder_h = keras.layers.concatenate([state_h, fe2])
encoder_c = keras.layers.concatenate([state_c, fe2])
encoder_states = [encoder_h, encoder_c]

# Decoder model
decoder_inputs = Input(shape=(None,))
dex=  Embedding(num_decoder_tokens, embedding_size,weights=[embedding_matrix_2], trainable=True)
final_dex= dex(decoder_inputs)
decoder_lstm = LSTM(hidden_nodes *2 , return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(final_dex,
                                     initial_state=encoder_states)
decoder_dense = Dense(num_decoder_tokens, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)
model = Model([encoder_inputs, decoder_inputs, inputs1], decoder_outputs)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])
encoder_images = [i.split('/')[-1].split('.')[0] for i in train_images]
encoder_input_images = np.array([train_features[i][0] if i in train_features else valid_features[i][0] for i in encoder_images ])
encoder_model = Model([encoder_inputs, inputs1], encoder_states)
decoder_state_input_h = Input(shape=(hidden_nodes *2,))
decoder_state_input_c = Input(shape=(hidden_nodes *2,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

final_dex2= dex(decoder_inputs)

decoder_outputs2, state_h2, state_c2 = decoder_lstm(final_dex2, initial_state=decoder_states_inputs)
decoder_states2 = [state_h2, state_c2]
decoder_outputs2 = decoder_dense(decoder_outputs2)
decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs2] + decoder_states2)
# Reverse-lookup token index to decode sequences back to
# something readable.
reverse_input_char_index = dict(
    (i, char) for char, i in input_token_index.items())
reverse_target_char_index = dict(
    (i, char) for char, i in target_token_index.items())
    
def decode_sequence(input_seq, image_features):
    # Encode the input as state vectors.
#    import pdb; pdb.set_trace()
    feature = np.array([image_features])
    states_value = encoder_model.predict([np.array([input_seq]), feature]) #encoder_model.predict(np.array([input_seq, feature]))
    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1,1))
    # Populate the first character of target sequence with the start character.
    target_seq[0, 0] = target_token_index['START_']
    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_value)

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = reverse_target_char_index[sampled_token_index]
        decoded_sentence += ' '+sampled_char

        # Exit condition: either hit max length
        # or find stop character.
        if (sampled_char == '_END' or
           len(decoded_sentence) > 52):
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1,1))
        target_seq[0, 0] = sampled_token_index

        # Update states
        states_value = [h, c]

    return decoded_sentence
    
#Evaluation 
from bleu import moses_multi_bleu
writer = open("results_comtri.txt", "a")
encoder_images_val = [i.split('/')[-1].split('.')[0] for i in val_images]
encoder_input_images_val = np.array([valid_features[i][0] for i in encoder_images_val])
val_tokens = np.array([[input_token_index[i]
                        for i in val_ans.split()] for val_ans in val_questions])
encoder_images_test = [i.split('/')[-1].split('.')[0] for i in test_images]
encoder_input_images_test = np.array([test_features[i][0] for i in encoder_images_test])

for i in range(epochs):
    print("epoch", str(i), "out of",str(epochs) )
    print(encoder_input_data.shape)
    print(decoder_input_data.shape)
    print(encoder_input_images.shape)
    print(decoder_target_data.shape)
    encoder_input_images.resize(4500,4096)
    model.fit([encoder_input_data, decoder_input_data, encoder_input_images], decoder_target_data, batch_size=512, epochs=1,validation_split=0)
    if (i+1)%10==0:
        writer.write('\n------ epoch '+str(i)+" ------\n")
        print ("Validating ...")
        actual = []
        pred = []
        actual_nltk = []
        pred_nltk = []
        output_valid = open("valid_results_comtri/valid_out_comtri_"+str(i)+".txt", "w")
        for seq_index in range(len(val_tokens)):
            input_seq = val_tokens[seq_index]
            print("Valid:",len(val_tokens))
            print('Hi7.5')
            decoded_sentence = decode_sequence(input_seq, encoder_input_images_val[seq_index])
            print('Hi8')
            ac = val_answers[seq_index].replace("START_", "")
            ac = ac.replace('_END', "").strip()
            pr = decoded_sentence.replace("START_", "")
            pr = pr.replace('_END', "").strip() 
            if seq_index<=20:
               print('-')
               print('Input sentence:', val_questions[seq_index])
               print('Actual sentence:', ac)
               print('Decoded sentence:', pr)
            output_valid.write(name[seq_index]+"|"+val_questions[seq_index]+"|"+ac+"\t"+pr+"\n")
            actual.append(ac)
            pred.append(pr)
            actual_nltk.append(ac.strip())
            pred_nltk.append(pr.strip())
        output_valid.close()
        src_new = [[i.strip().split()] for i in actual_nltk]
        trg_new = [i.strip().split() for i in pred_nltk]
        acc=difflib.SequenceMatcher(None,actual, pred).ratio()
        print("Acc:", acc)
        actual = []
        pred = []
        actual_nltk = []
        pred_nltk = []
        for seq_index in range(500):
            input_seq = encoder_input_data[seq_index]
            decoded_sentence = decode_sequence(input_seq, encoder_input_images[seq_index])

            ac = lines.fr[seq_index].replace("START_", "")
            ac = ac.replace('_END', "").strip()

            pr = decoded_sentence.replace("START_", "")
            pr = pr.replace('_END', "").strip()
            if seq_index<=20:

                print('-')
                print('Input sentence:', lines.eng[seq_index])
                print('Actual sentence:', ac)
                print('Decoded sentence:', pr)

            actual.append(ac)
            pred.append(pr)
            actual_nltk.append(ac.strip())
            pred_nltk.append(pr.strip())

        src_new = [[i.strip().split()] for i in actual_nltk]
        trg_new = [i.strip().split() for i in pred_nltk]
        print ("\nTesting...")
        output = open("test_results_comtri/out_"+str(i)+".txt", "w")
        #for seq_index in range(len(test_tokens)):
        for seq_index in range(500):
            #input_seq = test_tokens[seq_index]
            input_seq = encoder_input_data1[seq_index]
            decoded_sentence = decode_sequence(input_seq, encoder_input_images_test[seq_index])
            pr = decoded_sentence.replace("START_", "")
            pr = pr.replace('_END', "").strip()
            if seq_index<=20:
               print('-')
               print('Input sentence:', test_questions[seq_index])
               print('Decoded sentence:', pr)
            output.write(name[seq_index]+"|"+pr+"\n")
        output.close()
writer.close()


# Layerwise Relevance Propagation eXplainable Artificial Intelligence (LRP XAI)

def data_generator(data,
                   batch_size,
                   preprocessing_fn = None,
                   is_validation_data=False):
    # Get total number of samples in the data
    n = len(data)
    nb_batches = int(np.ceil(n/batch_size))

    # Get a numpy array of all the indices of the input data
    indices = np.arange(n)

    while True:
        if not is_validation_data:
            # shuffle indices for the training data
            np.random.shuffle(indices)

        for i in range(nb_batches):
            # get the next batch
            next_batch_indices = indices[i*batch_size:(i+1)*batch_size]
            nb_examples = len(next_batch_indices)

            # Define two numpy arrays for containing batch data and labels
            batch_data = np.zeros((nb_examples,
                           img_rows,
                           img_cols,
                           img_channels),
                          dtype=np.float32)
            batch_labels = np.zeros((nb_examples, nb_classes), dtype=np.float32)

            # process the next batch
            for j, idx in enumerate(next_batch_indices):
                img = cv2.imread(data.iloc[idx]["image"])
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                label = data.iloc[idx]["label"]

                if not is_validation_data:
                    img = seq.augment_image(img)

                img = cv2.resize(img, (img_rows, img_cols)).astype(np.float32)
                batch_data[j] = img
                batch_labels[j] = to_categorical(label,num_classes=nb_classes)

            if preprocessing_fn is not None:
                batch_data = preprocessing_fn(batch_data)

            yield batch_data, batch_labels
            
preprocessing_fn = vgg16.preprocess_input

#training data generator
train_data_gen = data_generator(train_df,
                                batch_size,
                                preprocessing_fn)

# validation data generator
valid_data_gen = data_generator(valid_df,
                                batch_size,
                                preprocessing_fn,
                                is_validation_data=True)


def get_base_model():
    base_model = vgg16.VGG16(input_shape=(img_rows, img_cols, img_channels),
                       weights='imagenet',
                       include_top=True)
    return base_model

# get the base model
base_model = get_base_model()

#  get the output of the second last dense layer
base_model_output = base_model.layers[-2].output

# add new layers
x = L.Dropout(0.5,name='drop2')(base_model_output)
output = L.Dense(nb_classes, activation='softmax', name='fc3')(x)

# define a new model
model = Model(base_model.input, output)

# Freeze all the base model layers
for layer in base_model.layers[:-1]:
    layer.trainable=False

# compile the model and check it
optimizer = RMSprop(0.001)
model.compile(loss='categorical_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])
model.summary()

# Use earlystopping
es = EarlyStopping(patience=3, restore_best_weights=True)

# checkpoint to save model
chkpt = ModelCheckpoint(filepath="vgg_model.h5", save_best_only=True)

# number of training and validation steps for training and validation
nb_train_steps = int(np.ceil(len(train_df)/batch_size))
nb_valid_steps = int(np.ceil(len(valid_df)/batch_size))

# number of epochs
nb_epochs=25

# train the model
history1 = model.fit_generator(train_data_gen,
                              epochs=nb_epochs,
                              steps_per_epoch=nb_train_steps,
                              validation_data=valid_data_gen,
                              validation_steps=nb_valid_steps,
                              callbacks=[es,chkpt])

# let's plot the loss and accuracy

# get the training and validation accuracy from the history object
train_acc = history1.history['accuracy']
valid_acc = history1.history['val_accuracy']

# get the loss
train_loss = history1.history['loss']
valid_loss = history1.history['val_loss']

# get the number of entries
xvalues = np.arange(len(train_acc))

# visualize
f,ax = plt.subplots(1,2, figsize=(10,5))
ax[0].plot(xvalues, train_loss)
ax[0].plot(xvalues, valid_loss)
ax[0].set_title("Loss curve")
ax[0].set_xlabel("Epoch")
ax[0].set_ylabel("loss")
ax[0].legend(['train', 'validation'])

ax[1].plot(xvalues, train_acc)
ax[1].plot(xvalues, valid_acc)
ax[1].set_title("Accuracy")
ax[1].set_xlabel("Epoch")
ax[1].set_ylabel("accuracy")
ax[1].legend(['train', 'validation'])

plt.show()


# select all the layers for which you want to visualize the outputs and store it in a list
outputs = [layer.output for layer in model.layers[1:18]]

# Define a new model that generates the above output
vis_model = Model(model.input, outputs)

# store the layer names we are interested in
layer_names = []
for layer in outputs:
    layer_names.append(layer.name.split("/")[0])


print("Layers going to be used for visualization: ")
print(layer_names)

def get_CAM(processed_image, predicted_label, layer_name='block5_conv3'):

    # This will be the model that would give us the graidents
    model_grad = Model([model.inputs],
                       [model.get_layer(layer_name).output, model.output])

    # Gradient tape gives you everything you need
    with tf.GradientTape() as tape:
        conv_output_values, predictions = model_grad(processed_image)
        loss = predictions[:, predicted_label]

    # Get the gradients wrt to the chosen layer
    grads_values = tape.gradient(loss, conv_output_values)

    # Take mean gradient per feature map
    grads_values = K.mean(grads_values, axis=(0,1,2))

    # Convert to numpy. This is done just for image operations.
    # Check for shapes and you would understand why we performed the squeeze operation here.
    conv_output_values = np.squeeze(conv_output_values.numpy())
    grads_values = grads_values.numpy()


    # Iterate over each feature map in yout conv output and multiply
    # the gradient values with the conv output values. This gives an
    # indication of "how important a feature is"
    for i in range(512): # we have 512 features in our last conv layer
        conv_output_values[:,:,i] *= grads_values[i]

    # create a heatmap
    heatmap = np.mean(conv_output_values, axis=-1)

    # remove negative values
    heatmap = np.maximum(heatmap, 0)

    # normalize
    heatmap /= heatmap.max()

    del model_grad, conv_output_values, grads_values, loss

    return heatmap

def show_random_sample(idx):
    # select the sample and read the corresponding image and label
    sample_image = cv2.imread(valid_df.iloc[idx]['image'])
    sample_image = cv2.cvtColor(sample_image, cv2.COLOR_BGR2RGB)
    sample_image = cv2.resize(sample_image, (img_rows, img_cols))
    sample_label = valid_df.iloc[idx]["label"]

    # pre-process the image
    sample_image_processed = np.expand_dims(sample_image, axis=0)
    sample_image_processed = preprocessing_fn(sample_image_processed)

    # generate activation maps from the intermediate layers using the visualization model
    activations = vis_model.predict(sample_image_processed)

    # get the label predicted by our original model
    pred_label = np.argmax(model.predict(sample_image_processed), axis=-1)[0]

    # choose any random activation map from the activation maps
    sample_activation = activations[0][0,:,:,16]

    # normalize the sample activation map
    sample_activation-=sample_activation.mean()
    sample_activation/=sample_activation.std()

    # convert pixel values between 0-255
    sample_activation *=255
    sample_activation = np.clip(sample_activation, 0, 255).astype(np.uint8)

    # get the heatmap for class activation map(CAM)
    heatmap = get_CAM(sample_image_processed, pred_label)
    heatmap = cv2.resize(heatmap, (sample_image.shape[0], sample_image.shape[1]))
    heatmap = heatmap *255
    heatmap = np.clip(heatmap, 0, 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    super_imposed_image = heatmap * 0.5 + sample_image
    super_imposed_image = np.clip(super_imposed_image, 0,255).astype(np.uint8)

    f,ax = plt.subplots(2,2, figsize=(15,8))
    ax[0,0].imshow(sample_image)
    ax[0,0].set_title(f"True label: {sample_label} \n Predicted label: {pred_label}")
    ax[0,0].axis('off')

    ax[0,1].imshow(sample_activation)
    ax[0,1].set_title("Random feature map")
    ax[0,1].axis('off')

    ax[1,0].imshow(heatmap)
    ax[1,0].set_title("Class Activation Map")
    ax[1,0].axis('off')

    ax[1,1].imshow(super_imposed_image)
    ax[1,1].set_title("Activation map superimposed")
    ax[1,1].axis('off')
    plt.show()

    return activations

# Visulaization for some examples now

_ = show_random_sample(idx=13)
for i in range(100):
  activations = show_random_sample(idx=i)

#Intermediate layers output visualization
def visualize_intermediate_activations(layer_names, activations):

    assert len(layer_names)==len(activations), "Make sure layers and activation values match"
    images_per_row=16

    for layer_name, layer_activation in zip(layer_names, activations):
        nb_features = layer_activation.shape[-1]
        size= layer_activation.shape[1]

        nb_cols = nb_features // images_per_row
        grid = np.zeros((size*nb_cols, size*images_per_row))

        for col in range(nb_cols):
            for row in range(images_per_row):
                feature_map = layer_activation[0,:,:,col*images_per_row + row]
                feature_map -= feature_map.mean()
                feature_map /= feature_map.std()
                feature_map *=255
                feature_map = np.clip(feature_map, 0, 255).astype(np.uint8)

                grid[col*size:(col+1)*size, row*size:(row+1)*size] = feature_map

        scale = 1./size
        plt.figure(figsize=(scale*grid.shape[1], scale*grid.shape[0]))
        plt.title(layer_name)
        plt.grid(False)
        plt.imshow(grid, aspect='auto', cmap='viridis')
    plt.show()

visualize_intermediate_activations(activations=activations,
                                   layer_names=layer_names)

