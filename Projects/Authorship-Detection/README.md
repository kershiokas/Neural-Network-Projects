
# Authorship detection

The example network tries to detect 4 different AI (Claude Sonnet 4.6, ChatGPT GPT-5.5, Gemini Flash 3.5, or Grok xAI)

The training data used to train the network is also included.

All of the files are basically txt files just with (.data, .vocab or .model)

# Architecture
3-layer feedforward network (128-64-4 outputs)

Trained using backpropagation and gradient descent.

# Training data format

This format exactly: `text #eos -AuthorName-`

Input data is automatically preprocessed:
- lowercased,
- punctuation is split into a separate token,
- numbers get turned into `<NUMBER>` tag.

If new authors appear in updated training data, the network expands automatically to accommodate them.


# To start out

All that is needed is training data.

It will grab vocabulary from the training data and index it after the training is done

trained weights will be created.

 a CSV file will be created with training rate, epoch and loss.

you have two structs, run and train.
```cpp
train(trainingDataFile, prevIndexedVocabFile, networkFile); //String Inputs

run(networkFile, prevIndexedVocabFile); //String Inputs
```

To train from scratch all you need is training data. Make sure to set `networkFile` and `prevIndexedVocabFile` to `"-"`.

While `run` needs both `networkFile` and `prevIndexedVocabFile`.





