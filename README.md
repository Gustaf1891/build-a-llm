## What I learned from the project

### Chapter 2: Working with text data
In chapter 2 the focus was on working with text data. We started by manually splitting the text on special characters and converted each word into a token (integer). The Tokenizer class was extended to also include special tokens such as <|endoftext|> and <|unk|>. After getting a general understanding of how tokens works we switched to a byte pair tokenizer, which is closer related to what LLMs use. Here we saw that tokens are no longer only whole words but more often character combinations that are frequently seen in the english language (e.g. be in below).
After that a sliding window approach was introduced, since we want the model to predict each new word using the former words in the sentence. Lastly, we looked at token embeddings and word positions, to give words relationships to each other.

### Chapter 3: The attention mechanism
In chapter 3 the attention mechanism was introduced. Firstly we calculate the dot product between all embedding vectors and then normalize the attention scores using softmax. In LLMs we have trainable query, key and value matrices where:
- Query: Represents the current words perspective, "what information is relevant to me".
- Key: Represents the infromation a word carries. The query is comapred against all keys to calculate attention scores between them (how important each word is to the query word).
- Value: Represent the actual content or meaning of a word. Weights the attention score between two words. 
Basically: keys = input @ W_key, queries = input @ W_query, values = input @ W_value -> attn_scores = queries @ keys -> attn_weights = softmax(attn_scores) (weighted before softmax to prevent vanishing gradient) -> context_vec = attn_weights @ values

We then introduced causal attention which hides future words i.e. only uses information from current words to predict the next word. This is done by introducing a mask that makes our attention scores matrix diagonal with the upper diagonal being all -inf which makes the values basically zero in our softmax function. Dropout is also introduced to prevent overfitting. These methods are applied before we calculate the final context vector. Lastly multi-head attention was introduced where we "stack" multiple causal attention heads on top of each other, but to save resources we instead calculate all heads in parallell and then combine them at the end. This is done by adding a head dimension to our tensor. 

### Chapter 4: Implementing a GPT model from scratch to generate text
In this chapter we start implementing a small GPT-2 model with 124M parameters. The chapter starts of by introducing normalization layers which helps prevent vanishing/exploding gradients. This is done by calculating the mean and variance seperately for all tokens and then for each token we subtract the mean and divide with the variance for all features in the token. Next we implemented a FeedForward network with a GELU as activation function and shortcut connections. With all this we were ready to build our transformer block, combining Multi-head attention layers, normalization layers, feedforward layers and shortcuts. Now the GPT model is done, stacking transformer blocks on each other then adding a final normalizing layer and lastly an output layer. We also looked at weight tying i.e. using the same layer as token embedding layer and output layer. We also started generating text using the GPT model. 

### Chapter 5: Pretraining on unlabeled data
Now that we have created a model we need to train it, otherwhise it only generates nonsense. To do that we need ways to evaluate the model, which was done using cross entropy loss. Cross entropy loss calculates the difference between two probability distributions, so in our case we get the logits (raw scores for each probable word), convert them to probabilites using softmax, and then compares our probability distribution with the correct one. Since the correct distribution is usually a one-hot encoding for the correct word, we, in practice, only compare our probability for that token: 
$$H(p,q)=-\sum_{i}p_i\log q_i$$
becomes
$$L=-1\times \log q_{correct}$$

We calculate the training and validation loss before training to get start values, start training the model, and every $$n$$ steps we evaluate the model by calculating the losses again. After training we can plot the losses to see how the model behaves. 

Right now every generated token is selected with only the highest probability. To add more "randomness" temperature scaling (dividing our logits with the temperature)was introduced to our model. This works but sometimes results in nonsensical outputs. To fix this top-k scaling was introduced, which sets all logits, excepts the top-k highest scores, to -inf before our softmax. This makes sure the model can only choose on of the higest k tokens, and removes the probability of choosing words which have nothing in common with the prior tokens. 

Lastly we looked at how to save the models parameters, which allows to reuse the model without training it again. We also looked at loading open weights from companies such as OpenAI since their models are better trained then what we can possibly do on a standard computer.  

### Chapter 6: Fine-tuning for classification
When loading open weights we get a model with general knowledge but it doesn't know our data or task. This we can solve by fine-tuning the model. Two common ways of fine-tuning are classification and instruction. Classification focuses on classifying using a set of labels e.g. filtering e-mail as spam or not spam. Instruction is more general and means that we train the model on a set of tasks using specific instructions e.g. translation.

In this chapter the focus was on classification fine-tuning on a dataset containing spam an normal emails. After loading the dataset there was a big difference in normal mails and spam mail. This was fixed by using a sub-set of the normal emails, mathcing the amount of spam emails. We then shuffled the dataset and split it into training, validation, and test datasets. Since the emails are not the same length padding (add <|endoftext|> tokens at the end of the shorter messages) was introduced so that the sliding window method could be used. 

Since we used a pre-trained model we had to change it slightly. Firstly, the output head consisted of 50257 output features whereas we only want 2 outputs now, 1 for spam and 0 for not spam. So we changed the models outputlayer to a linear layer with 2 outputs. And we don't want to retrain the whole model since the lower layers captures basic language structures and semantics, which we still need. Therefore the model was frozen, i.e. all layers becomes untrainable. Then our new output layer, the final normalizing layer and the last transformer block where unfreezed. Since theese layers are the ones capturing more specific features these are the ones we need to re-train (or fine-tune). 

A new loss function was introduced, focusing more on how many mails did the model corerectly classify. We then trained our model using the new dataset and new loss function and saw that the fine-tuned model performed well, scoring 96% accuracy on the held-out test set. 

### Chapter 7: Fine-tuning to follow instructions
The last chapter focused on instruction fine-tuning. Here a dataset following the format {'instruction': xxx, 'input': yyy, 'output': zzz} was used. The data was formated so that the model knows what it get and what it's expected to return. Once again the dataset was shuffled and split into a training set, validation set, and a test set. A custom collate function was created padding shorter messages so that they match the length of the longest one. A list of target token IDs was introduced for training and lastly some padding tokens was replaced with placeholders. By masking out all but the first padding token, the model will not learn from theese artificial tokens. They are also ignored in the loss calculation. The first padding token is kept since it allows the model to learn when a real sequence should end. 

A pre-trainde GPT2 model was loaded. The model was unchanged here since it's still used for generating text. This meant that the loss function and text generating functions from chapter 5 could be reused. The model was then trained and evaluated. Since the model is generating text answers it is not possible to look at accuracy as an evaluation method. Human evaluation is possible but when there's a large number of examples (1100 in this case) it's not feasible. Instead a more advanced model (Llama 3 in Ollama) was used to evaluate our models answers compared to the correct answers. This is done by sending the generated answer and the correct answer to the advanced model and asking it to return only a score from 0-100 on how similar the answers are, and then calculating an average over the entire dataset. Although this works LLMs are not deterministic and therefore they won't score the same every run, which could be a problem. 

### Exercise 7.4 Fine-tune with LoRA
Laslty the gpt2 model was fine-tuned on the classification dataset again, but this time using low-rank adaption, LoRA. With LoRA only a small low-rank subset of the model's parameters are adjusted, efficient for fine-tuning large models on task-specific data. In normal training and fine-tuning the model weights are updated during backpropagation using:
$$W_{\text{updated}} = W + \Delta W$$
LoRA offers a more efficient alternative where the $\Delta W$ matrix is approximated by:
$\Delta W \approx AB$
and by so: 
$$W_{\text{updated}} = W + AB$$
If $x$ is the input data, then we can write the following:
$$x (W+A B) = x W + x A B$$
Meaning that the LoRA matrices can be kept seperate from eachother. The models pre-trained weights can be kept frozen and only the new LoRA weights are trained, before being added to the original weights. The new weights are also scaled by alpha / rank which are two hyperparameters. 

Once again the models output head was changed into a linear layer with only 2 outputs since the model classifies mail as spam or not spam. We then freeze the original weights and wrap the original linear layers with the newly created LoRA layers. This drops the amount of trainable parameters from 124M to 2,7M. The model is then trained by reusing the training function from chapter 6. After training the model manages to score an accuracy of 99% on the test dataset, beating the normal pre-trained model. 

