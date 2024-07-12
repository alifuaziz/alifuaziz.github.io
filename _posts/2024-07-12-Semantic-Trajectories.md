---
layout: notebook
filename: "2024-07-12-semantic-trajectories.ipynb"
---
Based on work from [Matthew Nour](https://matthewnour.com), I have become interested in the trajectories through semantic meaning space for natural language. Particularly demonstrated in his paper [Trajectories through semantic spaces in schizophrenia and the relationship to ripple bursts](https://www.pnas.org/doi/10.1073/pnas.2305290120). 

Here I go through the required code to make a much more basic version of this type of analysis (to demonstrate to myself that I understand how to use semantic embeedings that from natural langugae models).

I used [this](https://mccormickml.com/2019/05/14/BERT-word-embeddings-tutorial/) tutorial to implement the tokeniser and the BERT model to get the semantic embeddings. 


```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="darkgrid")
import numpy as np
# PCA libraries
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
# Load model directly
import torch
device = torch.device("mps")
from transformers import AutoTokenizer, AutoModelForMaskedLM
from transformers import BertTokenizer, BertModel
# OPTIONAL: if you want to have more information on what's happening, activate the logger as follows
import logging
logging.basicConfig(level=logging.INFO)

```


```python
# Load pre-trained model tokenizer (vocabulary)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = AutoModelForMaskedLM.from_pretrained("google-bert/bert-base-uncased")

# Now you can use the tokenizer to encode text with the new special tokens
text = "This is an example with a special token <SPECIAL1>."
encoded = tokenizer.encode(text)
decoded = tokenizer.decode(encoded)

print(f"Encoded: {encoded}")
print(f"Decoded: {decoded}")

```

    Some weights of the model checkpoint at google-bert/bert-base-uncased were not used when initializing BertForMaskedLM: ['bert.pooler.dense.bias', 'bert.pooler.dense.weight', 'cls.seq_relationship.bias', 'cls.seq_relationship.weight']
    - This IS expected if you are initializing BertForMaskedLM from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
    - This IS NOT expected if you are initializing BertForMaskedLM from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).


    Encoded: [101, 2023, 2003, 2019, 2742, 2007, 1037, 2569, 19204, 1026, 2569, 2487, 1028, 1012, 102]
    Decoded: [CLS] this is an example with a special token < special1 >. [SEP]



```python
# Define a new example sentence with multiple meanings of the word "bank"
text = "Wikipedia's purpose is to benefit readers by presenting information on all branches of knowledge. Hosted by the Wikimedia Foundation, it consists of freely editable content, whose articles also have numerous links to guide readers towards more information.\
    Written collaboratively by largely anonymous volunteers known as Wikipedians, Wikipedia articles can be edited by anyone with Internet access, except in limited cases where editing is restricted to prevent disruption or vandalism. Since its creation on January 15, 2001, it has grown into the world's largest reference website, attracting over a billion visitors monthly. Wikipedia currently has more than sixty-three million articles in more than 300 languages, including 6,850,354 articles in English, with 114,409 active contributors in the past month.\
    Wikipedia's fundamental principles are summarized in its five pillars. The Wikipedia community has developed many policies and guidelines, although editors do not need to be familiar with them before contributing."
# Source of the text https://en.wikipedia.org/wiki/Wikipedia:About 
# Tokenize the text
tokenized_text = tokenizer.encode(text, add_special_tokens=True)
marked_text = tokenizer.decode(tokenized_text)

# Print out the marked text
marked_text
```




    "[CLS] wikipedia's purpose is to benefit readers by presenting information on all branches of knowledge. hosted by the wikimedia foundation, it consists of freely editable content, whose articles also have numerous links to guide readers towards more information. written collaboratively by largely anonymous volunteers known as wikipedians, wikipedia articles can be edited by anyone with internet access, except in limited cases where editing is restricted to prevent disruption or vandalism. since its creation on january 15, 2001, it has grown into the world's largest reference website, attracting over a billion visitors monthly. wikipedia currently has more than sixty - three million articles in more than 300 languages, including 6, 850, 354 articles in english, with 114, 409 active contributors in the past month. wikipedia's fundamental principles are summarized in its five pillars. the wikipedia community has developed many policies and guidelines, although editors do not need to be familiar with them before contributing. [SEP]"




```python
# Split the sentence into tokens.
tokenized_text = tokenizer.tokenize(marked_text)

# Map the token strings to their vocabulary indeces.
indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

# Display the words with their indices.
for tup in zip(tokenized_text, indexed_tokens):
    print('{:<12} {:>6,}'.format(tup[0], tup[1]))

```


```python
# Model is trained on sentence and sentence pairs. so the input needs to be sentence or sentence pairs.

# Mark each of the 22 tokens as belonging to sentence "1".
# This is for one sentence
segments_ids = [1] * len(tokenized_text)

print(segments_ids)

```


```python
# Convert inputs to PyTorch tensors
tokens_tensor    = torch.tensor([indexed_tokens])
segments_tensors = torch.tensor([segments_ids])
tokens_tensor.shape
```




    torch.Size([1, 189])




```python
# Load pre-trained model (weights)
model = BertModel.from_pretrained('bert-base-uncased',
                                  output_hidden_states = True, # Whether the model returns all hidden-states.
                                  )

# Put the model in "evaluation" mode, meaning feed-forward operation.
model.eval()


```




    BertModel(
      (embeddings): BertEmbeddings(
        (word_embeddings): Embedding(30522, 768, padding_idx=0)
        (position_embeddings): Embedding(512, 768)
        (token_type_embeddings): Embedding(2, 768)
        (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
        (dropout): Dropout(p=0.1, inplace=False)
      )
      (encoder): BertEncoder(
        (layer): ModuleList(
          (0-11): 12 x BertLayer(
            (attention): BertAttention(
              (self): BertSdpaSelfAttention(
                (query): Linear(in_features=768, out_features=768, bias=True)
                (key): Linear(in_features=768, out_features=768, bias=True)
                (value): Linear(in_features=768, out_features=768, bias=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
              (output): BertSelfOutput(
                (dense): Linear(in_features=768, out_features=768, bias=True)
                (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
            )
            (intermediate): BertIntermediate(
              (dense): Linear(in_features=768, out_features=3072, bias=True)
              (intermediate_act_fn): GELUActivation()
            )
            (output): BertOutput(
              (dense): Linear(in_features=3072, out_features=768, bias=True)
              (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
              (dropout): Dropout(p=0.1, inplace=False)
            )
          )
        )
      )
      (pooler): BertPooler(
        (dense): Linear(in_features=768, out_features=768, bias=True)
        (activation): Tanh()
      )
    )




```python
# with torch.no_grad():
#     wordembeddings = outputs.last_hidden_state
    

# Generate embeddings using BERT model
with torch.no_grad():
	outputs = model(tokens_tensor, segments_tensors)
	hidden_states = outputs
	# word_embeddings = 

# Output the shape of word embeddings
# print(f"Shape of Word Embeddings: {word_embeddings.shape}")
hidden_states.last_hidden_state.shape
```




    torch.Size([1, 189, 768])



Now for convience, I am going to put the words and their corresponding vectors into a table using a pandas dataframe


```python
# For each token word, vector pair
embedding_dict = {}
for word, vector in zip(marked_text.split(" "), hidden_states.last_hidden_state[0]):
    embedding_dict[word] = vector

embedding_dict = pd.DataFrame(embedding_dict)
    
```

Here it is!


```python
embedding_dict
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>[CLS]</th>
      <th>wikipedia's</th>
      <th>purpose</th>
      <th>is</th>
      <th>to</th>
      <th>benefit</th>
      <th>readers</th>
      <th>by</th>
      <th>presenting</th>
      <th>information</th>
      <th>...</th>
      <th>although</th>
      <th>editors</th>
      <th>do</th>
      <th>not</th>
      <th>need</th>
      <th>familiar</th>
      <th>them</th>
      <th>before</th>
      <th>contributing.</th>
      <th>[SEP]</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-0.397800</td>
      <td>0.225758</td>
      <td>-0.216066</td>
      <td>0.181724</td>
      <td>-0.548865</td>
      <td>-0.476328</td>
      <td>-0.187053</td>
      <td>0.327365</td>
      <td>0.021675</td>
      <td>0.026138</td>
      <td>...</td>
      <td>-0.463596</td>
      <td>0.326398</td>
      <td>-0.296987</td>
      <td>-0.249625</td>
      <td>0.021006</td>
      <td>-1.213159</td>
      <td>-0.253932</td>
      <td>-0.900819</td>
      <td>0.783844</td>
      <td>1.167579</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.170911</td>
      <td>-0.398236</td>
      <td>-0.105627</td>
      <td>0.584359</td>
      <td>-0.302917</td>
      <td>0.331164</td>
      <td>0.303018</td>
      <td>0.716316</td>
      <td>0.444830</td>
      <td>0.641611</td>
      <td>...</td>
      <td>0.180648</td>
      <td>0.349346</td>
      <td>0.121220</td>
      <td>-0.376064</td>
      <td>-0.166359</td>
      <td>-0.399527</td>
      <td>-0.701908</td>
      <td>-0.698676</td>
      <td>0.144448</td>
      <td>0.624116</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-0.407462</td>
      <td>0.499899</td>
      <td>0.240278</td>
      <td>0.510616</td>
      <td>0.623764</td>
      <td>0.183634</td>
      <td>0.364873</td>
      <td>0.250772</td>
      <td>1.127726</td>
      <td>-0.114914</td>
      <td>...</td>
      <td>0.406162</td>
      <td>0.404217</td>
      <td>0.172939</td>
      <td>0.466160</td>
      <td>0.511995</td>
      <td>0.033170</td>
      <td>-0.047894</td>
      <td>-0.308427</td>
      <td>-0.287521</td>
      <td>0.112691</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.120987</td>
      <td>-0.609629</td>
      <td>-0.559944</td>
      <td>-0.017765</td>
      <td>-0.101877</td>
      <td>-0.391126</td>
      <td>0.335346</td>
      <td>-0.113088</td>
      <td>0.139200</td>
      <td>-0.179634</td>
      <td>...</td>
      <td>-0.306918</td>
      <td>-0.289880</td>
      <td>-0.302768</td>
      <td>-0.239914</td>
      <td>-0.123906</td>
      <td>0.114429</td>
      <td>0.511084</td>
      <td>-0.473308</td>
      <td>0.306821</td>
      <td>-0.123258</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-0.333791</td>
      <td>-0.200983</td>
      <td>-0.451513</td>
      <td>0.486447</td>
      <td>-0.030032</td>
      <td>0.181724</td>
      <td>0.465903</td>
      <td>-0.647177</td>
      <td>0.083822</td>
      <td>-0.200970</td>
      <td>...</td>
      <td>-0.360363</td>
      <td>-0.306351</td>
      <td>-0.269635</td>
      <td>0.295733</td>
      <td>-0.080058</td>
      <td>0.246851</td>
      <td>0.131162</td>
      <td>-0.036426</td>
      <td>-0.169029</td>
      <td>-0.060356</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>763</th>
      <td>-0.258199</td>
      <td>0.958463</td>
      <td>-0.121484</td>
      <td>0.079637</td>
      <td>-0.564082</td>
      <td>0.262332</td>
      <td>-0.668383</td>
      <td>-0.263841</td>
      <td>-0.074577</td>
      <td>-0.483026</td>
      <td>...</td>
      <td>-0.040582</td>
      <td>-0.376516</td>
      <td>-0.050971</td>
      <td>0.854265</td>
      <td>-0.172597</td>
      <td>-0.265581</td>
      <td>-0.099148</td>
      <td>0.094765</td>
      <td>-0.592551</td>
      <td>-0.996676</td>
    </tr>
    <tr>
      <th>764</th>
      <td>-0.155529</td>
      <td>0.413391</td>
      <td>0.373446</td>
      <td>-0.071976</td>
      <td>-0.211906</td>
      <td>0.460308</td>
      <td>0.277623</td>
      <td>-0.627632</td>
      <td>0.622187</td>
      <td>0.092060</td>
      <td>...</td>
      <td>0.413279</td>
      <td>0.135485</td>
      <td>0.302015</td>
      <td>-0.061225</td>
      <td>0.140509</td>
      <td>-0.230951</td>
      <td>-0.088816</td>
      <td>-0.148471</td>
      <td>-0.005205</td>
      <td>-0.088604</td>
    </tr>
    <tr>
      <th>765</th>
      <td>-0.085188</td>
      <td>0.027113</td>
      <td>0.437014</td>
      <td>-0.187558</td>
      <td>-0.299858</td>
      <td>0.092524</td>
      <td>-0.260553</td>
      <td>-0.045953</td>
      <td>-0.553185</td>
      <td>0.215945</td>
      <td>...</td>
      <td>0.167228</td>
      <td>0.209997</td>
      <td>-0.126639</td>
      <td>-0.028156</td>
      <td>-0.118846</td>
      <td>-0.522136</td>
      <td>0.081270</td>
      <td>-0.387698</td>
      <td>0.275690</td>
      <td>0.131839</td>
    </tr>
    <tr>
      <th>766</th>
      <td>0.646723</td>
      <td>0.451272</td>
      <td>0.320475</td>
      <td>-0.405532</td>
      <td>0.323747</td>
      <td>0.029601</td>
      <td>0.372685</td>
      <td>0.031079</td>
      <td>0.340696</td>
      <td>0.257735</td>
      <td>...</td>
      <td>-0.221704</td>
      <td>-0.137952</td>
      <td>0.445956</td>
      <td>0.071810</td>
      <td>-0.147727</td>
      <td>0.250231</td>
      <td>0.636241</td>
      <td>0.272613</td>
      <td>-0.542538</td>
      <td>0.278729</td>
    </tr>
    <tr>
      <th>767</th>
      <td>1.123875</td>
      <td>0.424794</td>
      <td>0.754408</td>
      <td>1.082577</td>
      <td>-0.079018</td>
      <td>0.964547</td>
      <td>1.021484</td>
      <td>0.416334</td>
      <td>0.475320</td>
      <td>0.995239</td>
      <td>...</td>
      <td>0.639198</td>
      <td>-0.095425</td>
      <td>0.116375</td>
      <td>0.492220</td>
      <td>0.091785</td>
      <td>0.176961</td>
      <td>0.047007</td>
      <td>-0.673351</td>
      <td>-0.267106</td>
      <td>0.408299</td>
    </tr>
  </tbody>
</table>
<p>768 rows × 121 columns</p>
</div>



For plotting, we can't plot 768 dimensions, so we will reduce the dimensions to 2 using PCA. The dimensions we are going to use are the first two dimensions of the PCA. This will change for the particular set of text that is being analyzed.


```python
# PCA to reduce the dimensionality of the flow_dataframe to 2D
pca = PCA(n_components=2)
pca.fit_transform(embedding_dict.T)
pca.components_.shape, embedding_dict.shape

# Project the flow_dataframe to the 2D PC space using the PCA components
output = embedding_dict.T @ pca.components_.T


output.T

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>[CLS]</th>
      <th>wikipedia's</th>
      <th>purpose</th>
      <th>is</th>
      <th>to</th>
      <th>benefit</th>
      <th>readers</th>
      <th>by</th>
      <th>presenting</th>
      <th>information</th>
      <th>...</th>
      <th>although</th>
      <th>editors</th>
      <th>do</th>
      <th>not</th>
      <th>need</th>
      <th>familiar</th>
      <th>them</th>
      <th>before</th>
      <th>contributing.</th>
      <th>[SEP]</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-0.738609</td>
      <td>0.135391</td>
      <td>-0.460528</td>
      <td>-2.991425</td>
      <td>-0.811570</td>
      <td>0.073405</td>
      <td>7.045363</td>
      <td>-2.053942</td>
      <td>-5.218218</td>
      <td>-0.179280</td>
      <td>...</td>
      <td>1.233771</td>
      <td>1.876566</td>
      <td>6.532729</td>
      <td>2.686400</td>
      <td>1.114694</td>
      <td>1.714168</td>
      <td>1.446539</td>
      <td>0.978354</td>
      <td>-0.590626</td>
      <td>-9.598780</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.410755</td>
      <td>6.602470</td>
      <td>0.050615</td>
      <td>-3.725148</td>
      <td>2.182217</td>
      <td>-1.167463</td>
      <td>-3.344800</td>
      <td>-2.576366</td>
      <td>1.447452</td>
      <td>-1.652778</td>
      <td>...</td>
      <td>2.879586</td>
      <td>3.578430</td>
      <td>2.824045</td>
      <td>3.603118</td>
      <td>4.511847</td>
      <td>3.351452</td>
      <td>4.518424</td>
      <td>3.172381</td>
      <td>0.603879</td>
      <td>5.161048</td>
    </tr>
  </tbody>
</table>
<p>2 rows × 121 columns</p>
</div>



Here is the plot of the words in the semantic space and their flow through the meaning space. 


```python
# Plot the flow_dataframe in a vector field plot
s = 7
plt.figure(figsize=(s, s))
# remove the grid
plt.grid(False)
plt.title("Vector Field Plot of FlowDataFrame")

cmap = plt.get_cmap('magma')

# Plot arrows
for index in range(0,output.T.shape[1]-1):
    color = cmap(index/output.T.shape[1])
    # Add label to the color of the first and last arrow
    if index == 0:
        plt.arrow(output.iloc[index, 0],
                  output.iloc[index, 1],
                  output.iloc[index+1, 0] - output.iloc[index, 0],
                  output.iloc[index+1, 1] - output.iloc[index, 1],
                  head_width =0.2, 
                  head_length=0.2, 
                  fc=color, 
                  ec=color,
                  label="Start"
                  )
    elif index == output.T.shape[1]-2:
        plt.arrow(output.iloc[index, 0],
                  output.iloc[index, 1],
                  output.iloc[index+1, 0] - output.iloc[index, 0],
                  output.iloc[index+1, 1] - output.iloc[index, 1],
                  head_width =0.2, 
                  head_length=0.2, 
                  fc=color, 
                  ec=color,
                  label="End"
                  )


    plt.arrow(output.iloc[index, 0],
              output.iloc[index, 1],
              output.iloc[index+1, 0] - output.iloc[index, 0],
              output.iloc[index+1, 1] - output.iloc[index, 1],
              head_width =0.2, 
              head_length=0.2, 
              fc=color, 
              ec=color
              )

plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend()
plt.show()
```


    
![png]({{ site.url }}{{ site.baseurl }}/assets/images/2024-07-12-semantic-trajectories_files/2024-07-12-semantic-trajectories_16_0.png)
    


This is a very basic version of the analysis. A simple thing do to to improve it would be to only embed words that are nowns as that might remove some of the noise of the data. 

Nevertheless, this is a demonstration of how to visualise the embeddings of the words as they move through the space. 

