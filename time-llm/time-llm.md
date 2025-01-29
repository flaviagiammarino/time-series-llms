# Time Series Forecasting with Time-LLM

## Set Up

Install the dependencies.


```python
!pip install transformers sentencepiece kaleido
```

    Collecting transformers
      Downloading transformers-4.48.1-py3-none-any.whl.metadata (44 kB)
    Collecting sentencepiece
      Downloading sentencepiece-0.2.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (7.7 kB)
    Collecting kaleido
      Downloading kaleido-0.2.1-py2.py3-none-manylinux1_x86_64.whl.metadata (15 kB)
    Requirement already satisfied: filelock in /home/ec2-user/anaconda3/envs/pytorch_p310/lib/python3.10/site-packages (from transformers) (3.16.1)
    Collecting huggingface-hub<1.0,>=0.24.0 (from transformers)
      Downloading huggingface_hub-0.28.0-py3-none-any.whl.metadata (13 kB)
    Requirement already satisfied: numpy>=1.17 in /home/ec2-user/anaconda3/envs/pytorch_p310/lib/python3.10/site-packages (from transformers) (1.26.4)
    Requirement already satisfied: packaging>=20.0 in /home/ec2-user/anaconda3/envs/pytorch_p310/lib/python3.10/site-packages (from transformers) (21.3)
    Requirement already satisfied: pyyaml>=5.1 in /home/ec2-user/anaconda3/envs/pytorch_p310/lib/python3.10/site-packages (from transformers) (6.0.2)
    Collecting regex!=2019.12.17 (from transformers)
      Downloading regex-2024.11.6-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (40 kB)
    Requirement already satisfied: requests in /home/ec2-user/anaconda3/envs/pytorch_p310/lib/python3.10/site-packages (from transformers) (2.32.3)
    Collecting tokenizers<0.22,>=0.21 (from transformers)
      Downloading tokenizers-0.21.0-cp39-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (6.7 kB)
    Collecting safetensors>=0.4.1 (from transformers)
      Downloading safetensors-0.5.2-cp38-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (3.8 kB)
    Requirement already satisfied: tqdm>=4.27 in /home/ec2-user/anaconda3/envs/pytorch_p310/lib/python3.10/site-packages (from transformers) (4.67.1)
    Requirement already satisfied: fsspec>=2023.5.0 in /home/ec2-user/anaconda3/envs/pytorch_p310/lib/python3.10/site-packages (from huggingface-hub<1.0,>=0.24.0->transformers) (2024.12.0)
    Requirement already satisfied: typing-extensions>=3.7.4.3 in /home/ec2-user/anaconda3/envs/pytorch_p310/lib/python3.10/site-packages (from huggingface-hub<1.0,>=0.24.0->transformers) (4.12.2)
    Requirement already satisfied: pyparsing!=3.0.5,>=2.0.2 in /home/ec2-user/anaconda3/envs/pytorch_p310/lib/python3.10/site-packages (from packaging>=20.0->transformers) (3.2.1)
    Requirement already satisfied: charset_normalizer<4,>=2 in /home/ec2-user/anaconda3/envs/pytorch_p310/lib/python3.10/site-packages (from requests->transformers) (3.4.1)
    Requirement already satisfied: idna<4,>=2.5 in /home/ec2-user/anaconda3/envs/pytorch_p310/lib/python3.10/site-packages (from requests->transformers) (3.10)
    Requirement already satisfied: urllib3<3,>=1.21.1 in /home/ec2-user/anaconda3/envs/pytorch_p310/lib/python3.10/site-packages (from requests->transformers) (2.3.0)
    Requirement already satisfied: certifi>=2017.4.17 in /home/ec2-user/anaconda3/envs/pytorch_p310/lib/python3.10/site-packages (from requests->transformers) (2024.12.14)
    Downloading transformers-4.48.1-py3-none-any.whl (9.7 MB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m9.7/9.7 MB[0m [31m137.8 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading sentencepiece-0.2.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (1.3 MB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m1.3/1.3 MB[0m [31m121.3 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading kaleido-0.2.1-py2.py3-none-manylinux1_x86_64.whl (79.9 MB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m79.9/79.9 MB[0m [31m162.3 MB/s[0m eta [36m0:00:00[0m00:01[0m00:01[0m
    [?25hDownloading huggingface_hub-0.28.0-py3-none-any.whl (464 kB)
    Downloading regex-2024.11.6-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (781 kB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m781.7/781.7 kB[0m [31m81.3 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading safetensors-0.5.2-cp38-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (461 kB)
    Downloading tokenizers-0.21.0-cp39-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (3.0 MB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m3.0/3.0 MB[0m [31m172.1 MB/s[0m eta [36m0:00:00[0m
    [?25hInstalling collected packages: sentencepiece, kaleido, safetensors, regex, huggingface-hub, tokenizers, transformers
    Successfully installed huggingface-hub-0.28.0 kaleido-0.2.1 regex-2024.11.6 safetensors-0.5.2 sentencepiece-0.2.0 tokenizers-0.21.0 transformers-4.48.1


Import the dependencies.

**Note:** You need to clone the [Time-LLM GitHub repository](https://github.com/KimMeen/Time-LLM) and then move this notebook to the folder with the Time-LMM source code.


```python
import os
import time
import types
import random
import torch
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

from models.TimeLLM import Model
```

Set the device.


```python
device = torch.device("cuda:0")
```

Fix all random seeds.


```python
random_seed = 0
np.random.seed(random_seed)
random.seed(random_seed)
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

## Data

Get the data. 


```python
df = pd.read_csv(
    "https://raw.githubusercontent.com/jbrownlee/Datasets/refs/heads/master/airline-passengers.csv",
    parse_dates=["Month"],
    dtype=float
)
```


```python
df.shape
```




    (144, 2)




```python
df.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Month</th>
      <th>Passengers</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1949-01-01</td>
      <td>112.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1949-02-01</td>
      <td>118.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1949-03-01</td>
      <td>132.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1949-04-01</td>
      <td>129.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1949-05-01</td>
      <td>121.0</td>
    </tr>
  </tbody>
</table>
</div>



Visualize the data.


```python
fig = go.Figure(
    layout=dict(
        paper_bgcolor="white",
        plot_bgcolor="white",
        margin=dict(t=10, b=10, l=10, r=10),
        width=900,
        height=400,
        xaxis=dict(
            mirror=True,
            linecolor="#24292f",
            gridcolor="#d0d7de",
            gridwidth=1,
            tickfont=dict(
                color="#1b1f24"
            ),
        ),
        yaxis=dict(
            mirror=True,
            linecolor="#24292f",
            gridcolor="#d0d7de",
            gridwidth=1,
            tickfont=dict(
                color="#1b1f24"
            ),
        ),
    ),
    data=[
        go.Scatter(
            x=df["Month"],
            y=df["Passengers"],
            mode="lines",
            name="Actual",
            line=dict(
                color="#8c959f",
                width=2
            )
        )
    ]
)
fig.show("png")
```


    
![png](https://demo-projects-files.s3.eu-west-1.amazonaws.com/time-llm/figure_1.png)
    


Define the sequence lenghts.


```python
seq_len = 24
pred_len = 12
```

Split the data into training and test sets.


```python
df_train = df[["Passengers"]].iloc[:- pred_len]
df_test = df[["Passengers"]].iloc[- (seq_len + pred_len):]
```

Calculate the scaling parameters.


```python
mu = df_train.mean(axis=0).item()
sigma = df_train.std(axis=0, ddof=1).item()
```

Scale the data.


```python
df_train = (df_train - mu) / sigma
df_test = (df_test - mu) / sigma
```

Define a function for splitting the data into sequences. 


```python
def get_sequences(df, seq_len, pred_len):
    x = []
    y = []
    for t in range(seq_len, len(df) - pred_len + 1):
        x.append(df.iloc[t - seq_len: t].values)
        y.append(df.iloc[t: t + pred_len].values)
    x = np.array(x)
    y = np.array(y)
    return x, y
```

Generate the training sequences.


```python
x_train, y_train = get_sequences(df_train, seq_len, pred_len)
```

Generate the test sequences.


```python
x_test, y_test = get_sequences(df_test, seq_len, pred_len)
```

## Model

### Training

Define the training parameters.


```python
batch_size = 8
lr = 0.001
epochs = 40
```

Create the training dataset.


```python
dataset = torch.utils.data.TensorDataset(
    torch.from_numpy(x_train).float(),
    torch.from_numpy(y_train).float()
)
```

Create the training dataloader.


```python
dataloader = torch.utils.data.DataLoader(
    dataset=dataset,
    batch_size=batch_size,
    shuffle=True,
)
```

Instantiate the model.


```python
model = Model(
    configs=types.SimpleNamespace(
        prompt_domain=True,
        content="Monthly totals of a airline passengers from USA, from January 1949 through December 1960.",
        task_name="short_term_forecast",  # not used
        enc_in=None,  # not used
        pred_len=pred_len,
        seq_len=seq_len,
        llm_model="LLAMA",
        llm_dim=4096,
        llm_layers=1,
        d_model=32,
        d_ff=32,
        patch_len=16,
        stride=8,
        n_heads=4,
        dropout=0,
    )
)
model.to(torch.bfloat16)
model.to(device)
```


    config.json:   0%|          | 0.00/594 [00:00<?, ?B/s]



    model.safetensors.index.json:   0%|          | 0.00/26.8k [00:00<?, ?B/s]



    Downloading shards:   0%|          | 0/2 [00:00<?, ?it/s]



    model-00001-of-00002.safetensors:   0%|          | 0.00/9.98G [00:00<?, ?B/s]



    model-00002-of-00002.safetensors:   0%|          | 0.00/3.50G [00:00<?, ?B/s]



    Loading checkpoint shards:   0%|          | 0/2 [00:00<?, ?it/s]



    tokenizer_config.json:   0%|          | 0.00/2.28k [00:00<?, ?B/s]



    tokenizer.model:   0%|          | 0.00/500k [00:00<?, ?B/s]



    special_tokens_map.json:   0%|          | 0.00/411 [00:00<?, ?B/s]



    tokenizer.json:   0%|          | 0.00/1.84M [00:00<?, ?B/s]





    Model(
      (llm_model): LlamaModel(
        (embed_tokens): Embedding(32000, 4096, padding_idx=0)
        (layers): ModuleList(
          (0): LlamaDecoderLayer(
            (self_attn): LlamaAttention(
              (q_proj): Linear(in_features=4096, out_features=4096, bias=False)
              (k_proj): Linear(in_features=4096, out_features=4096, bias=False)
              (v_proj): Linear(in_features=4096, out_features=4096, bias=False)
              (o_proj): Linear(in_features=4096, out_features=4096, bias=False)
            )
            (mlp): LlamaMLP(
              (gate_proj): Linear(in_features=4096, out_features=11008, bias=False)
              (up_proj): Linear(in_features=4096, out_features=11008, bias=False)
              (down_proj): Linear(in_features=11008, out_features=4096, bias=False)
              (act_fn): SiLU()
            )
            (input_layernorm): LlamaRMSNorm((4096,), eps=1e-06)
            (post_attention_layernorm): LlamaRMSNorm((4096,), eps=1e-06)
          )
        )
        (norm): LlamaRMSNorm((4096,), eps=1e-06)
        (rotary_emb): LlamaRotaryEmbedding()
      )
      (dropout): Dropout(p=0, inplace=False)
      (patch_embedding): PatchEmbedding(
        (padding_patch_layer): ReplicationPad1d()
        (value_embedding): TokenEmbedding(
          (tokenConv): Conv1d(16, 32, kernel_size=(3,), stride=(1,), padding=(1,), bias=False, padding_mode=circular)
        )
        (dropout): Dropout(p=0, inplace=False)
      )
      (mapping_layer): Linear(in_features=32000, out_features=1000, bias=True)
      (reprogramming_layer): ReprogrammingLayer(
        (query_projection): Linear(in_features=32, out_features=128, bias=True)
        (key_projection): Linear(in_features=4096, out_features=128, bias=True)
        (value_projection): Linear(in_features=4096, out_features=128, bias=True)
        (out_projection): Linear(in_features=128, out_features=4096, bias=True)
        (dropout): Dropout(p=0.1, inplace=False)
      )
      (output_projection): FlattenHead(
        (flatten): Flatten(start_dim=-2, end_dim=-1)
        (linear): Linear(in_features=96, out_features=12, bias=True)
        (dropout): Dropout(p=0, inplace=False)
      )
      (normalize_layers): Normalize()
    )




```python
print(f"Number of parameters: {format(sum(p.numel() for p in model.parameters()), ',.0f')}")
print(f"Number of frozen parameters: {format(sum(p.numel() for p in model.parameters() if not p.requires_grad), ',.0f')}")
print(f"Number of trainable parameters: {format(sum(p.numel() for p in model.parameters() if p.requires_grad), ',.0f')}")
```

    Number of parameters: 367,044,596
    Number of frozen parameters: 333,459,456
    Number of trainable parameters: 33,585,140


Instantiate the optimizer.


```python
optimizer = torch.optim.Adam(
    params=[p for p in model.parameters() if p.requires_grad],
    lr=lr
)
```

Train the model.


```python
start = time.time()
model.train()
for epoch in range(epochs):
    losses = []
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        yhat = model(
            x_enc=x,
            x_mark_enc=None,
            x_dec=None,
            x_mark_dec=None
        )
        loss = torch.nn.functional.mse_loss(yhat, y)
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
    print(f'epoch: {format(1 + epoch, ".0f")} loss: {format(np.mean(losses), ",.8f")}')
model.eval()
end = time.time()
print(f"Training time: {format(end - start, ',.2f')} seconds")
```

    epoch: 1 loss: 0.29851346
    epoch: 2 loss: 0.15955588
    epoch: 3 loss: 0.08472076
    epoch: 4 loss: 0.05967870
    epoch: 5 loss: 0.06023220
    epoch: 6 loss: 0.06349146
    epoch: 7 loss: 0.06728122
    epoch: 8 loss: 0.06126707
    epoch: 9 loss: 0.05353234
    epoch: 10 loss: 0.05658055
    epoch: 11 loss: 0.04822905
    epoch: 12 loss: 0.04843022
    epoch: 13 loss: 0.04471244
    epoch: 14 loss: 0.05880081
    epoch: 15 loss: 0.05236155
    epoch: 16 loss: 0.05229929
    epoch: 17 loss: 0.04131612
    epoch: 18 loss: 0.03974260
    epoch: 19 loss: 0.04010143
    epoch: 20 loss: 0.04502358
    epoch: 21 loss: 0.05139145
    epoch: 22 loss: 0.04821363
    epoch: 23 loss: 0.05030083
    epoch: 24 loss: 0.04305501
    epoch: 25 loss: 0.03821681
    epoch: 26 loss: 0.03685588
    epoch: 27 loss: 0.04157941
    epoch: 28 loss: 0.03874546
    epoch: 29 loss: 0.04502921
    epoch: 30 loss: 0.04382845
    epoch: 31 loss: 0.04558949
    epoch: 32 loss: 0.04334464
    epoch: 33 loss: 0.04241312
    epoch: 34 loss: 0.03419355
    epoch: 35 loss: 0.03252470
    epoch: 36 loss: 0.03761013
    epoch: 37 loss: 0.03659076
    epoch: 38 loss: 0.04324745
    epoch: 39 loss: 0.03865493
    epoch: 40 loss: 0.03808260
    Training time: 53.80 seconds


### Evaluation

Generate the test set predictions.


```python
start = time.time()
yhat_test = model(
    x_enc=torch.from_numpy(x_test).float().to(device),
    x_mark_enc=None,
    x_dec=None,
    x_mark_dec=None
).detach().cpu().numpy().flatten()
end = time.time()
print(f"Inference time: {format(end - start, ',.2f')} seconds")
```

    Inference time: 0.01 seconds


Transform the test set predictions back to the original scale.


```python
yhat_test = mu + sigma * yhat_test
```

Organize the test set predictions in a data frame.


```python
predictions = pd.DataFrame(
    data=yhat_test.astype(int),
    columns=["Passengers"],
    index=df["Month"].iloc[-pred_len:]
).reset_index()
```


```python
predictions
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Month</th>
      <th>Passengers</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1960-01-01</td>
      <td>391</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1960-02-01</td>
      <td>388</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1960-03-01</td>
      <td>444</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1960-04-01</td>
      <td>443</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1960-05-01</td>
      <td>455</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1960-06-01</td>
      <td>540</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1960-07-01</td>
      <td>617</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1960-08-01</td>
      <td>618</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1960-09-01</td>
      <td>527</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1960-10-01</td>
      <td>438</td>
    </tr>
    <tr>
      <th>10</th>
      <td>1960-11-01</td>
      <td>384</td>
    </tr>
    <tr>
      <th>11</th>
      <td>1960-12-01</td>
      <td>426</td>
    </tr>
  </tbody>
</table>
</div>



Visualize the test set predictions.


```python
fig = go.Figure(
    layout=dict(
        paper_bgcolor="white",
        plot_bgcolor="white",
        margin=dict(t=10, b=10, l=10, r=10),
        legend=dict(font=dict(color="#1b1f24")),
        width=900,
        height=400,
        xaxis=dict(
            mirror=True,
            linecolor="#24292f",
            gridcolor="#d0d7de",
            gridwidth=1,
            tickfont=dict(
                color="#1b1f24"
            ),
        ),
        yaxis=dict(
            mirror=True,
            linecolor="#24292f",
            gridcolor="#d0d7de",
            gridwidth=1,
            tickfont=dict(
                color="#1b1f24"
            ),
        ),
    ),
    data=[
        go.Scatter(
            x=df["Month"],
            y=df["Passengers"],
            mode="lines",
            name="Actual",
            line=dict(
                color="#8c959f",
                width=2
            )
        ),
        go.Scatter(
            x=predictions["Month"],
            y=predictions["Passengers"],
            mode="lines",
            name="Test Set Prediction",
            line=dict(
                color="#fb8f44",
                width=4,
                dash="dot",
            )
        ),
    ]
)
fig.show("png")
```


    
![png](https://demo-projects-files.s3.eu-west-1.amazonaws.com/time-llm/figure_2.png)
    


Calculate the test error.


```python
print("Mean Absolute Error (MAE):", round(mean_absolute_error(y_true=df["Passengers"].iloc[-pred_len:], y_pred=predictions["Passengers"]), 2))
print("Root Mean Squared Error (RMSE):", round(root_mean_squared_error(y_true=df["Passengers"].iloc[-pred_len:], y_pred=predictions["Passengers"]), 2))
```

    Mean Absolute Error (MAE): 13.75
    Root Mean Squared Error (RMSE): 16.02


### Inference

Generate the forecasts.


```python
start = time.time()
yhat_future = model(
    x_enc=torch.from_numpy(np.expand_dims(df_test.iloc[-seq_len:], axis=0)).float().to(device),
    x_mark_enc=None,
    x_dec=None,
    x_mark_dec=None
).detach().cpu().numpy().flatten()
end = time.time()
print(f"Inference time: {format(end - start, ',.2f')} seconds")
```

    Inference time: 0.01 seconds


Transform the forecasts back to the original scale.


```python
yhat_future = mu + sigma * yhat_future
```

Organize the forecasts in a data frame.


```python
forecasts = pd.DataFrame(
    data=yhat_future.astype(int),
    columns=["Passengers"],
    index=pd.Series(
        data=pd.date_range(
            start=df["Month"].iloc[-1] + pd.tseries.offsets.MonthBegin(1),
            end=df["Month"].iloc[-1] + pd.tseries.offsets.MonthBegin(pred_len),
            freq="MS"
        ),
        name="Month"
    )
).reset_index()
```


```python
forecasts
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Month</th>
      <th>Passengers</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1961-01-01</td>
      <td>437</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1961-02-01</td>
      <td>434</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1961-03-01</td>
      <td>496</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1961-04-01</td>
      <td>494</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1961-05-01</td>
      <td>507</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1961-06-01</td>
      <td>600</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1961-07-01</td>
      <td>684</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1961-08-01</td>
      <td>685</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1961-09-01</td>
      <td>585</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1961-10-01</td>
      <td>489</td>
    </tr>
    <tr>
      <th>10</th>
      <td>1961-11-01</td>
      <td>430</td>
    </tr>
    <tr>
      <th>11</th>
      <td>1961-12-01</td>
      <td>475</td>
    </tr>
  </tbody>
</table>
</div>



Visualize the forecasts.


```python
fig = go.Figure(
    layout=dict(
        paper_bgcolor="white",
        plot_bgcolor="white",
        margin=dict(t=10, b=10, l=10, r=10),
        legend=dict(font=dict(color="#1b1f24")),
        width=900,
        height=400,
        xaxis=dict(
            mirror=True,
            linecolor="#24292f",
            gridcolor="#d0d7de",
            gridwidth=1,
            tickfont=dict(
                color="#1b1f24"
            ),
        ),
        yaxis=dict(
            mirror=True,
            linecolor="#24292f",
            gridcolor="#d0d7de",
            gridwidth=1,
            tickfont=dict(
                color="#1b1f24"
            ),
        ),
    ),
    data=[
        go.Scatter(
            x=df["Month"],
            y=df["Passengers"],
            mode="lines",
            name="Actual",
            line=dict(
                color="#8c959f",
                width=2
            )
        ),
        go.Scatter(
            x=predictions["Month"],
            y=predictions["Passengers"],
            mode="lines",
            name="Test Set Prediction",
            line=dict(
                color="#fb8f44",
                width=4,
                dash="dot",
            )
        ),
        go.Scatter(
            x=forecasts["Month"],
            y=forecasts["Passengers"],
            mode="lines",
            name="Out-of-Sample Forecast",
            line=dict(
                color="#54aeff",
                width=4,
                dash="dot",
            )
        ),
    ]
)
fig.show("png")
```


    
![png](https://demo-projects-files.s3.eu-west-1.amazonaws.com/time-llm/figure_3.png)
    

