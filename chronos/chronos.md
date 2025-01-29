# Time Series Forecasting with Chronos

<img src=https://demo-projects-files.s3.eu-west-1.amazonaws.com/chronos/architecture.png style="width:65%;margin-top:50px;margin-bottom:50px"/>

*Chronos architecture (source: [doi:10.48550/arXiv.2403.07815](https://doi.org/10.48550/arXiv.2403.07815))*

## Set Up

Install the dependencies.


```python
!pip install git+https://github.com/amazon-science/chronos-forecasting.git transformers kaleido
```

    Collecting git+https://github.com/amazon-science/chronos-forecasting.git
      Cloning https://github.com/amazon-science/chronos-forecasting.git to /tmp/pip-req-build-y894v74y
      Running command git clone --filter=blob:none --quiet https://github.com/amazon-science/chronos-forecasting.git /tmp/pip-req-build-y894v74y
      Resolved https://github.com/amazon-science/chronos-forecasting.git to commit 39515ff0fcdae55bbbf546d90193dbe54b201556
      Installing build dependencies ... [?25ldone
    [?25h  Getting requirements to build wheel ... [?25ldone
    [?25h  Preparing metadata (pyproject.toml) ... [?25ldone
    [?25hRequirement already satisfied: transformers in /home/ec2-user/anaconda3/envs/pytorch_p310/lib/python3.10/site-packages (4.47.1)
    Requirement already satisfied: kaleido in /home/ec2-user/anaconda3/envs/pytorch_p310/lib/python3.10/site-packages (0.2.1)
    Requirement already satisfied: accelerate<1,>=0.32 in /home/ec2-user/anaconda3/envs/pytorch_p310/lib/python3.10/site-packages (from chronos-forecasting==1.4.1) (0.34.2)
    Requirement already satisfied: torch<2.6,>=2.0 in /home/ec2-user/anaconda3/envs/pytorch_p310/lib/python3.10/site-packages (from chronos-forecasting==1.4.1) (2.2.2)
    Requirement already satisfied: filelock in /home/ec2-user/anaconda3/envs/pytorch_p310/lib/python3.10/site-packages (from transformers) (3.16.1)
    Requirement already satisfied: huggingface-hub<1.0,>=0.24.0 in /home/ec2-user/anaconda3/envs/pytorch_p310/lib/python3.10/site-packages (from transformers) (0.28.0)
    Requirement already satisfied: numpy>=1.17 in /home/ec2-user/anaconda3/envs/pytorch_p310/lib/python3.10/site-packages (from transformers) (1.26.4)
    Requirement already satisfied: packaging>=20.0 in /home/ec2-user/anaconda3/envs/pytorch_p310/lib/python3.10/site-packages (from transformers) (21.3)
    Requirement already satisfied: pyyaml>=5.1 in /home/ec2-user/anaconda3/envs/pytorch_p310/lib/python3.10/site-packages (from transformers) (6.0.2)
    Requirement already satisfied: regex!=2019.12.17 in /home/ec2-user/anaconda3/envs/pytorch_p310/lib/python3.10/site-packages (from transformers) (2024.11.6)
    Requirement already satisfied: requests in /home/ec2-user/anaconda3/envs/pytorch_p310/lib/python3.10/site-packages (from transformers) (2.32.3)
    Requirement already satisfied: tokenizers<0.22,>=0.21 in /home/ec2-user/anaconda3/envs/pytorch_p310/lib/python3.10/site-packages (from transformers) (0.21.0)
    Requirement already satisfied: safetensors>=0.4.1 in /home/ec2-user/anaconda3/envs/pytorch_p310/lib/python3.10/site-packages (from transformers) (0.5.2)
    Requirement already satisfied: tqdm>=4.27 in /home/ec2-user/anaconda3/envs/pytorch_p310/lib/python3.10/site-packages (from transformers) (4.67.1)
    Requirement already satisfied: psutil in /home/ec2-user/anaconda3/envs/pytorch_p310/lib/python3.10/site-packages (from accelerate<1,>=0.32->chronos-forecasting==1.4.1) (6.1.1)
    Requirement already satisfied: fsspec>=2023.5.0 in /home/ec2-user/anaconda3/envs/pytorch_p310/lib/python3.10/site-packages (from huggingface-hub<1.0,>=0.24.0->transformers) (2024.12.0)
    Requirement already satisfied: typing-extensions>=3.7.4.3 in /home/ec2-user/anaconda3/envs/pytorch_p310/lib/python3.10/site-packages (from huggingface-hub<1.0,>=0.24.0->transformers) (4.12.2)
    Requirement already satisfied: pyparsing!=3.0.5,>=2.0.2 in /home/ec2-user/anaconda3/envs/pytorch_p310/lib/python3.10/site-packages (from packaging>=20.0->transformers) (3.2.1)
    Requirement already satisfied: sympy in /home/ec2-user/anaconda3/envs/pytorch_p310/lib/python3.10/site-packages (from torch<2.6,>=2.0->chronos-forecasting==1.4.1) (1.13.3)
    Requirement already satisfied: networkx in /home/ec2-user/anaconda3/envs/pytorch_p310/lib/python3.10/site-packages (from torch<2.6,>=2.0->chronos-forecasting==1.4.1) (3.4.2)
    Requirement already satisfied: jinja2 in /home/ec2-user/anaconda3/envs/pytorch_p310/lib/python3.10/site-packages (from torch<2.6,>=2.0->chronos-forecasting==1.4.1) (3.1.5)
    Requirement already satisfied: charset_normalizer<4,>=2 in /home/ec2-user/anaconda3/envs/pytorch_p310/lib/python3.10/site-packages (from requests->transformers) (3.4.1)
    Requirement already satisfied: idna<4,>=2.5 in /home/ec2-user/anaconda3/envs/pytorch_p310/lib/python3.10/site-packages (from requests->transformers) (3.10)
    Requirement already satisfied: urllib3<3,>=1.21.1 in /home/ec2-user/anaconda3/envs/pytorch_p310/lib/python3.10/site-packages (from requests->transformers) (2.3.0)
    Requirement already satisfied: certifi>=2017.4.17 in /home/ec2-user/anaconda3/envs/pytorch_p310/lib/python3.10/site-packages (from requests->transformers) (2024.12.14)
    Requirement already satisfied: MarkupSafe>=2.0 in /home/ec2-user/anaconda3/envs/pytorch_p310/lib/python3.10/site-packages (from jinja2->torch<2.6,>=2.0->chronos-forecasting==1.4.1) (3.0.2)
    Requirement already satisfied: mpmath<1.4,>=1.1.0 in /home/ec2-user/anaconda3/envs/pytorch_p310/lib/python3.10/site-packages (from sympy->torch<2.6,>=2.0->chronos-forecasting==1.4.1) (1.3.0)


Import the dependencies.


```python
import time
import torch
import transformers
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

from chronos import ChronosPipeline
```

## Data

Get the data. 


```python
df = pd.read_csv(
    "https://raw.githubusercontent.com/jbrownlee/Datasets/refs/heads/master/daily-min-temperatures.csv",
    parse_dates=["Date"],
    dtype=float
)
```


```python
df.shape
```




    (3650, 2)




```python
df.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Date</th>
      <th>Temp</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1981-01-01</td>
      <td>20.7</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1981-01-02</td>
      <td>17.9</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1981-01-03</td>
      <td>18.8</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1981-01-04</td>
      <td>14.6</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1981-01-05</td>
      <td>15.8</td>
    </tr>
  </tbody>
</table>
</div>



Resample the data.


```python
df = df.set_index("Date").resample("MS").mean().reset_index()
```


```python
df.shape
```




    (120, 2)




```python
df.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Date</th>
      <th>Temp</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1981-01-01</td>
      <td>17.712903</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1981-02-01</td>
      <td>17.678571</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1981-03-01</td>
      <td>13.500000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1981-04-01</td>
      <td>12.356667</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1981-05-01</td>
      <td>9.490323</td>
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
            range=[0, 25],
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
            x=df["Date"],
            y=df["Temp"],
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


    
![png](https://demo-projects-files.s3.eu-west-1.amazonaws.com/chronos/figure_1.png)
    


## Model

Instantiate the model.


```python
model = ChronosPipeline.from_pretrained(
    "amazon/chronos-t5-large",
    device_map="cuda",
    torch_dtype=torch.bfloat16,
)
```

Define the prediction length.


```python
prediction_length = 18
```

Define the number of samples.


```python
num_samples = 1000
```

### Evaluation

Generate the test set predictions.


```python
start = time.time()
transformers.set_seed(42)
yhat_test = model.predict(
    context=torch.from_numpy(df["Temp"].iloc[:-prediction_length].values),
    prediction_length=prediction_length,
    num_samples=num_samples
).detach().cpu().numpy().squeeze(axis=0)
end = time.time()
print(f"Inference time: {format(end - start, ',.2f')} seconds")
```

    Inference time: 2.76 seconds


Organize the test set predictions in a data frame.


```python
predictions = pd.DataFrame(
    data={
        "Mean": np.mean(yhat_test, axis=0),
        "Std. Dev.": np.std(yhat_test, ddof=1, axis=0),
    },
    index=df["Date"].iloc[-prediction_length:]
).reset_index()
```


```python
predictions.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Date</th>
      <th>Mean</th>
      <th>Std. Dev.</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1989-07-01</td>
      <td>7.143206</td>
      <td>0.937785</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1989-08-01</td>
      <td>8.343010</td>
      <td>0.971801</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1989-09-01</td>
      <td>9.337883</td>
      <td>0.921955</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1989-10-01</td>
      <td>10.664992</td>
      <td>0.835569</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1989-11-01</td>
      <td>12.383593</td>
      <td>1.111717</td>
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
        width=1100,
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
            range=[0, 25],
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
            x=df["Date"],
            y=df["Temp"],
            mode="lines",
            name="Actual",
            line=dict(
                color="#8c959f",
                width=2
            )
        ),
        go.Scatter(
            x=predictions["Date"],
            y=predictions["Mean"] + 2 * predictions["Std. Dev."],
            mode="lines",
            showlegend=False,
            line=dict(
                width=0.5, 
                color="#fb8f44",
            ),
            fillcolor="rgba(251, 143, 68, 0.33)",
            legendgroup="Test Set Predicted Mean +/- 2 Std. Dev.",
        ),
        go.Scatter(
            x=predictions["Date"],
            y=predictions["Mean"] - 2 * predictions["Std. Dev."],
            mode="lines",
            name="Test Set Predicted Mean +/- 2 Std. Dev.",
            legendgroup="Test Set Predicted Mean +/- 2 Std. Dev.",
            line=dict(
                width=0.5, 
                color="#fb8f44",
            ),
            fillcolor="rgba(251, 143, 68, 0.33)",
            fill="tonexty",
        ),
        go.Scatter(
            x=predictions["Date"],
            y=predictions["Mean"],
            mode="lines",
            name="Test Set Predicted Mean",
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


    
![png](https://demo-projects-files.s3.eu-west-1.amazonaws.com/chronos/figure_2.png)
    


Calculate the test error.


```python
print("Mean Absolute Error (MAE):", round(mean_absolute_error(y_true=df["Temp"].iloc[-prediction_length:], y_pred=predictions["Mean"]), 2))
print("Root Mean Squared Error (RMSE):", round(root_mean_squared_error(y_true=df["Temp"].iloc[-prediction_length:], y_pred=predictions["Mean"]), 2))
```

    Mean Absolute Error (MAE): 0.58
    Root Mean Squared Error (RMSE): 0.67


### Inference

Generate the forecasts.


```python
start = time.time()
transformers.set_seed(42)
yhat_future = model.predict(
    context=torch.from_numpy(df["Temp"].values),
    prediction_length=prediction_length,
    num_samples=num_samples
).detach().cpu().numpy().squeeze(axis=0)
end = time.time()
print(f"Inference time: {format(end - start, ',.2f')} seconds")
```

    Inference time: 2.38 seconds


Organize the forecasts in a data frame.


```python
forecasts = pd.DataFrame(
    data={
        "Mean": np.mean(yhat_future, axis=0),
        "Std. Dev.": np.std(yhat_future, ddof=1, axis=0),
    },
    index=pd.Series(
        data=pd.date_range(
            start=df["Date"].iloc[-1] + pd.tseries.offsets.MonthBegin(1),
            end=df["Date"].iloc[-1] + pd.tseries.offsets.MonthBegin(prediction_length),
            freq="MS"
        ),
        name="Date"
    )
).reset_index()
```


```python
forecasts.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Date</th>
      <th>Mean</th>
      <th>Std. Dev.</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1991-01-01</td>
      <td>15.517287</td>
      <td>0.856018</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1991-02-01</td>
      <td>16.110432</td>
      <td>0.817915</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1991-03-01</td>
      <td>15.519515</td>
      <td>0.860548</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1991-04-01</td>
      <td>13.323360</td>
      <td>0.913167</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1991-05-01</td>
      <td>10.240448</td>
      <td>0.903526</td>
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
        width=1100,
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
            range=[0, 25],
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
            x=df["Date"],
            y=df["Temp"],
            mode="lines",
            name="Actual",
            line=dict(
                color="#8c959f",
                width=2
            )
        ),
        go.Scatter(
            x=predictions["Date"],
            y=predictions["Mean"] + 2 * predictions["Std. Dev."],
            mode="lines",
            showlegend=False,
            line=dict(
                width=0.5, 
                color="#fb8f44",
            ),
            fillcolor="rgba(251, 143, 68, 0.33)",
            legendgroup="Test Set Predicted Mean +/- 2 Std. Dev.",
        ),
        go.Scatter(
            x=predictions["Date"],
            y=predictions["Mean"] - 2 * predictions["Std. Dev."],
            mode="lines",
            name="Test Set Predicted Mean +/- 2 Std. Dev.",
            legendgroup="Test Set Predicted Mean +/- 2 Std. Dev.",
            line=dict(
                width=0.5, 
                color="#fb8f44",
            ),
            fillcolor="rgba(251, 143, 68, 0.33)",
            fill="tonexty",
        ),
        go.Scatter(
            x=predictions["Date"],
            y=predictions["Mean"],
            mode="lines",
            name="Test Set Predicted Mean",
            line=dict(
                color="#fb8f44",
                width=4,
                dash="dot",
            )
        ),
        go.Scatter(
            x=forecasts["Date"],
            y=forecasts["Mean"] + 2 * forecasts["Std. Dev."],
            mode="lines",
            showlegend=False,
            line=dict(
                width=0.5, 
                color="#54aeff",
            ),
            fillcolor="rgba(84, 174, 255, 0.33)",
            legendgroup="Out-of-Sample Predicted Mean +/- 2 Std. Dev.",
        ),
        go.Scatter(
            x=forecasts["Date"],
            y=forecasts["Mean"] - 2 * forecasts["Std. Dev."],
            mode="lines",
            name="Out-of-Sample Predicted Mean +/- 2 Std. Dev.",
            legendgroup="Out-of-Sample Predicted Mean +/- 2 Std. Dev.",
            line=dict(
                width=0.5, 
                color="#54aeff",
            ),
            fillcolor="rgba(84, 174, 255, 0.33)",
            fill="tonexty",
        ),
        go.Scatter(
            x=forecasts["Date"],
            y=forecasts["Mean"],
            mode="lines",
            name="Out-of-Sample Predicted Mean",
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


    
![png](https://demo-projects-files.s3.eu-west-1.amazonaws.com/chronos/figure_3.png)
    

