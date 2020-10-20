# IceVision Deployment using FastAPI and Streamlit


## Installation
You need to install both FastAPI and Streamlit in your working environment 

### FastAPI Installation
From your the project root directory , run the following commands

```bash
cd fastapi
pip install -r requirements.txt
```

### Streamlit Installation
From your the project root directory , run the following commands

```bash
cd streamlit
pip install -r requirements.txt
```

## Running using a local machine

1- Open a terminal for the FastAPI server, and run the following command:

```bash
cd fastapi
uvicorn server:app --reload
```

2- Open another terminal for the Streamlit client, and run the following command:

```bash
cd streamlit
streamlit run ui.py
```
Press **Ctrl+Click** on the http://192.168.2.11:8501 URL to open the app in your local browser

## Credit:
This implementation is inspired by the following blog post:

[Simple example of usage of streamlit and FastAPI for ML model serving](https://davidefiocco.github.io/2020/06/27/streamlit-fastapi-ml-serving.html).

