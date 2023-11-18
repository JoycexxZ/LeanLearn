import streamlit as st
import pandas as pd
import numpy as np

st.title('A Simple Pruning Helper')
st.write('This is a simple web app to help you prune your neural network.')

st.sidebar.title('Pruning Options')
model_name = st.sidebar.text_input('model name', 'LeNet')
weight = st.sidebar.text_input('weight', 'weights/lenet.pt')
dataset = st.sidebar.selectbox('dataset', ['MNIST', 'CIFAR10'])
pruner = st.sidebar.selectbox('pruner', ['l1_norm', 'l2_norm'])
pruning_ratio = st.sidebar.slider('pruning ratio', 0.0, 1.0, 0.5, 0.1)

st.sidebar.title('Training Configs')
