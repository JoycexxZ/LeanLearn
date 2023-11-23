import streamlit as st
import pandas as pd
import numpy as np

from ui_components.status_bar import StatusBar


### Front End
st.title('A Simple Pruning Helper')
st.caption('_This is a simple app to help you prune your neural network._')

## Status window
status_window = StatusBar(st.container())

## Sidebar
args = {}
st.sidebar.title('Pruning Options')
with st.sidebar:
    with st.form("pruning options"):
        args['model_name'] = st.text_input('model name', 'resnet18')
        args['weight'] = st.text_input('weight', 'weights/resnet18.pt')
        args['dataset'] = st.selectbox('dataset', ['CIFAR10', 'MNIST'])
        args['pretrained'] = st.selectbox('pretrained', ['True', 'False'])
        
        with st.expander("Advanced Options"):
            args['data_dir'] = st.text_input('data directory', './data/cifar10')
            args['pruner'] = st.selectbox('pruner', [
                "level",
                "l1norm",
                "l2norm",
                "fpgm",
                "slim",
                "taylor",
                "linear",
                "agp",
                "movement",], index=2)
            args['sparse_ratio'] = st.slider('sparse ratio', 0.0, 1.0, 0.5, 0.1)
            args['batch_size'] = st.slider('batch size', 1, 256, 128, 1)
            args['train_epoch'] = st.slider('train epoch', 1, 200, 100, 10)
            args['tune_epoch'] = st.slider('tune epoch', 1, 100, 20, 10)
            args['learning_rate'] = st.slider('learning rate', 0.0, 1.0, 0.01, 0.01)
            args['weight_decay'] = st.slider('weight decay', 0.0, 1.0, 0.01, 0.01)
        
        submitted = st.form_submit_button("Run Experiment")
        if submitted:
            status_window.state_running()
            

