import streamlit as st
import pandas as pd
import numpy as np
from omegaconf import OmegaConf

from utils.engine import Engine
from utils.figures import SummaryWriter
from ui_components.status_bar import StatusBar
from ui_components.main_container import MainPage
from ui_components.dashboard import create_dashboard


SESSION_STATE_FOLDER = './checkpoints/session_state'

def run_experiment(args):
    summary_writer = SummaryWriter()
    engine = Engine(args, summary_writer)
    engine.end_points['model_name'] = args.model_name
    # Train if not pretrained
    if args.pretrained != 'True':
        engine.train(mode='train', progress_bar_sec=status_window.status)
    # Test before pruning
    test_log_window = main_page.logs_page.expander("Test before pruning")    
    engine.test(progress_bar_sec=status_window.status, st_log_window=test_log_window)
    # Prune the model
    prune_log_window = main_page.logs_page.expander("Pruning")
    prune_log_window_1, prune_log_window_2 = prune_log_window.columns(2)
    # prune_param_window = main_page.logs_page.expander("Model Parameters Info")
    # prune_spec_window = main_page.logs_page.expander("Model Specification Info")
    engine.prune(prune_log_window_1, prune_log_window_2)
    # Finetune the model
    finetune_log_window = main_page.logs_page.expander("Finetuning logs")
    engine.train(mode='tune', progress_bar_sec=status_window.status, st_log_window=finetune_log_window)
    # Test after pruning
    test_log_window = main_page.logs_page.expander("Test after pruning")
    engine.test(progress_bar_sec=status_window.status, st_log_window=test_log_window)
    # Update UI
    summary_writer.update_endpoints(engine.end_points)
    status_window.state_finished()
    figure_names = [
        "Basic Info",
        "Tune Loss", 
        "Tune Accuracy",
        # "Test Classwise Accuracy",
        "Train Test Classwise Accuracy",
        # "test",
        # "test_bar"
        "model_param",
        "model_spec",
    ]
    create_dashboard(main_page.dashboard_page, summary_writer, figure_names)
    
def dump_session_state():
    json_file_name = time.strftime("%Y%m%d-%H%M%S") + ".json"
    json_file_path = os.path.join(SESSION_STATE_FOLDER, json_file_name)
    with open(json_file_path, 'w') as f:
        json.dump(session_state, f)

def find_session_state():
    json_files = os.listdir(SESSION_STATE_FOLDER)
    for json_file in json_files:
        if json_file.endswith(".json"):
            json_file_path = os.path.join(SESSION_STATE_FOLDER, json_file)
            with open(json_file_path, 'r') as f:
                state = json.load(f)
                if state == session_state:
                    load_session_state(json_file_path)
                    
def load_session_state():
    pass

# Front End
st.title('A Simple Pruning Helper')
st.caption('_This is a simple app to help you prune your neural network._')

## Status window
status_window = StatusBar(st.container())

## Main page
main_container = st.container()
main_page = MainPage(main_container)

## Sidebar
args = {}
st.sidebar.title('Pruning Options')
with st.sidebar:
    with st.form("pruning options"):
        args['model_name'] = st.text_input('model name', 'resnet18', key='model_name')
        args['weight'] = st.text_input('weight', 'weights/resnet18.pt', key='weight')
        args['dataset'] = st.selectbox('dataset', ['CIFAR10', 'MNIST'], key='dataset')
        args['pretrained'] = st.selectbox('pretrained', ['True', 'False'], key='pretrained')
        
        with st.expander("Advanced Options"):
            args['data_dir'] = st.text_input('data directory', './data/cifar10', key='data_dir')
            args['pruner'] = st.selectbox('pruner', [
                "level",
                "l1norm",
                "l2norm",
                "fpgm",
                "slim",
                "taylor",
                "linear",
                "agp",
                "movement",], index=2, key='pruner')
            args['sparse_ratio'] = st.slider('sparse ratio', 0.0, 1.0, 0.5, 0.1, key='sparse_ratio')
            args['batch_size'] = st.slider('batch size', 1, 256, 128, 1, key='batch_size')
            args['train_epoch'] = st.slider('train epoch', 1, 200, 100, 1, key='train_epoch')
            args['tune_epoch'] = st.slider('tune epoch', 1, 100, 20, 1, key='tune_epoch')
            args['optimizer'] = st.selectbox('optimizer', ['sgd', 'adam'], key='optimizer')
            args['learning_rate'] = st.number_input('learning rate', value=1e-2, step=1e-4)
            args['weight_decay'] = st.number_input('weight decay', value=0.01, step=1e-4, key='weight_decay')
            args['seed'] = st.number_input('seed', value=3407, step=1, key='seed')
            args['num_workers'] = st.number_input('num workers', value=8, step=1, key='num_workers')
            args = OmegaConf.create(args)
        
        submitted = st.form_submit_button("Run Experiment")
        if submitted:
            status_window.state_running()
            # st.write(args.keys())
            main_page.init_page()
            run_experiment(args)

