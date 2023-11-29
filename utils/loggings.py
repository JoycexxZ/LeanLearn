import logging
import streamlit as st


class StreamlitHandler(logging.Handler):
    def __init__(self, st_container, num_id, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.st_container = st_container
        self.num_id = num_id
    
    def emit(self, record):
        log_entry = self.format(record)
        self.st_container.write(log_entry)


class Log(object):
    def __init__(self, logger_file):
        self.logger = logging.getLogger()
        
        self.formatter = logging.Formatter(
            fmt="%(asctime)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        self.logger_file = logger_file
        self.logger.addHandler(self.get_file_handler(self.logger_file))
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
        
        self.num_streamlit_handler = 0

    def get_file_handler(self, logger_file):
        file_handler = logging.FileHandler(logger_file, encoding='utf-8')
        file_handler.setFormatter(self.formatter)
        return file_handler
    
    def start_capture(self, st_log_window, num_id=-1, fmt='log'):
        if num_id == -1:
            num_id = self.num_streamlit_handler
            self.num_streamlit_handler += 1
        streamlit_handler = StreamlitHandler(st_log_window, num_id)
        if fmt == 'log':
            streamlit_handler.setFormatter(self.formatter)
        elif fmt == 'plain':
            streamlit_handler.setFormatter(logging.Formatter(fmt='%(message)s  \n'))
        self.logger.addHandler(streamlit_handler)
        return num_id
        
    def stop_capture(self, num_id):
        for handler in self.logger.handlers:
            if isinstance(handler, StreamlitHandler) and handler.num_id == num_id:
                self.logger.removeHandler(handler)
                break
            
    def info(self, message):
        self.logger.info(message)