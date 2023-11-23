import streamlit as st

class StatusBar:
    def __init__(self, status_window):
        self.status = status_window
        self.state_waiting()
        
    def state_waiting(self):
        self.status = self.status.empty()
        self.status.info("Waiting for input...")
        
    def state_running(self):
        self.status = self.status.empty()
        self.status.info("Running experiment...")
    
    def state_finished(self):
        self.status.success("Finished experiment!")
    