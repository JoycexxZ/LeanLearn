import streamlit as st

class MainPage:
    def __init__(self, container):
        self.page = container
        
    def empty(self):
        self.page.empty()
        
    def init_page(self):
        with self.page:
            self.dashboard_page, self.logs_page = st.tabs(["Dashboard", "Logs"])
        