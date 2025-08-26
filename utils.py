import os

def create_dir(path: str):
    """Create directory if not exists"""
    if not os.path.exists(path):
        os.makedirs(path)