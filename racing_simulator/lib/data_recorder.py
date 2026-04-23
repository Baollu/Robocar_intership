import pandas as pd

class DataRecorder:
    def __init__(self, columns_name):
        self.record_list = []
        self.columns_name = columns_name
    
    def record(self, data):
        self.record_list.append(data)
        
    def save(self, path):
        df = pd.DataFrame(self.record_list, columns=self.columns_name)
        df.to_csv(path)
