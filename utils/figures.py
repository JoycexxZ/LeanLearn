import torch
from streamlit_elements import elements, nivo, mui
from utils.model_info_utils import simplify_number

class SummaryWriter:
    def __init__(self):
        self.tracker = {}
        self.colors = [
            "hsl(176, 70%, 50%)",
            "hsl(221, 70%, 50%)",
            "hsl(109, 70%, 50%)",
            "hsl(239, 70%, 50%)",
            "hsl(45, 70%, 50%)",
            "hsl(99, 70%, 50%)",
            "hsl(68, 70%, 50%)"
        ]
        self.class_names = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
        
    def update_endpoints(self, end_points):
        self.end_points = end_points
            
    def add_scalar(self, key, value, step):
        if key not in self.tracker:
            self.tracker[key] = []
        self.tracker[key].append((step, value.detach().cpu().item()))
        
    def add_class_acc(self, key, value):
        value = value * 100
        self.tracker[key] = value.detach().cpu().tolist()
        
    def get_data(self, key):
        if isinstance(key, str):
            key = [key]
        data = []
        
        if "Loss" in key[0] or "Accuracy" in key[0]:
            for i, k in enumerate(key):
                data_k = {"id": k, "color": self.colors[i]}
                data_k["data"] = [{"x": step, "y": value} for step, value in self.tracker[k]]
                data.append(data_k)
        elif "ClassAcc" in key[0]:
            key_name = [k.split("/")[1] for k in key]
            print(key_name)
            num_classes = len(self.tracker[key[0]])
            for class_id in range(num_classes):
                data_c = {
                    "class": self.class_names[class_id],
                }
                data_c.update({keyname: self.tracker[key[i]][class_id] for i, keyname in enumerate(key_name)})
                data_c.update({f"{keyname}Color": self.colors[i] for i, keyname in enumerate(key_name)})
                data.append(data_c)
        else:
            raise ValueError("Invalid key")
        
        # shuffle colors
        self.colors = self.colors[1:] + self.colors[:1]
        return data
    
        # print(data)
        # with elements("nivo_charts"):
        #     with mui.Box(width="100%"):
        #         nivo.Line(
        #             data=data,
        #             axisBottom=[{
        #                 "legend": "step",
        #             }],
        #             axisLeft=[{
        #                 "legend": "value",
        #             }]
        #         )
        
    def get_model_param_table(self):
        param_num_bef, param_num_aft = self.end_points['model_param_bef'], self.end_points['model_param_aft']
        columns = [
            {"field": "id", "headerName": "ID", "width": 50},
            {"field": "name", "headerName": "Layer", "width": 200},
            {"field": "bef", "headerName": "Before", "width": 100},
            {"field": "aft", "headerName": "After", "width": 100},
            {"field": "ratio", "headerName": "Ratio", "width": 80},
        ]
        rows = [{"id": i,
                "name": key, 
                "bef": simplify_number(param_num_bef[key]), 
                "aft": simplify_number(param_num_aft[key]),
                "ratio": f"{param_num_aft[key] / param_num_bef[key]:2f}"} 
                for i, key in enumerate(param_num_bef)]
        
        return columns, rows

    def get_model_spec_table(self):
        spec_bef, spec_aft = self.end_points['model_spec_bef'], self.end_points['model_spec_aft']
        columns = [
            {"field": "id", "headerName": "ID", "width": 50},
            {"field": "name", "headerName": "Layer", "width": 200},
            {"field": "bef", "headerName": "Before", "width": 150},
            {"field": "aft", "headerName": "After", "width": 150},
            {"field": "pruned", "headerName": "Pruned", "width": 100},
        ]
        rows = [{"id": i,
                "name": key, 
                "bef": spec_bef[key], 
                "aft": spec_aft[key], 
                "pruned": spec_bef[key] != spec_aft[key]} 
                for i, key in enumerate(spec_bef)]
        
        return columns, rows
    
    def get_basic_info(self):
        basic_info = [
            f"Model Out Path: ./outputs/{self.end_points['model_name']}.pt",
            f"Param: {simplify_number(self.end_points['total_param_bef'])} -> {simplify_number(self.end_points['total_param_aft'])}",
            f"Accuracy: {self.end_points['accuracy_bef']:.2f} -> {self.end_points['accuracy_aft']:.2f}",
        ]
        return basic_info