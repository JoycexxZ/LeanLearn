import torch
from streamlit_elements import elements, nivo, mui

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