from streamlit_elements import elements, mui, dashboard, nivo
# from utils.figures import get_model_param_table, get_model_spec_table

# class DashBoard:
#     def __init__(self, container, num_parts=4) -> None:
#         with container:
#             with elements("dashboard"):
#                 self.layout = [
#                     dashboard.Item(f"item_{i}", (i%2)*2, (i//2)*2, 2, 2) for i in range(num_parts)
#                 ]
#                 self.window = []
#                 with dashboard.Grid(self.layout):
#                     for i in range(num_parts):
#                         self.window.append(mui.Paper(key=f"item_{i}"))
        
#     def enter(self, part):
#         assert 0 <= part < len(self.window) 
#         return self.window[part]

def create_dashboard(container, summary_writer, figure_names):
    num_parts = len(figure_names)
    
    with container:
        with elements("dashboard"):
            layout = [
                dashboard.Item(f"item_{i}", (i%2)*2, (i//2)*2, 2, 2) for i in range(num_parts)
            ]
            # layout = [
            #     dashboard.Item(f"item_{i}", 0, i*2, 4, 2) for i in range(num_parts)
            # ]
            i = 0
            with dashboard.Grid(layout):
                if "Basic Info" in figure_names:
                    with mui.Paper(key=f"item_{i}"):
                        mui.Typography("Basic Info")
                        for item in summary_writer.get_basic_info():
                            mui.Typography(item)
                    i += 1
                
                if "Tune Loss" in figure_names:
                    with mui.Paper(key=f"item_{i}"):
                        data = summary_writer.get_data("Loss/tune")
                        with elements("nivo_charts"):
                            mui.Typography("Tune Loss")
                            with mui.Box(sx={"height":300}):
                               nivo.Line(
                                    data=data,
                                    margin={ "top": 30, "right": 30, "bottom": 50, "left": 60 },
                                    # borderColor={ 'from': 'color' },
                                    # colors={ 'scheme': 'nivo' },
                                    xScale={ 'type': 'linear' },
                                    yScale={
                                        'type': 'linear',
                                        'min': 'auto',
                                        'max': 'auto',
                                        # 'stacked': 'true',
                                        # 'reverse': 'false'
                                    },
                                    axisBottom={
                                        'tickSize': 5,
                                        'tickPadding': 5,
                                        'tickRotation': 0,
                                        'legend': 'Epoch',
                                        'legendOffset': 36,
                                        'legendPosition': 'middle'
                                    },
                                    axisLeft={
                                        'tickSize': 5,
                                        'tickPadding': 5,
                                        'tickRotation': 0,
                                        'legend': 'Loss',
                                        'legendOffset': -40,
                                        'legendPosition': 'middle'
                                    },
                                    # pointSize=10,
                                    # pointColor={ 'theme': 'background' },
                                    # pointBorderWidth={2},
                                    # pointBorderColor={ 'from': 'serieColor' },
                                    # pointLabelYOffset=-12,
                                    useMesh='true',
                                    theme={
                                        "background": "#FFFFFF",
                                        "textColor": "#31333F"
                                    }
                                )
                    i += 1
                    
                if "Tune Accuracy" in figure_names:
                    with mui.Paper(key=f"item_{i}"):
                        data = summary_writer.get_data("Accuracy/tune")
                        with elements("nivo_charts"):
                            with mui.Box(sx={"height":300}):
                                mui.Typography("Tune Accuracy")
                                nivo.Line(
                                    data=data,
                                    margin={ "top": 30, "right": 30, "bottom": 50, "left": 60 },
                                    # borderColor={ 'from': 'color' },
                                    # colors={ 'scheme': 'nivo' },
                                    xScale={ 'type': 'linear' },
                                    yScale={
                                        'type': 'linear',
                                        'min': 'auto',
                                        'max': 'auto',
                                        # 'stacked': 'true',
                                        # 'reverse': 'false'
                                    },
                                    axisBottom={
                                        'tickSize': 5,
                                        'tickPadding': 5,
                                        'tickRotation': 0,
                                        'legend': 'Epoch',
                                        'legendOffset': 36,
                                        'legendPosition': 'middle'
                                    },
                                    axisLeft={
                                        'tickSize': 5,
                                        'tickPadding': 5,
                                        'tickRotation': 0,
                                        'legend': 'Acc',
                                        'legendOffset': -40,
                                        'legendPosition': 'middle'
                                    },
                                    # pointSize=10,
                                    # pointColor={ 'theme': 'background' },
                                    # pointBorderWidth={2},
                                    # pointBorderColor={ 'from': 'serieColor' },
                                    # pointLabelYOffset=-12,
                                    useMesh='true',
                                    theme={
                                        "background": "#FFFFFF",
                                        "textColor": "#31333F"
                                    }
                                )
                    i += 1
                    
                if "Test Classwise Accuracy" in figure_names or "Train Test Classwise Accuracy" in figure_names:
                    if "Train Test Classwise Accuracy" in figure_names:
                        data = summary_writer.get_data(["ClassAcc/tune", "ClassAcc/test"])
                    else:
                        data = summary_writer.get_data("ClassAcc/test")
                    print(data)
                    with mui.Paper(key=f"item_{i}"):
                        with elements("nivo_charts"):
                            with mui.Box(sx={"height": 280}):
                                mui.Typography("Test Classwise Accuracy")
                                nivo.Bar(
                                    data=data,
                                    keys=[ "tune", "test" ] if "Train Test Classwise Accuracy" in figure_names else [ "test" ],
                                    indexBy="class",
                                    margin={ "top": 30, "right": 30, "bottom": 60, "left": 60 },
                                    padding=0.3,
                                    groupMode="grouped",
                                    valueScale={ 'type': 'linear' },
                                    colors={ "scheme": "nivo" },
                                    # defs=[
                                    #     {
                                    #         "id": "dots",
                                    #         "type": "patternDots",
                                    #         "background": "inherit",
                                    #         "color": "#38bcb2",
                                    #         "size": 4,
                                    #         "padding": 1,
                                    #         "stagger": True
                                    #     },
                                    #     {
                                    #         "id": "lines",
                                    #         "type": "patternLines",
                                    #         "background": "inherit",
                                    #         "color": "#eed312",
                                    #         "rotation": -45,
                                    #         "lineWidth": 6,
                                    #         "spacing": 10
                                    #     }
                                    # ],
                                    # fill={[
                                    #     {
                                    #         "match": {
                                    #             "id": "fries"
                                    #         },
                                    #         "id": "dots"
                                    #     },
                                    #     {
                                    #         "match": {
                                    #             "id": "sandwich"
                                    #         },
                                    #         "id": "lines"
                                    #     }
                                    # ]},
                                    # borderColor={ "from": "color", "modifiers": [ [ "darker", 1.6 ] ] },
                                    # axisTop={ "tickSize": 0, "tickPadding": 12 },
                                    # axisRight=None,
                                    axisBottom={
                                        "tickSize": 5,
                                        "tickPadding": 5,
                                        "tickRotation": 0,
                                        "legend": "class",
                                        "legendPosition": "middle",
                                        "legendOffset": 32
                                    },
                                    axisLeft={
                                        "tickSize": 5,
                                        "tickPadding": 5,
                                        "tickRotation": 0,
                                        "legend": "accuracy",
                                        "legendPosition": "middle",
                                        "legendOffset": -40
                                    },
                                )
                    i += 1
                    
                if "model_param" in figure_names:
                    col, row = summary_writer.get_model_param_table()
                    with mui.Paper(key=f"item_{i}"):
                        mui.DataGrid(
                            columns=col,
                            rows=row,
                            # pageSize=5,
                            # rowsPerPageOptions=[5],
                            # checkboxSelection=True,
                            disableSelectionOnClick=True,
                        )
                    i += 1
                    
                if "model_spec" in figure_names:
                    col, row = summary_writer.get_model_spec_table()
                    with mui.Paper(key=f"item_{i}"):
                        mui.DataGrid(
                            columns=col,
                            rows=row,
                            # pageSize=5,
                            # rowsPerPageOptions=[5],
                            # checkboxSelection=True,
                            disableSelectionOnClick=True,
                        )
                    i += 1
                    
                if "test" in figure_names:
                    with mui.Paper(key=f"item_{i}"):
                        data = [
                            { "taste": "fruity", "chardonay": 93.2, "carmenere": 61, "syrah": 114 },
                            { "taste": "bitter", "chardonay": 91.7, "carmenere": 37, "syrah": 72 },
                            { "taste": "heavy", "chardonay": 56, "carmenere": 95, "syrah": 99 },
                            { "taste": "strong", "chardonay": 64, "carmenere": 90, "syrah": 30 },
                            { "taste": "sunny", "chardonay": 119, "carmenere": 94, "syrah": 103 },
                        ]
                        with elements("nivo_charts"):
                            with mui.Box(sx={"height": 200}):
                                nivo.Radar(
                                    data=data,
                                    keys=[ "chardonay", "carmenere", "syrah" ],
                                    indexBy="taste",
                                    valueFormat=">-.2f",
                                    margin={ "top": 70, "right": 80, "bottom": 40, "left": 80 },
                                    borderColor={ "from": "color" },
                                    gridLabelOffset=36,
                                    dotSize=10,
                                    dotColor={ "theme": "background" },
                                    dotBorderWidth=2,
                                    motionConfig="wobbly",
                                    legends=[
                                        {
                                            "anchor": "top-left",
                                            "direction": "column",
                                            "translateX": -50,
                                            "translateY": -40,
                                            "itemWidth": 80,
                                            "itemHeight": 20,
                                            "itemTextColor": "#999",
                                            "symbolSize": 12,
                                            "symbolShape": "circle",
                                            "effects": [
                                                {
                                                    "on": "hover",
                                                    "style": {
                                                        "itemTextColor": "#000"
                                                    }
                                                }
                                            ]
                                        }
                                    ],
                                    theme={
                                        "background": "#FFFFFF",
                                        "textColor": "#31333F",
                                        "tooltip": {
                                            "container": {
                                                "background": "#FFFFFF",
                                                "color": "#31333F",
                                            }
                                        }
                                    }
                                )
                    i += 1
                    
                if "test_bar" in figure_names:
                    with mui.Paper(key=f"item_{i}"):
                        data = [
                            { "country": "AD", "hot dog": 184.2, "hot dogColor": "hsl(322, 70%, 50%)", "burger": 73, "burgerColor": "hsl(288, 70%, 50%)", "sandwich": 177, "sandwichColor": "hsl(59, 70%, 50%)", "kebab": 121, "kebabColor": "hsl(58, 70%, 50%)", "fries": 29, "friesColor": "hsl(189, 70%, 50%)", "donut": 134, "donutColor": "hsl(109, 70%, 50%)" },
                            { "country": "AE", "hot dog": 54.3, "hot dogColor": "hsl(143, 70%, 50%)", "burger": 7, "burgerColor": "hsl(207, 70%, 50%)", "sandwich": 136, "sandwichColor": "hsl(97, 70%, 50%)", "kebab": 100, "kebabColor": "hsl(223, 70%, 50%)", "fries": 179, "friesColor": "hsl(119, 70%, 50%)", "donut": 94, "donutColor": "hsl(11, 70%, 50%)" },
                            { "country": "AF", "hot dog": 79, "hot dogColor": "hsl(169, 70%, 50%)", "burger": 34, "burgerColor": "hsl(276, 70%, 50%)", "sandwich": 177, "sandwichColor": "hsl(298, 70%, 50%)", "kebab": 36, "kebabColor": "hsl(209, 70%, 50%)", "fries": 165, "friesColor": "hsl(203, 70%, 50%)", "donut": 66, "donutColor": "hsl(44, 70%, 50%)" },
                        ]
                        with mui.Paper(key=f"item_{i}"):
                            with elements("nivo_charts"):
                                with mui.Box(sx={"height": 280}):
                                    mui.Typography("Test Classwise Accuracy")
                                    nivo.Bar(
                                        data=data,
                                        keys=["hot dog", "burger", "sandwich", "kebab", "fries", "donut"],
                                        indexBy="country",
                                        margin={ "top": 30, "right": 30, "bottom": 50, "left": 60 },
                                        padding=0.3,
                                        groupMode="grouped",
                                        valueScale={ 'type': 'linear' },
                                        colors={ "scheme": "nivo" },
                                        # defs=[
                                        #     {
                                        #         "id": "dots",
                                        #         "type": "patternDots",
                                        #         "background": "inherit",
                                        #         "color": "#38bcb2",
                                        #         "size": 4,
                                        #         "padding": 1,
                                        #         "stagger": True
                                        #     },
                                        #     {
                                        #         "id": "lines",
                                        #         "type": "patternLines",
                                        #         "background": "inherit",
                                        #         "color": "#eed312",
                                        #         "rotation": -45,
                                        #         "lineWidth": 6,
                                        #         "spacing": 10
                                        #     }
                                        # ],
                                        # fill={[
                                        #     {
                                        #         "match": {
                                        #             "id": "fries"
                                        #         },
                                        #         "id": "dots"
                                        #     },
                                        #     {
                                        #         "match": {
                                        #             "id": "sandwich"
                                        #         },
                                        #         "id": "lines"
                                        #     }
                                        # ]},
                                        # borderColor={ "from": "color", "modifiers": [ [ "darker", 1.6 ] ] },
                                        # axisTop={ "tickSize": 0, "tickPadding": 12 },
                                        # axisRight=None,
                                        axisBottom={
                                            "tickSize": 5,
                                            "tickPadding": 5,
                                            "tickRotation": 0,
                                            "legend": "class",
                                            "legendPosition": "middle",
                                            "legendOffset": 32
                                        },
                                        axisLeft={
                                            "tickSize": 5,
                                            "tickPadding": 5,
                                            "tickRotation": 0,
                                            "legend": "accuracy",
                                            "legendPosition": "middle",
                                            "legendOffset": -40
                                        },
                                    )