import plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from PIL import Image
import base64
from io import BytesIO
import numpy as np

from visualization.subplots import Plot

class PlotlyCustom(Plot):
    
    def __init__(self, 
                 nrows  = 1,
                 ncols  = 1,
                 images = [None],
                 show_type = [None],
                 show_axis = [None],
                 show_title= [None],
                 init_fig  = False):
        super().__init__(nrows = nrows, 
                         ncols = ncols, 
                         images = images, 
                         show_type = show_type,
                         show_axis = show_axis,
                         show_title = show_title,
                         init_fig = init_fig)
        self.prefix = "data:image/png;base64,"
        self.fig = make_subplots(self.nrows, self.ncols)
        
    
    def add(self):
        for row in range(self.nrows):
            for col in range(self.ncols):
                if self.show_type[row][col] == "RGB":
                    img = Image.fromarray(np.uint8( self.images[row][col] )).convert(self.show_type[row][col])
                else:
                    img = Image.fromarray(np.uint8( self.images[row][col] ))
                base64_string = None
                with BytesIO() as stream:
                    img.save(stream, format="png")
                    base64_string = self.prefix + base64.b64encode(stream.getvalue()).decode("utf-8")
                self.fig.add_trace(go.Image(source=base64_string), row+1, col+1)
        
        
    def plot_web(self):
        self.fig.update_layout(
            height=500, width=500,
        )
        #x axis
        self.fig.update_xaxes(visible=False)

        #y axis    
        self.fig.update_yaxes(visible=False)
        self.fig.show()
        