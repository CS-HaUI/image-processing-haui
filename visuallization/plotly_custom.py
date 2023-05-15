# from plotly.subplots import make_subplots
# from skimage import data
# img = data.chelsea()
# fig = make_subplots(1, 2)
# # We use go.Image because subplots require traces, whereas px functions return a figure
# fig.add_trace(go.Image(z=img), 1, 1)
# for channel, color in enumerate(['red', 'green', 'blue']):
#     fig.add_trace(go.Histogram(x=img[..., channel].ravel(), opacity=0.5,
#                                marker_color=color, name='%s channel' %color), 1, 2)
# fig.update_layout(height=400)
# fig.show()