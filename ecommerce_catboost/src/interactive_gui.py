import ipywidgets as widgets
from IPython.display import display

# Define the widget inputs
recency_input = widgets.IntSlider(min=0, max=365, step=1, description='Recency')
frequency_input = widgets.IntSlider(min=0, max=100, step=1, description='Frequency')
monetary_input = widgets.FloatSlider(min=0, max=10000, step=0.01, description='Monetary')

# Define the output area
output = widgets.Output()

# Define the function to make predictions
def make_prediction(recency, frequency, monetary):
    data = pd.DataFrame({'Recency': [recency], 'Frequency': [frequency], 'Monetary': [monetary]})
    prediction = model.predict(data)[0]
    return "Recommend" if prediction == 1 else "Do not recommend"

# Define the update function
def update_prediction(change):
    with output:
        output.clear_output()
        prediction = make_prediction(recency_input.value, frequency_input.value, monetary_input.value)
        print(f'Recommendation: {prediction}')

# Attach the update function to the widget inputs
recency_input.observe(update_prediction, names='value')
frequency_input.observe(update_prediction, names='value')
monetary_input.observe(update_prediction, names='value')

# Display the widgets and output
display(recency_input, frequency_input, monetary_input, output)
