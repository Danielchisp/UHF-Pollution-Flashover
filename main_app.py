# main.py
import dash
from callbacks import app_callbacks
from layout import app_layout

app = dash.Dash(__name__, title='UHF Pollution Flashover Monitor')
archivo_hdf5 = 'Main Databases/master.hdf5'

app_layout(app)
app_callbacks(app, archivo_hdf5)

if __name__ == '__main__':
    app.run(debug=False)
    print('http://127.0.0.1:8050')