import os
import time
import flask
import dash
import dash_uploader as du
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State

from dash import Dash, dcc, html, Input, Output
import plotly.express as px
import plotly

import pandas as pd


import fmpy
from fmpy import read_model_description, simulate_fmu, extract
from fmpy.util import create_plotly_figure

import argparse

parser = argparse.ArgumentParser(description="Run the FMPy WebApp")

# parser.add_argument('--fmu_filename', help="Filename of the FMU")
parser.add_argument('--start-values', nargs='+', help="Variables for which start values can be set")
parser.add_argument('--output-variables', nargs='+', help="Variables to plot")
parser.add_argument('--host', default='127.0.0.1', help="Host IP used to serve the application")
parser.add_argument('--port', default='8050', type=int, help="Port used to serve the application")
parser.add_argument('--debug', action='store_true', help="Set Flask debug mode and enable dev tools")

args = parser.parse_args()

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP],  suppress_callback_exceptions=True )


states = []
names  = []

fmu_filename=''

temp_folder='temp'


df=pd.DataFrame(columns=['time','Output','_value'])


def reload_df():

    global df

    try:
        tmpdf = pd.read_csv('/tmp/result.csv')

        for col in tmpdf.columns:
            if col in ['internalTime','binfilename', 'minsamplestep']:
                tmpdf.drop(col, axis=1, inplace=True)
            elif col.startswith('SGP'):
                tmpdf.drop(col, axis=1, inplace=True)
            elif col.startswith('der'):
                tmpdf.drop(col, axis=1, inplace=True)

        # tmpdf=tmpdf[0:n*100]


        newdf = pd.DataFrame(columns=['time','Output','_value'])
        for col in tmpdf.columns:
            if col=='time':
                continue
            xdf=tmpdf[['time',col]].copy()
            xdf['Output']=col
            xdf.rename(columns={col:'_value'}, inplace= True)

            newdf = newdf.append( xdf )

    except Exception as e:
        print ('Error loadfile..' + str(e))
        return df


    #df.drop(df.index, inplace=True)
    #df=df.append(newdf, ignore_index=True)

    return newdf



du.configure_upload(app, folder=temp_folder, use_upload_id=False)

colors = {
    'background': '#111111',
    'text': '#555555'
}

app.layout = html.Div(
    dbc.Container(
        [
            #html.Div('Orthogonal Realtime Simulation', id='testi'),
            html.H1(
                children='Orthogonal Realtime Simulation Enviroment',
                id='testi',
                style={
                    'textAlign': 'center',
                    'color': colors['text']
                }
            ),



            #html.Div([
            #    dcc.Dropdown(
            #        options=df['Output'].unique(),
            #        value='',
            #        id='yaxis-column'
            #    ) ], style={'width': '48%', 'display': 'none'}),
            #    #) ], style={'width': '48%', 'display': 'inline-block'}),



            html.Div(' ', id='upload_status'),




            #html.Div(
            #        children=[
            #            dcc.Interval(id='interval-component',
            #            interval=2 * 1000,
            #            n_intervals=0),
            #            ##### html.Div(id="acoolId", children=[]),
            #        ]),
            dcc.Graph(id='rtos-graph'),

            #dbc.Button('Simulate', id='simulate-button1', color='primary', className='mr-4'),

            du.Upload(id='uploader',filetypes=['fmu'] ),

            html.Pre('', id='alogpanel', style={'width':'100%'}),
        ]
    ),

)


#@app.callback(
#    Output('acoolId', 'children'),
#    [Input('interval-component', 'n_intervals')])
#def timer(n):
#    # print(asdlol) # if this line is enabled, the periodic behavior happens
#
#    global df
#
#    df = reload_df()
#
#    return [html.P("Cnt sec:" + str(n))]
#    #return [html.P("")]


#@app.callback(
#    Output('rtos-graph', 'figure'),
#    [Input('interval-component', 'n_intervals')])
#def timer2(n):
#
#    print(time.strftime('start   %Y-%m-%d %H:%M:%S %p' ), n)
#    df = reload_df(n)
#    print(time.strftime('end   %Y-%m-%d %H:%M:%S %p'), n )
#
#    # print (df.columns)
#
#    fig = plotly.tools.make_subplots(rows=1, cols=1, vertical_spacing=0.2)
#
#    fig = px.line(df, x='time', y='_value', color='Output' )
#    fig['layout']['margin'] = {
#        'l': 30, 'r': 10, 'b': 30, 't': 10
#    }
#    #fig['layout']['legend'] = {'x': 0, 'y': 1, 'xanchor': 'left'}
#
#    #fig.append_trace({
#    #    'x': df['time'],
#    #    'y': df['_value'],
#    #    ##'color': df['Output'],
#    #    'mode': 'lines+markers',
#    #    'type': 'scatter'
#    #}, 1, 1)
#
#    return fig







## 定义造车规则
#def rule(input_value):
#    print ('iiiiiiiiiiii')
#    return str(input_value) + '/' + '🍎'
## 创建一个造车机器人
#data_processer = app.callback(
#    Output(component_id='ldfsfdgpanel',component_property='children'),
#    Input(component_id='testi',component_property='value')
#)
# 造车机器人加载规则进行造车
# data_processer(rule)



def start_backend_service( fmupath_so_path ):

    import subprocess as sub

    global df

    rtserver_ip='172.16.3.13'
    rtserver_damonpath='/orthlib/server_tcp'
    rtserver_fmu_path='/tmp/fmu.fmu'

    try:
        cmd = "rm -rf /tmp/result.csv ; scp %s %s:%s" % (fmupath_so_path,rtserver_ip,rtserver_fmu_path)
        os.system( cmd )

        cmd = "ssh %s '/orthlib/start_server2.sh > /dev/null'" % (rtserver_ip)
        os.system( cmd )

        #cmd = "scp %s:%s /tmp/result.csv " % (rtserver_ip,"/tmp/result.csv")
        #os.system( cmd )

        #cmd = "ssh %s '/orthlib/fmu_load_to_tmp %s '" % (rtserver_ip, rtserver_fmu_path)
        #os.system( cmd )
        #cmd = "ssh %s 'cp -f /tmp/fmufolder/binaries/linux64/*.so /tmp/fmu.so'" % (rtserver_ip)   ## fixed path for server_tcp loading
        #os.system( cmd )

        print( 'FMU simulation done, result.csv generated.')

        #start_values = dict(zip(names, values))
        #result = simulate_fmu(filename=fmu_filename,
        #                      start_values=start_values,
        #                      stop_time=stop_time,
        #                      output=args.output_variables)
        #fig = create_plotly_figure(result=result)
        #return dcc.Graph(figure=fig)

    except Exception as e:
        return dbc.Alert("Simulation failed. %s " % (e), color='danger'),




def gen_fmu_page( unzipdir  ):

    model_description = read_model_description(unzipdir)
    has_documentation = os.path.isdir(os.path.join(unzipdir, 'documentation'))
    has_model_png = os.path.isfile(os.path.join(unzipdir, 'model.png'))
    app.title = model_description.modelName

    rows   = []

    # parameters = args.start_values
    parameters = None

    if parameters is None:
        parameters = []
        for variable in model_description.modelVariables:
            if variable.causality == 'parameter' and variable.initial != 'calculated':
                parameters.append(variable.name)


    for i, variable in enumerate(model_description.modelVariables):
    
        if variable.name not in parameters:
            continue

        unit = variable.unit

        if unit is None and variable.declaredType is not None:
            unit = variable.declaredType.unit

        names.append(variable.name)

        id = f'variable-{i}'

        row = dbc.Row(
            [
                dbc.Label(variable.name, html_for=id, width=6),
                dbc.Col(
                    dbc.InputGroup(
                        [
                            dbc.Input(id=id, value=variable.start, style={'text-align': 'right'}),
                            dbc.InputGroupText(unit if unit else " ")
                        ], size="sm"
                    ),
                    width=6,
                ),
            ],
            className='mb-2'
        )

        rows.append(row)
    
        states.append(State(id, 'value'))


    stop_time = None

    if model_description.defaultExperiment:
        stop_time = model_description.defaultExperiment.stopTime

    if stop_time is None:
        stop_time = '1'

    fmi_types = []

    if model_description.modelExchange:
        fmi_types.append('Model Exchange')

    if model_description.coSimulation:
        fmi_types.append('Co-Simulation')


    fmu_page = dbc.Container([
        dbc.Tabs(
            [
                dbc.Tab(label="Model Info", tab_id="model-info-tab"),
#                dbc.Tab(label="Simulation", tab_id="simulation-tab"),
#                dbc.Tab(label="Documentation", tab_id="documentation-tab", disabled=not has_documentation),
            ],
            className='pt-4 mb-4',
            active_tab="model-info-tab",
            id='tabs'
        ),
    
        dbc.Container(
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Row([
                                dbc.Col(html.Span("FMI Version"), width=4),
                                dbc.Col(html.Span(model_description.fmiVersion), width=8),
                            ], className='py-1'),
                            dbc.Row([
                                dbc.Col("FMI Type", width=4),
                                dbc.Col(', '.join(fmi_types), width=8),
                            ], className='py-1'),
                            dbc.Row([
                                dbc.Col(html.Span("Continuous States"), width=4),
                                dbc.Col(html.Span(model_description.numberOfContinuousStates), width=8),
                            ], className='py-1'),
                            dbc.Row([
                                dbc.Col(html.Span("Event Indicators"), width=4),
                                dbc.Col(html.Span(model_description.numberOfEventIndicators), width=8),
                            ], className='py-1'),
                            dbc.Row([
                                dbc.Col(html.Span("Variables"), width=4),
                                dbc.Col(html.Span(len(model_description.modelVariables)), width=8),
                            ], className='py-1'),
                            dbc.Row([
                                dbc.Col(html.Span("Generation Date"), width=4),
                                dbc.Col(html.Span(model_description.generationDateAndTime), width=8),
                            ], className='py-1'),
                            dbc.Row([
                                dbc.Col(html.Span("Generation Tool"), width=4),
                                dbc.Col(html.Span(model_description.generationTool), width=8),
                            ], className='py-1'),
                            dbc.Row([
                                dbc.Col(html.Span("Description"), width=4),
                                dbc.Col(html.Span(model_description.description), width=8),
                            ], className='py-1'),
                        ], width=8
                    ),
                    dbc.Col(
                        [
                            html.Img(src="/model.png", className='img-fluid')
                        ] if has_model_png else [], width=4
                    ),
                ]
            ),
            id='model-info-container',
        ),
    
        dbc.Container(
            [
                dbc.Form(
                    [
                        dbc.InputGroup(
                            [
                                dbc.Button('Simulate', id='simulate-button', color='primary', className='mr-4'),
                                dbc.Input(id="stop-time", value=stop_time, style={'text-align': 'right', 'width': '5rem'}),
                                dbc.InputGroupText("s", style={'width': '2rem'}),
                            ], className='mr-4', style={'width': '15rem'}
                        )
                    ],
                ),
                dbc.Row(
                    [
                        dbc.Col(rows, width=12, lg=4, style={'margin-top': '2rem'}),
                        dbc.Col(id='result-col', width=12, lg=8),
                    ], className='mt-4'
                ),
            ],
            id='simulation-container'
        ),
    
        dbc.Container(
            [
                html.Iframe(
                    src='/documentation/index.html',
                    style={'width': '100%', 'height': '100%'},
                )
            ],
            id='documentation-container',
            className='p-0',
        ),
    
        #html.Footer(
        #    [
        #        html.A("Orthogonal-Tech" , href='https://github.com/CATIA-Systems/FMPy', className='d-block text-muted small'),
        #    ], className='my-4 pt-3 border-top')

    
    ])


    return (fmu_page)




@app.callback(
    Output('result-col', 'children'),
    [Input('simulate-button', 'n_clicks')],
    [State('stop-time', 'value')] + states
)
def update_output_div(n_clicks, stop_time, *values):

    fmu_filename="/root/BouncingBall/BouncingBall.zip"

    try:

        start_values = dict(zip(names, values))

        result = simulate_fmu(filename=fmu_filename,
                              start_values=start_values,
                              stop_time=stop_time,
                              output=args.output_variables)
        fig = create_plotly_figure(result=result)

        return dcc.Graph(figure=fig)
    except Exception as e:
        return dbc.Alert("Simulation failed. %s path[%s]" % (e, fmu_filename), color='danger'),


@app.callback(
    [Output('model-info-container', 'style'),
     Output('simulation-container', 'style'),
     Output('documentation-container', 'style')],
    [Input("tabs", "active_tab")])
def switch_tab(active_tab):
    return (
        {'display': 'block' if active_tab == 'model-info-tab' else 'none'},
        {'display': 'block' if active_tab == 'simulation-tab' else 'none'},
        {'display': 'block' if active_tab == 'documentation-tab' else 'none', 'height': '75vh'}
    )



#@app.server.route('/model.png')
#def send_static_resource():
#    return flask.send_from_directory(os.path.join(unzipdir), 'model.png', cache_timeout=0)


#@app.server.route('/documentation/<resource>')
#def serve_documentation(resource):
#    return flask.send_from_directory(os.path.join(unzipdir, 'documentation'), resource, cache_timeout=0)



@app.callback(
    Output('upload_status', 'children'),
    Output('rtos-graph', 'figure'),
    Output(component_id='alogpanel', component_property='children'),
    Input('uploader', 'isCompleted'),
    State('uploader', 'fileNames')
)
def show_upload_status(isCompleted, fileNames):
    if isCompleted:

        fmu_filename = temp_folder+'/'+fileNames[0] 

        unzipdir = extract( fmu_filename )

        ret = gen_fmu_page( unzipdir ) 

        start_backend_service(  fmu_filename )

        df = reload_df()

        fig = plotly.tools.make_subplots(rows=1, cols=1, vertical_spacing=0.2)

        fig = px.line(df, x='time', y='_value', color='Output' )
        fig['layout']['margin'] = {
            'l': 30, 'r': 10, 'b': 30, 't': 10
        }

        #fig['layout']['legend'] = {'x': 0, 'y': 1, 'xanchor': 'left'}


        alogtext="=====   Orthogonal RTOS running log:   =====\n"
        with open("/tmp/rtai.log") as file_obj:
             for content in file_obj:
                 alogtext = alogtext + content


        return (ret , fig,  alogtext )


    return dash.no_update




#@app.callback(
#    Output('rtos-graph', 'figure'),
#    Input('testi', 'children'))
#def update_graph(colname):
#
#    # dff=df[df['Output']==colname]
#    #if len(df)<=0:
#    #    return None
#
#    dff = df
#
#    fig = px.line(dff, x='time', y='_value', color='Output' )
#
#    #fig.update_layout(margin={'l': 40, 'b': 40, 't': 10, 'r': 0}, hovermode='closest')
#    #fig.update_xaxes(title=xaxis_column_name,
#    #                 type='linear' if xaxis_type == 'Linear' else 'log')
#    #fig.update_yaxes(title=yaxis_column_name,
#    #                 type='linear' if yaxis_type == 'Linear' else 'log')
#
##    return fig




#####################################################################

if __name__ == '__main__':
    app.run_server(host="0.0.0.0", debug=True, port=8050)



