import dash
from dash import html
from dash import dcc          # 动态交互的模块
from dash.dependencies import Input,Output  # I/O输入输出控制
 
app = dash.Dash()

app.layout = html.Div(children=[
    html.H6('Change the value in the text box to see callbacks in action!'),
    html.Div([
        'input: ',
        dcc.Input(id='my-input',value='initial value',type='text')
    ]),  						#对应输入的Div，输入的里面有一个提示语和一个输入框
    html.Br(),  				#创建换行
    html.Div(id='my-output') 	#对应输出的Div，输出里面暂时为空
])


# 定义造车规则
def rule(input_value, aaa):
    return input_value + '/' + aaa +  '🍎'

# 创建一个造车机器人
data_processer = app.callback(
    Output(component_id='my-output',component_property='children'),
    [Input(component_id='my-input',component_property='value'),
     Input(component_id='my-input',component_property='value')]
)   

# 造车机器人加载规则进行造车
data_processer(rule)


if __name__ == '__main__':
    app.run_server(debug=True, host="192.168.226.130")
