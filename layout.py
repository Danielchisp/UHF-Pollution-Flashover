from dash import dcc, html
import dash_daq as daq

C = {
    'bg':       '#0d0f14',
    'panel':    '#13161e',
    'border':   '#1e2330',
    'border2':  '#2a3045',
    'cyan':     '#00d4ff',
    'cyan_dim': '#0099bb',
    'green':    '#00ff9d',
    'red':      '#ff3b5c',
    'text':     '#c8d0e0',
    'text_dim': '#5a6480',
    'label':    '#8a94b0',
}
FONT_MONO = "'JetBrains Mono', 'Fira Code', monospace"
FONT_UI   = "'DM Sans', 'Segoe UI', sans-serif"


def badge(text, color=None):
    color = color or C['cyan']
    return html.Span(text, style={
        'fontFamily': FONT_MONO, 'fontSize': '10px',
        'color': color, 'border': f'1px solid {color}',
        'borderRadius': '3px', 'padding': '1px 6px',
        'letterSpacing': '0.08em', 'opacity': '0.8',
    })


def section_label(text):
    return html.Div(text, style={
        'fontFamily': FONT_MONO, 'fontSize': '9px',
        'color': C['text_dim'], 'letterSpacing': '0.15em',
        'textTransform': 'uppercase', 'marginBottom': '5px',
    })


def control_block(children, label=None, extra_style=None):
    style = {'marginRight': '14px', 'marginBottom': '8px'}
    if extra_style:
        style.update(extra_style)
    content = ([section_label(label)] if label else [])
    content += children if isinstance(children, list) else [children]
    return html.Div(content, style=style)


def styled_dropdown(id_, options, value, multi=False, clearable=False,
                    placeholder=None, width='200px'):
    return dcc.Dropdown(
        id=id_, options=options, value=value,
        multi=multi, clearable=clearable, placeholder=placeholder,
        style={'width': width, 'fontFamily': FONT_MONO, 'fontSize': '11px'},
    )


def action_button(id_, label, color=None):
    color = color or C['cyan']
    return html.Button(label, id=id_, style={
        'background': 'transparent',
        'border': f'1px solid {color}',
        'color': color,
        'fontFamily': FONT_MONO,
        'fontSize': '10px',
        'letterSpacing': '0.1em',
        'padding': '5px 12px',
        'cursor': 'pointer',
        'borderRadius': '3px',
        'textTransform': 'uppercase',
        'marginTop': '2px',
    })


def vdivider():
    return html.Div(style={
        'width': '1px', 'background': C['border2'],
        'margin': '0 10px 8px 0', 'alignSelf': 'stretch',
    })


def graph_panel(graph_id, title, subtitle=None, flex=None, margin_right=False, height=460):
    topbar_children = [
        html.Div([
            html.Span(title, style={
                'fontFamily': FONT_MONO, 'fontSize': '9px',
                'color': C['text_dim'], 'letterSpacing': '0.15em',
            }),
            *([html.Span(f' — {subtitle}', style={
                'fontFamily': FONT_MONO, 'fontSize': '9px',
                'color': C['text_dim'], 'opacity': '0.5',
            })] if subtitle else []),
        ]),
        badge(graph_id),
    ]
    panel_style = {
        'overflow': 'hidden', 'background': C['panel'],
        'border': f'1px solid {C["border"]}', 'borderRadius': '4px',
    }
    if flex:
        panel_style['flex'] = flex
    if margin_right:
        panel_style['marginRight'] = '8px'

    return html.Div([
        html.Div(topbar_children, className='graph-topbar'),
        dcc.Graph(id=graph_id, style={'height': f'{height}px'},
                  config={'displayModeBar': True, 'scrollZoom': True}),
    ], style=panel_style)


def make_header():
    return html.Div([
        html.Div([
            html.Img(src='assets/logo.png', style={
                'height': '40px', 'marginRight': '14px', 'opacity': '0.9',
            }),
            html.Div([
                html.Div([
                    html.Span('UHF POLLUTION FLASHOVER', style={
                        'fontFamily': FONT_MONO, 'fontSize': '13px',
                        'fontWeight': '500', 'color': C['text'],
                        'letterSpacing': '0.12em',
                    }),
                    html.Span(' MONITOR', style={
                        'fontFamily': FONT_MONO, 'fontSize': '13px',
                        'fontWeight': '300', 'color': C['cyan'],
                        'letterSpacing': '0.12em',
                    }),
                ]),
                html.Div([
                    badge('DASHBOARD'),
                    html.Span('  '),
                    badge('v0.1', C['text_dim']),
                ], style={'marginTop': '4px'}),
            ]),
        ], style={'display': 'flex', 'alignItems': 'center'}),

        html.Div([
            daq.LEDDisplay(
                id='led-duration',
                value='00:00:00', color=C['cyan'],
                backgroundColor=C['panel'], size=18,
            ),
        ], style={'display': 'flex', 'alignItems': 'center'}),

        html.Div([
            html.Div([
                html.Span('●', className='blink', style={
                    'color': C['green'], 'fontSize': '10px', 'marginRight': '5px',
                }),
                html.Span('LIVE', style={
                    'fontFamily': FONT_MONO, 'fontSize': '10px',
                    'color': C['green'], 'letterSpacing': '0.1em',
                }),
            ], style={'display': 'flex', 'alignItems': 'center', 'marginBottom': '4px'}),
            html.Div('SYS OK', style={
                'fontFamily': FONT_MONO, 'fontSize': '9px',
                'color': C['text_dim'], 'letterSpacing': '0.1em',
            }),
        ], style={'textAlign': 'right'}),

    ], style={
        'display': 'flex', 'justifyContent': 'space-between',
        'alignItems': 'center', 'padding': '12px 20px',
        'background': C['panel'],
        'borderBottom': f'1px solid {C["border2"]}',
        'position': 'sticky', 'top': '0', 'zIndex': '100',
    })


def make_explore_tab(metric_opts):
    return html.Div([
        html.Div([
            # ── Selector de grupo ────────────────────────────────────────
            control_block(
                styled_dropdown('dropdown-group', [], None,
                                placeholder='Select measurement group…', width='280px'),
                label='Group',
            ),
            # ── Atributos del grupo ──────────────────────────────────────
            control_block(
                html.Div(
                    id='group-rated-voltage',
                    children='—',
                    style={
                        'fontFamily': FONT_MONO,
                        'fontSize': '13px',
                        'color': C['cyan'],
                        'letterSpacing': '0.05em',
                        'fontWeight': '600',
                        'minWidth': '60px',
                    },
                ),
                label='Rated Voltage (kV)',
            ),
            control_block(
                html.Div(
                    id='group-flashover-status',
                    children='—',
                    style={
                        'fontFamily': FONT_MONO,
                        'fontSize': '13px',
                        'color': C['text_dim'],
                        'letterSpacing': '0.05em',
                        'fontWeight': '600',
                        'minWidth': '60px',
                    },
                ),
                label='Flashover',
            ),
            # ── Separador ───────────────────────────────────────────────
            vdivider(),
            control_block(
                styled_dropdown('dropdown-y-axis', metric_opts, ['vpp'],
                                multi=True, width='300px'),
                label='Time series  Y',
            ),
            control_block(
                dcc.RadioItems(
                    id='x-axis-mode',
                    options=[{'label': 'Timestamp', 'value': 'timestamp'},
                             {'label': 'Index',     'value': 'index'}],
                    value='timestamp', inline=True,
                ),
                label='X axis mode',
            ),
            control_block(
                dcc.Checklist(
                    id='normalize-toggle',
                    options=[{'label': 'Normalize', 'value': 'normalize'}],
                    value=[], inline=True,
                ),
                label='Options',
            ),
            control_block(
                dcc.Input(id='atbs-window-input', type='number',
                          value=50, min=2, max=1000, step=1,
                          style={'width': '70px'}),
                label='ATBS window',
            ),
            vdivider(),
            control_block(
                styled_dropdown('dropdown-x-axis', metric_opts, 'eqTime', width='130px'),
                label='Scatter  X',
            ),
            control_block(
                styled_dropdown('dropdown-y-axis-scatter', metric_opts, 'eqFreq', width='130px'),
                label='Scatter  Y',
            ),
            control_block(
                styled_dropdown('dropdown-z-axis', metric_opts, None,
                                clearable=True, placeholder='None (2D)', width='130px'),
                label='Scatter  Z',
            ),
            vdivider(),
            # ── Controles de animación ───────────────────────────────────
            control_block(
                dcc.Checklist(
                    id='animate-mode',
                    options=[{'label': ' Animate', 'value': 'on'}],
                    value=[],
                    inline=True,
                    inputStyle={'marginRight': '5px', 'accentColor': C['cyan']},
                    style={
                        'fontFamily': FONT_MONO, 'fontSize': '11px',
                        'color': C['text'], 'cursor': 'pointer',
                    },
                ),
                label='Scatter anim',
            ),
            control_block(
                dcc.Input(
                    id='animate-step-input', type='number',
                    value=0, min=0, step=1,
                    placeholder='0 = auto',
                    style={
                        'width': '80px', 'fontFamily': FONT_MONO,
                        'fontSize': '11px',
                    },
                ),
                label='Step (0=auto)',
            ),
            control_block(
                dcc.Input(
                    id='animate-speed-input', type='number',
                    value=40, min=5, step=5,
                    placeholder='ms',
                    style={
                        'width': '65px', 'fontFamily': FONT_MONO,
                        'fontSize': '11px',
                    },
                ),
                label='Speed (ms)',
            ),
            # ── fin animación ────────────────────────────────────────────
            vdivider(),
            control_block([
                action_button('btn-download-scatter', '↓ CSV'),
                dcc.Download(id='download-scatter-csv'),
                html.Div(style={'height': '4px'}),
                action_button('btn-download-signals', '↓ Signals', C['green']),
                dcc.Download(id='download-signals-csv'),
            ], label='Export'),
            vdivider(),
            control_block([
                dcc.Input(id='input-selection-name', type='text',
                          placeholder='Selection name…',
                          style={'width': '140px', 'fontFamily': FONT_MONO,
                                 'fontSize': '11px', 'marginBottom': '4px'}),
                action_button('btn-insert-selection', '+ Insert', C['green']),
            ], label='Save selection'),
        ], style={
            'display': 'flex', 'alignItems': 'flex-end',
            'flexWrap': 'wrap',
            'padding': '10px 20px 2px 20px',
            'background': f'linear-gradient(180deg, {C["panel"]} 0%, {C["bg"]} 100%)',
            'borderBottom': f'1px solid {C["border"]}',
        }),

        html.Div([
            html.Div([
                graph_panel('graph-3', 'TIME SERIES', flex='7', margin_right=True, height=460),
                graph_panel('graph-4', 'SCATTER', flex='3', height=460),
            ], style={'display': 'flex', 'marginBottom': '8px'}),
            html.Div([
                graph_panel('graph-1', 'TEMPORAL SIGNAL', 'click a point',
                            flex='1', margin_right=True, height=340),
                graph_panel('graph-2', 'FFT', 'click a point',
                            flex='1', height=340),
            ], style={'display': 'flex'}),
        ], style={
            'padding': '12px 20px 20px 20px',
            'background': C['bg'],
            'minHeight': 'calc(100vh - 200px)',
        }),
    ])


def make_compare_tab(metric_opts):
    return html.Div([
        html.Div([
            control_block(
                styled_dropdown('dropdown-compare-x', metric_opts, 'eqTime', width='150px'),
                label='Scatter X',
            ),
            control_block(
                styled_dropdown('dropdown-compare-y', metric_opts, 'eqFreq', width='150px'),
                label='Scatter Y',
            ),
            vdivider(),
            control_block(
                styled_dropdown('dropdown-compare-ts-metric', metric_opts, 'vpp', width='150px'),
                label='Time series metric',
            ),
            vdivider(),
            control_block(
                dcc.Input(id='input-random-signals', type='number',
                          value=10, min=1, max=100, step=1,
                          style={'width': '70px'}),
                label='Random signals',
            ),
        ], style={
            'display': 'flex', 'alignItems': 'flex-end',
            'flexWrap': 'wrap',
            'padding': '10px 20px 2px 20px',
            'background': f'linear-gradient(180deg, {C["panel"]} 0%, {C["bg"]} 100%)',
            'borderBottom': f'1px solid {C["border"]}',
        }),

        html.Div([
            section_label('Saved selections'),
            html.Div(id='selections-list-container'),
        ], style={
            'padding': '12px 20px 0 20px',
            'background': C['bg'],
        }),

        html.Div([
            html.Div([
                graph_panel('graph-compare-scatter', 'SCATTER COMPARE',
                            flex='1', margin_right=True, height=380),
                graph_panel('graph-compare-timeseries', 'TIME SERIES COMPARE',
                            flex='1', height=380),
            ], style={'display': 'flex', 'marginBottom': '8px'}),

            html.Div([
                graph_panel('graph-compare-avg-fft', 'AVERAGE FFT',
                            flex='1', height=380),
            ], style={'display': 'flex', 'marginBottom': '8px'}),

            html.Div([
                graph_panel('graph-compare-signals', 'RANDOM SIGNALS',
                            flex='1', margin_right=True, height=380),
                graph_panel('graph-compare-fft', 'RANDOM FFT',
                            flex='1', height=380),
            ], style={'display': 'flex'}),

        ], style={
            'padding': '12px 20px 20px 20px',
            'background': C['bg'],
            'minHeight': 'calc(100vh - 200px)',
        }),
    ])


def app_layout(app):
    metric_labels = {
        'B0': 'B0  0–100 MHz',
        'B1': 'B1  100–600 MHz',
        'energy': 'Energy', 'eqFreq': 'Eq. Frequency',
        'eqTime': 'Eq. Time', 'vpp': 'Vpp',
        'timestamp': 'Timestamp', 'ATBS': 'ATBS',
        'kurtosis': 'Kurtosis',
        'skewness': 'Skewness',
        'crest_factor': 'crest_factor'
    }
    available_metrics = list(metric_labels.keys())
    metric_opts = [{'label': metric_labels[m], 'value': m} for m in available_metrics]

    app.layout = html.Div([
        dcc.Store(id='store-selections', data=[]),
        make_header(),
        dcc.Tabs(id='main-tabs', value='tab-explore', children=[
            dcc.Tab(label='EXPLORE', value='tab-explore', style={
                'fontFamily': FONT_MONO, 'fontSize': '11px',
                'letterSpacing': '0.1em', 'padding': '8px 20px',
                'backgroundColor': C['panel'], 'borderColor': C['border2'],
                'color': C['text_dim'],
            }, selected_style={
                'fontFamily': FONT_MONO, 'fontSize': '11px',
                'letterSpacing': '0.1em', 'padding': '8px 20px',
                'backgroundColor': C['bg'], 'borderColor': C['cyan'],
                'borderTop': f'2px solid {C["cyan"]}',
                'color': C['cyan'],
            }, children=[make_explore_tab(metric_opts)]),

            dcc.Tab(label='COMPARE', value='tab-compare', style={
                'fontFamily': FONT_MONO, 'fontSize': '11px',
                'letterSpacing': '0.1em', 'padding': '8px 20px',
                'backgroundColor': C['panel'], 'borderColor': C['border2'],
                'color': C['text_dim'],
            }, selected_style={
                'fontFamily': FONT_MONO, 'fontSize': '11px',
                'letterSpacing': '0.1em', 'padding': '8px 20px',
                'backgroundColor': C['bg'], 'borderColor': C['cyan'],
                'borderTop': f'2px solid {C["cyan"]}',
                'color': C['cyan'],
            }, children=[make_compare_tab(metric_opts)]),
        ], style={'fontFamily': FONT_MONO}),

    ], style={'background': C['bg'], 'minHeight': '100vh'})