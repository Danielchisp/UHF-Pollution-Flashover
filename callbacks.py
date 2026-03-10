from dash import Input, Output, State, callback_context, dcc, html, no_update, ALL
import plotly.graph_objects as go
import h5py
import numpy as np
import atexit
import io
import csv
import random

C = {
    'bg':       '#0d0f14',
    'panel':    '#13161e',
    'border':   '#1e2330',
    'border2':  '#2a3045',
    'cyan':     '#00d4ff',
    'green':    '#00ff9d',
    'red':      '#ff3b5c',
    'text':     '#c8d0e0',
    'text_dim': '#5a6480',
}

PLOT_THEME = dict(
    template='plotly_dark',
    paper_bgcolor=C['panel'],
    plot_bgcolor=C['bg'],
    font=dict(family="'JetBrains Mono', monospace", color=C['text'], size=10),
    xaxis=dict(gridcolor=C['border'], gridwidth=1, linecolor=C['border2'], zerolinecolor=C['border2']),
    yaxis=dict(gridcolor=C['border'], gridwidth=1, linecolor=C['border2'], zerolinecolor=C['border2']),
    legend=dict(bgcolor='rgba(0,0,0,0)', bordercolor=C['border2'], borderwidth=1,
                font=dict(size=10, color=C['text'])),
    margin=dict(l=50, r=20, t=12, b=40),
)

COLORS = ['#00d4ff', '#ff3b5c', '#00ff9d', '#ffaa00', '#b66dff',
          '#ff6db0', '#5bc8ff', '#aaffcc', '#ffdd88', '#e8e8e8']

SELECTION_COLORS = [
    '#00d4ff', '#ff3b5c', '#00ff9d', '#ffaa00', '#b66dff',
    '#ff6db0', '#5bc8ff', '#ff8855', '#88ff88', '#e8e8e8',
    '#ff55ff', '#55ffff', '#ffff55', '#aa88ff', '#ff8888',
]


class HDF5Manager:
    def __init__(self, filepath):
        self.filepath = filepath
        self._file = None

    @property
    def file(self):
        if self._file is None:
            self._file = h5py.File(self.filepath, 'r')
        return self._file

    def get_all_groups(self):
        return list(self.file.keys())

    def get_metrics_from_group(self, group_name, atbs_window=5):
        if group_name == 'all':
            return None
        try:
            metrics_group = self.file[group_name]['Metrics']
            signals_group = self.file[group_name]['Signals']
            timestamps = metrics_group['timestamp'][:].tolist()

            atbs_values, atbs_timestamps, atbs_cumulative = [], [], 0
            for i in range(atbs_window - 1, len(timestamps)):
                window = timestamps[i - atbs_window + 1:i + 1]
                atbs = (window[-1] - window[0]) / (len(window) - 1)
                atbs_cumulative += atbs
                atbs_values.append(atbs_cumulative)
                atbs_timestamps.append(timestamps[i])

            return {
                'B0':             metrics_group['B0'][:].tolist(),
                'B1':             metrics_group['B1'][:].tolist(),
                'energy':         metrics_group['energy'][:].tolist(),
                'eqFreq':         metrics_group['eqFreq'][:].tolist(),
                'eqTime':         metrics_group['eqTime'][:].tolist(),
                'vpp':            metrics_group['vpp'][:].tolist(),
                'timestamp':      timestamps,
                'ATBS':           atbs_values,
                'ATBS_timestamp': atbs_timestamps,
                'signal_ids':     list(signals_group.keys()),
                'group':          group_name,
            }
        except Exception as e:
            print(f"Error leyendo grupo {group_name}: {e}")
            return None

    def get_signal_data(self, group_name, signal_id):
        try:
            return self.file[group_name]['Signals'][signal_id][:]
        except Exception as e:
            print(f"Error leyendo señal {signal_id}: {e}")
            return np.array([])

    def get_fft_data(self, group_name, signal_id):
        try:
            if 'FFT' in self.file[group_name]:
                return self.file[group_name]['FFT'][signal_id][:]
            return np.array([])
        except Exception as e:
            print(f"FFT no disponible para {signal_id}: {e}")
            return np.array([])

    def close(self):
        if self._file:
            self._file.close()
            self._file = None


_hdf5_manager = None


def get_hdf5_manager(filepath):
    global _hdf5_manager
    if _hdf5_manager is None:
        _hdf5_manager = HDF5Manager(filepath)
    return _hdf5_manager


def normalize_data(data):
    a = np.array(data)
    mn, mx = np.min(a), np.max(a)
    return a if mx == mn else (a - mn) / (mx - mn)


def extract_selected_signal_ids(selected_data):
    if not selected_data:
        return None
    pts = selected_data.get('points', [])
    if len(pts) <= 1:
        return None
    ids = set()
    for p in pts:
        cd = p.get('customdata')
        if cd is not None:
            ids.add(cd[0])
    return ids or None


def filter_by_signal_ids(metrics_data, selected_ids):
    if selected_ids is None:
        return None
    indices = []
    for i, sid in enumerate(metrics_data['signal_ids']):
        if sid in selected_ids:
            indices.append(i)
    return sorted(indices) if indices else None


def full_range(arr, padding=0.05):
    mn, mx = np.min(arr), np.max(arr)
    pad = (mx - mn) * padding if mx != mn else 1
    return [mn - pad, mx + pad]


def dark_layout(**extra):
    layout = dict(**PLOT_THEME)
    layout.update(extra)
    return layout


def empty_fig(height=460):
    fig = go.Figure()
    fig.update_layout(**dark_layout(height=height))
    return fig


def app_callbacks(app, archivo_hdf5):
    hdf5_manager = get_hdf5_manager(archivo_hdf5)

    band_labels = {
        'B0': 'B0  0–100 MHz',
        'B1': 'B1  100–600 MHz',
        'ATBS': 'ATBS',
    }

    @app.callback(
        Output('dropdown-group', 'options'),
        Output('dropdown-group', 'value'),
        Input('dropdown-group', 'id'),
    )
    def populate_group_dropdown(_):
        groups = hdf5_manager.get_all_groups()
        return [{'label': g, 'value': g} for g in groups], None

    @app.callback(
        Output('led-duration', 'value'),
        Input('dropdown-group', 'value'),
    )
    def update_led_duration(selected_group):
        if not selected_group:
            return '00:00:00'
        try:
            timestamps = hdf5_manager.file[selected_group]['Metrics']['timestamp'][:]
            duration = float(np.max(timestamps) - np.min(timestamps))
            h = int(duration // 3600)
            m = int((duration % 3600) // 60)
            s = int(duration % 60)
            return f'{h:02d}:{m:02d}:{s:02d}'
        except Exception:
            return '00:00:00'

    @app.callback(
        Output('graph-3', 'figure'),
        Input('dropdown-y-axis', 'value'),
        Input('dropdown-group', 'value'),
        Input('x-axis-mode', 'value'),
        Input('normalize-toggle', 'value'),
        Input('atbs-window-input', 'value'),
        Input('graph-4', 'selectedData'),
    )
    def update_time_series(y_metrics, selected_group, x_mode, normalize, atbs_window, scatter_selection):
        if not selected_group:
            return empty_fig(460)

        y_metrics = y_metrics or ['vpp']
        x_mode = x_mode or 'timestamp'
        atbs_window = max(atbs_window or 5, 2)

        metrics_data = hdf5_manager.get_metrics_from_group(selected_group, atbs_window)
        if not metrics_data:
            return empty_fig(460)

        selected_ids = extract_selected_signal_ids(scatter_selection)
        filter_idx = filter_by_signal_ids(metrics_data, selected_ids)

        try:
            base_metric = next((m for m in y_metrics if m != 'ATBS'), None)
            x_data = metrics_data['timestamp'] if x_mode == 'timestamp' else (
                list(range(len(metrics_data[base_metric]))) if base_metric else [])
            x_label = 'Timestamp' if x_mode == 'timestamp' else 'Index'

            all_x, all_y = [], []
            for ym in y_metrics:
                xm = metrics_data['ATBS_timestamp'] if ym == 'ATBS' and x_mode == 'timestamp' else (
                    list(range(len(metrics_data['ATBS']))) if ym == 'ATBS' else x_data)
                yv = np.array(metrics_data['ATBS'] if ym == 'ATBS' else metrics_data[ym])
                if normalize and len(y_metrics) > 1:
                    yv = normalize_data(yv)
                all_x.extend(xm)
                all_y.extend(yv.tolist())

            x_rng = full_range(np.array(all_x)) if all_x else None
            y_rng = full_range(np.array(all_y)) if all_y else None

            fig = go.Figure()
            for idx, ym in enumerate(y_metrics):
                xm = metrics_data['ATBS_timestamp'] if ym == 'ATBS' and x_mode == 'timestamp' else (
                    list(range(len(metrics_data['ATBS']))) if ym == 'ATBS' else x_data)
                yv = np.array(metrics_data['ATBS'] if ym == 'ATBS' else metrics_data[ym])
                sids = metrics_data['signal_ids'][:len(metrics_data['ATBS'])] if ym == 'ATBS' else metrics_data['signal_ids']
                if normalize and len(y_metrics) > 1:
                    yv = normalize_data(yv)

                cd = np.column_stack((np.array(sids), np.full(len(sids), selected_group)))
                color = COLORS[idx % len(COLORS)]

                if filter_idx is not None:
                    sl = filter_idx
                    xm = [xm[i] for i in sl]
                    yv = yv[sl]
                    cd = cd[sl]
                    marker_opacity = 0.9
                else:
                    marker_opacity = 0

                fig.add_trace(go.Scattergl(
                    x=xm, y=yv, mode='lines+markers',
                    marker=dict(size=3, color=color, opacity=marker_opacity),
                    line=dict(color=color, width=1),
                    customdata=cd, name=band_labels.get(ym, ym), opacity=0.85,
                ))

            y_title = 'Normalized' if (normalize and len(y_metrics) > 1) else 'Value'
            fig.update_layout(**dark_layout(
                height=460,
                xaxis=dict(range=x_rng, title=x_label, gridcolor=C['border'], linecolor=C['border2'], zerolinecolor=C['border2']),
                yaxis=dict(range=y_rng, title=y_title, gridcolor=C['border'], linecolor=C['border2'], zerolinecolor=C['border2']),
                hovermode='closest', clickmode='event+select',
                showlegend=True, dragmode='lasso', uirevision=selected_group,
            ))
            return fig
        except Exception as e:
            print(f"Error time series: {e}")
            return empty_fig(460)

    # ── SCATTER (con modo animación) ─────────────────────────────────────────
    @app.callback(
        Output('graph-4', 'figure'),
        Input('dropdown-x-axis', 'value'),
        Input('dropdown-y-axis-scatter', 'value'),
        Input('dropdown-z-axis', 'value'),
        Input('dropdown-group', 'value'),
        Input('graph-3', 'selectedData'),
        Input('animate-mode', 'value'),
        Input('animate-step-input', 'value'),
        Input('animate-speed-input', 'value'),
    )
    def update_scatter(x_metric, y_metric, z_metric, selected_group,
                       ts_selection, animate_mode, anim_step, anim_speed):
        if not selected_group:
            return empty_fig(460)

        x_metric   = x_metric or 'eqTime'
        y_metric   = y_metric or 'eqFreq'
        anim_step  = max(int(anim_step or 0), 0)   # 0 = automático
        anim_speed = max(int(anim_speed or 40), 5)  # ms por frame
        # animate_mode llega como [] (desmarcado) o ['on'] (marcado)
        do_animate = bool(animate_mode)

        metrics_data = hdf5_manager.get_metrics_from_group(selected_group)
        if not metrics_data:
            return empty_fig(460)

        selected_ids = extract_selected_signal_ids(ts_selection)
        filter_idx   = filter_by_signal_ids(metrics_data, selected_ids)

        try:
            x_vals = np.array(metrics_data[x_metric], dtype=float)
            y_vals = np.array(metrics_data[y_metric], dtype=float)
            sids   = np.array(metrics_data['signal_ids'])
            cd     = np.column_stack((sids, np.full(len(sids), selected_group)))

            if filter_idx is not None:
                x_vals = x_vals[filter_idx]
                y_vals = y_vals[filter_idx]
                cd     = cd[filter_idx]

            # ── MODO 3D (sin animación) ──────────────────────────────────
            if z_metric:
                z_vals = np.array(metrics_data[z_metric], dtype=float)
                if filter_idx is not None:
                    z_vals = z_vals[filter_idx]

                fig = go.Figure()
                fig.add_trace(go.Scatter3d(
                    x=x_vals, y=y_vals, z=z_vals, mode='markers',
                    marker=dict(size=2.5, color=C['red'], opacity=0.7, line=dict(width=0)),
                    customdata=cd, name=f'{x_metric}/{y_metric}/{z_metric}',
                ))
                fig.update_layout(**dark_layout(
                    height=460,
                    scene=dict(
                        xaxis=dict(title=x_metric, backgroundcolor=C['bg'], gridcolor=C['border2'], showbackground=True),
                        yaxis=dict(title=y_metric, backgroundcolor=C['bg'], gridcolor=C['border2'], showbackground=True),
                        zaxis=dict(title=z_metric, backgroundcolor=C['bg'], gridcolor=C['border2'], showbackground=True),
                        bgcolor=C['bg'],
                    ),
                    uirevision=selected_group,
                ))
                return fig

            # Rango fijo calculado sobre todos los puntos (no cambia durante anim)
            x_rng = full_range(x_vals)
            y_rng = full_range(y_vals)

            # ── MODO ESTÁTICO (por defecto) ──────────────────────────────
            if not do_animate:
                fig = go.Figure()
                fig.add_trace(go.Scattergl(
                    x=x_vals.tolist(), y=y_vals.tolist(), mode='markers',
                    marker=dict(size=3, color=C['red'], opacity=0.65),
                    customdata=cd, name=f'{y_metric} vs {x_metric}',
                ))
                fig.update_layout(**dark_layout(
                    height=460,
                    xaxis=dict(range=x_rng, title=x_metric, gridcolor=C['border'], linecolor=C['border2'], zerolinecolor=C['border2']),
                    yaxis=dict(range=y_rng, title=y_metric, gridcolor=C['border'], linecolor=C['border2'], zerolinecolor=C['border2']),
                    hovermode='closest', clickmode='event+select',
                    dragmode='lasso', uirevision=selected_group,
                ))
                return fig

            # ── MODO ANIMACIÓN ───────────────────────────────────────────
            # go.Frames requieren go.Scatter (no Scattergl)
            n    = len(x_vals)
            step = anim_step if anim_step > 0 else max(1, n // 200)

            x_list = x_vals.tolist()
            y_list = y_vals.tolist()

            frames = []
            for end in range(step, n + step, step):
                end = min(end, n)
                frames.append(go.Frame(
                    data=[go.Scatter(
                        x=x_list[:end],
                        y=y_list[:end],
                        mode='markers',
                        marker=dict(size=3, color=C['red'], opacity=0.65),
                        name=f'{y_metric} vs {x_metric}',
                    )],
                    name=str(end),
                ))

            # Trace inicial: primer lote visible desde el arranque
            fig = go.Figure(
                data=[go.Scatter(
                    x=x_list[:step],
                    y=y_list[:step],
                    mode='markers',
                    marker=dict(size=3, color=C['red'], opacity=0.65),
                    name=f'{y_metric} vs {x_metric}',
                )],
                frames=frames,
            )

            fig.update_layout(**dark_layout(
                height=460,
                xaxis=dict(
                    range=x_rng, title=x_metric,
                    gridcolor=C['border'], linecolor=C['border2'], zerolinecolor=C['border2'],
                ),
                yaxis=dict(
                    range=y_rng, title=y_metric,
                    gridcolor=C['border'], linecolor=C['border2'], zerolinecolor=C['border2'],
                ),
                hovermode='closest',
                uirevision=selected_group,
                updatemenus=[dict(
                    type='buttons',
                    showactive=False,
                    x=0.0, y=1.13,
                    xanchor='left', yanchor='top',
                    bgcolor=C['panel'],
                    bordercolor=C['border2'],
                    font=dict(color=C['text'], size=11),
                    buttons=[
                        dict(
                            label='▶  Play',
                            method='animate',
                            args=[None, {
                                'frame':       {'duration': anim_speed, 'redraw': True},
                                'transition':  {'duration': 0},
                                'fromcurrent': True,
                                'mode':        'immediate',
                            }],
                        ),
                        dict(
                            label='⏸  Pause',
                            method='animate',
                            args=[[None], {
                                'frame':      {'duration': 0},
                                'transition': {'duration': 0},
                                'mode':       'immediate',
                            }],
                        ),
                    ],
                )],
                sliders=[dict(
                    active=0,
                    currentvalue=dict(
                        prefix='Señales: ',
                        font=dict(color=C['text_dim'], size=10),
                        visible=True,
                        xanchor='center',
                    ),
                    pad=dict(t=45, b=5),
                    bgcolor=C['panel'],
                    bordercolor=C['border2'],
                    tickcolor=C['border2'],
                    activebgcolor=C['cyan'],
                    font=dict(color=C['text_dim'], size=9),
                    steps=[
                        dict(
                            method='animate',
                            args=[[f.name], {
                                'frame':      {'duration': 0, 'redraw': True},
                                'transition': {'duration': 0},
                                'mode':       'immediate',
                            }],
                            label=f.name,
                        )
                        for f in frames
                    ],
                )],
            ))
            return fig

        except Exception as e:
            print(f"Error scatter: {e}")
            import traceback; traceback.print_exc()
            return empty_fig(460)

    @app.callback(
        Output('graph-1', 'figure'),
        Input('graph-3', 'clickData'),
        Input('graph-4', 'clickData'),
    )
    def update_temporal_signal(cd3, cd4):
        ctx = callback_context
        if not ctx.triggered:
            return empty_fig(340)
        click_data = cd3 if ctx.triggered[0]['prop_id'].split('.')[0] == 'graph-3' else cd4
        if not click_data:
            return empty_fig(340)
        try:
            signal_id, group_name = click_data['points'][0]['customdata']
            data = hdf5_manager.get_signal_data(group_name, signal_id)
            if not len(data):
                return empty_fig(340)
            fig = go.Figure()
            fig.add_trace(go.Scattergl(
                x=np.linspace(0, len(data), len(data)), y=data,
                mode='lines', line=dict(color=C['green'], width=1), name='Signal',
            ))
            fig.update_layout(**dark_layout(
                height=340,
                xaxis=dict(title='Time (µs)', gridcolor=C['border'], linecolor=C['border2'], zerolinecolor=C['border2']),
                yaxis=dict(title='Amplitude', gridcolor=C['border'], linecolor=C['border2'], zerolinecolor=C['border2']),
            ))
            return fig
        except Exception as e:
            print(f"Error temporal: {e}")
            return empty_fig(340)

    @app.callback(
        Output('graph-2', 'figure'),
        Input('graph-3', 'clickData'),
        Input('graph-4', 'clickData'),
    )
    def update_fft(cd3, cd4):
        ctx = callback_context
        if not ctx.triggered:
            return empty_fig(340)
        click_data = cd3 if ctx.triggered[0]['prop_id'].split('.')[0] == 'graph-3' else cd4
        if not click_data:
            return empty_fig(340)
        try:
            signal_id, group_name = click_data['points'][0]['customdata']
            fft = hdf5_manager.get_fft_data(group_name, signal_id)
            if not len(fft):
                return empty_fig(340)
            fig = go.Figure()
            fig.add_trace(go.Scattergl(
                y=fft, mode='lines', line=dict(color=C['red'], width=1), name='FFT',
            ))
            fig.update_layout(**dark_layout(
                height=340,
                xaxis=dict(title='Frequency (MHz)', gridcolor=C['border'], linecolor=C['border2'], zerolinecolor=C['border2']),
                yaxis=dict(title='Magnitude', gridcolor=C['border'], linecolor=C['border2'], zerolinecolor=C['border2']),
            ))
            return fig
        except Exception as e:
            print(f"Error FFT: {e}")
            return empty_fig(340)

    @app.callback(
        Output('download-scatter-csv', 'data'),
        Input('btn-download-scatter', 'n_clicks'),
        State('dropdown-x-axis', 'value'),
        State('dropdown-y-axis-scatter', 'value'),
        State('dropdown-z-axis', 'value'),
        State('dropdown-group', 'value'),
        State('graph-3', 'selectedData'),
        prevent_initial_call=True,
    )
    def download_scatter_csv(n_clicks, x_metric, y_metric, z_metric, selected_group, ts_selection):
        if not selected_group or not x_metric or not y_metric:
            return None
        metrics_data = hdf5_manager.get_metrics_from_group(selected_group)
        if not metrics_data:
            return None

        x_vals = np.array(metrics_data[x_metric])
        y_vals = np.array(metrics_data[y_metric])
        z_vals = np.array(metrics_data[z_metric]) if z_metric else None

        selected_ids = extract_selected_signal_ids(ts_selection)
        filter_idx = filter_by_signal_ids(metrics_data, selected_ids)
        if filter_idx is not None:
            x_vals, y_vals = x_vals[filter_idx], y_vals[filter_idx]
            if z_vals is not None:
                z_vals = z_vals[filter_idx]

        buf = io.StringIO()
        w = csv.writer(buf)
        if z_vals is not None:
            for row in zip(x_vals, y_vals, z_vals):
                w.writerow(row)
        else:
            for row in zip(x_vals, y_vals):
                w.writerow(row)
        return dcc.send_string(buf.getvalue(), f'scatter_{selected_group}_{x_metric}_{y_metric}.csv')

    @app.callback(
        Output('download-signals-csv', 'data'),
        Input('btn-download-signals', 'n_clicks'),
        State('dropdown-group', 'value'),
        State('graph-3', 'selectedData'),
        State('graph-4', 'selectedData'),
        prevent_initial_call=True,
    )
    def download_signals(n_clicks, selected_group, ts_selection, sc_selection):
        if not selected_group:
            return None
        metrics_data = hdf5_manager.get_metrics_from_group(selected_group)
        if not metrics_data:
            return None

        selected_ids = extract_selected_signal_ids(ts_selection) or extract_selected_signal_ids(sc_selection)
        if selected_ids:
            signal_ids = [sid for sid in metrics_data['signal_ids'] if sid in selected_ids]
        else:
            signal_ids = metrics_data['signal_ids']

        columns = []
        for sid in signal_ids:
            data = hdf5_manager.get_signal_data(selected_group, sid)
            if len(data):
                columns.append(data)

        if not columns:
            return None

        max_len = max(len(col) for col in columns)
        buf = io.StringIO()
        w = csv.writer(buf)
        for i in range(max_len):
            row = [col[i] if i < len(col) else '' for col in columns]
            w.writerow(row)

        return dcc.send_string(buf.getvalue(), f'signals_{selected_group}.csv')

    @app.callback(
        Output('store-selections', 'data'),
        Input('btn-insert-selection', 'n_clicks'),
        Input({'type': 'btn-delete-selection', 'index': ALL}, 'n_clicks'),
        State('store-selections', 'data'),
        State('dropdown-group', 'value'),
        State('graph-3', 'selectedData'),
        State('graph-4', 'selectedData'),
        State('input-selection-name', 'value'),
        prevent_initial_call=True,
    )
    def manage_selections(insert_clicks, delete_clicks, current_selections,
                          selected_group, ts_sel, sc_sel, sel_name):
        ctx = callback_context
        if not ctx.triggered:
            return no_update

        triggered_id = ctx.triggered[0]['prop_id']
        current_selections = current_selections or []

        if 'btn-delete-selection' in triggered_id:
            import json
            idx_info = json.loads(triggered_id.split('.')[0])
            idx = idx_info['index']
            current_selections = [s for i, s in enumerate(current_selections) if i != idx]
            return current_selections

        if not selected_group:
            return no_update

        selected_ids = extract_selected_signal_ids(ts_sel) or extract_selected_signal_ids(sc_sel)
        if not selected_ids:
            return no_update

        name = sel_name or f'Selection {len(current_selections) + 1}'
        color = SELECTION_COLORS[len(current_selections) % len(SELECTION_COLORS)]

        current_selections.append({
            'name': name,
            'group': selected_group,
            'signal_ids': list(selected_ids),
            'color': color,
        })
        return current_selections

    @app.callback(
        Output('selections-list-container', 'children'),
        Input('store-selections', 'data'),
    )
    def update_selections_list(selections):
        selections = selections or []
        if not selections:
            return html.Div('No selections saved yet.', style={
                'color': C['text_dim'], 'fontFamily': "'JetBrains Mono', monospace",
                'fontSize': '11px', 'padding': '20px',
            })

        items = []
        for i, sel in enumerate(selections):
            items.append(html.Div([
                html.Span('●', style={'color': sel['color'], 'fontSize': '14px', 'marginRight': '8px'}),
                html.Span(sel['name'], style={
                    'color': C['text'], 'fontFamily': "'JetBrains Mono', monospace",
                    'fontSize': '11px', 'marginRight': '10px',
                }),
                html.Span(f"({len(sel['signal_ids'])} signals)", style={
                    'color': C['text_dim'], 'fontFamily': "'JetBrains Mono', monospace",
                    'fontSize': '10px', 'marginRight': '10px',
                }),
                html.Span(f"[{sel['group']}]", style={
                    'color': C['text_dim'], 'fontFamily': "'JetBrains Mono', monospace",
                    'fontSize': '10px', 'marginRight': '10px',
                }),
                html.Button('✕', id={'type': 'btn-delete-selection', 'index': i}, style={
                    'background': 'transparent', 'border': f'1px solid {C["red"]}',
                    'color': C['red'], 'fontSize': '10px', 'cursor': 'pointer',
                    'borderRadius': '3px', 'padding': '1px 6px',
                }),
            ], style={
                'display': 'flex', 'alignItems': 'center',
                'padding': '6px 10px', 'marginBottom': '4px',
                'background': C['bg'], 'borderRadius': '4px',
                'border': f'1px solid {C["border"]}',
            }))
        return items

    @app.callback(
        Output('graph-compare-avg-fft', 'figure'),
        Input('store-selections', 'data'),
    )
    def compare_avg_fft(selections):
        selections = selections or []
        if not selections:
            return empty_fig(380)

        fig = go.Figure()
        for sel in selections:
            group = sel['group']
            sids = sel['signal_ids']
            ffts = []
            for sid in sids:
                fft = hdf5_manager.get_fft_data(group, sid)
                if len(fft):
                    ffts.append(fft)
            if not ffts:
                continue
            max_len = max(len(f) for f in ffts)
            padded = [np.pad(f, (0, max_len - len(f))) for f in ffts]
            avg_fft = np.mean(padded, axis=0)
            fig.add_trace(go.Scattergl(
                y=avg_fft, mode='lines',
                line=dict(color=sel['color'], width=1.5),
                name=sel['name'], opacity=0.85,
            ))

        fig.update_layout(**dark_layout(
            height=380,
            xaxis=dict(title='Frequency (MHz)', gridcolor=C['border'], linecolor=C['border2'], zerolinecolor=C['border2']),
            yaxis=dict(title='Magnitude (avg)', gridcolor=C['border'], linecolor=C['border2'], zerolinecolor=C['border2']),
            showlegend=True,
        ))
        return fig

    @app.callback(
        Output('graph-compare-signals', 'figure'),
        Input('store-selections', 'data'),
        Input('input-random-signals', 'value'),
    )
    def compare_random_signals(selections, n_signals):
        selections = selections or []
        n_signals = max(int(n_signals or 10), 1)
        if not selections:
            return empty_fig(380)

        fig = go.Figure()
        for sel in selections:
            group = sel['group']
            sids = sel['signal_ids']
            sample = random.sample(sids, min(n_signals, len(sids)))
            for i, sid in enumerate(sample):
                data = hdf5_manager.get_signal_data(group, sid)
                if not len(data):
                    continue
                fig.add_trace(go.Scattergl(
                    x=np.linspace(0, len(data), len(data)), y=data,
                    mode='lines', line=dict(color=sel['color'], width=0.8),
                    name=sel['name'] if i == 0 else None,
                    showlegend=(i == 0), opacity=0.6,
                    legendgroup=sel['name'],
                ))

        fig.update_layout(**dark_layout(
            height=380,
            xaxis=dict(title='Time (µs)', gridcolor=C['border'], linecolor=C['border2'], zerolinecolor=C['border2']),
            yaxis=dict(title='Amplitude', gridcolor=C['border'], linecolor=C['border2'], zerolinecolor=C['border2']),
            showlegend=True,
        ))
        return fig

    @app.callback(
        Output('graph-compare-fft', 'figure'),
        Input('store-selections', 'data'),
        Input('input-random-signals', 'value'),
    )
    def compare_random_fft(selections, n_signals):
        selections = selections or []
        n_signals = max(int(n_signals or 10), 1)
        if not selections:
            return empty_fig(380)

        fig = go.Figure()
        for sel in selections:
            group = sel['group']
            sids = sel['signal_ids']
            sample = random.sample(sids, min(n_signals, len(sids)))
            for i, sid in enumerate(sample):
                fft = hdf5_manager.get_fft_data(group, sid)
                if not len(fft):
                    continue
                fig.add_trace(go.Scattergl(
                    y=fft, mode='lines',
                    line=dict(color=sel['color'], width=0.8),
                    name=sel['name'] if i == 0 else None,
                    showlegend=(i == 0), opacity=0.6,
                    legendgroup=sel['name'],
                ))

        fig.update_layout(**dark_layout(
            height=380,
            xaxis=dict(title='Frequency (MHz)', gridcolor=C['border'], linecolor=C['border2'], zerolinecolor=C['border2']),
            yaxis=dict(title='Magnitude', gridcolor=C['border'], linecolor=C['border2'], zerolinecolor=C['border2']),
            showlegend=True,
        ))
        return fig

    @app.callback(
        Output('graph-compare-scatter', 'figure'),
        Input('store-selections', 'data'),
        Input('dropdown-compare-x', 'value'),
        Input('dropdown-compare-y', 'value'),
    )
    def compare_scatter(selections, x_metric, y_metric):
        selections = selections or []
        x_metric = x_metric or 'eqTime'
        y_metric = y_metric or 'eqFreq'
        if not selections:
            return empty_fig(380)

        fig = go.Figure()
        for sel in selections:
            group = sel['group']
            metrics_data = hdf5_manager.get_metrics_from_group(group)
            if not metrics_data:
                continue
            filter_idx = filter_by_signal_ids(metrics_data, set(sel['signal_ids']))
            if filter_idx is None:
                continue
            x_vals = np.array(metrics_data[x_metric])[filter_idx]
            y_vals = np.array(metrics_data[y_metric])[filter_idx]
            fig.add_trace(go.Scattergl(
                x=x_vals, y=y_vals, mode='markers',
                marker=dict(size=3, color=sel['color'], opacity=0.7),
                name=sel['name'],
            ))

        fig.update_layout(**dark_layout(
            height=380,
            xaxis=dict(title=x_metric, gridcolor=C['border'], linecolor=C['border2'], zerolinecolor=C['border2']),
            yaxis=dict(title=y_metric, gridcolor=C['border'], linecolor=C['border2'], zerolinecolor=C['border2']),
            showlegend=True,
        ))
        return fig

    @app.callback(
        Output('graph-compare-timeseries', 'figure'),
        Input('store-selections', 'data'),
        Input('dropdown-compare-ts-metric', 'value'),
    )
    def compare_timeseries(selections, metric):
        selections = selections or []
        metric = metric or 'vpp'
        if not selections:
            return empty_fig(380)

        fig = go.Figure()
        for sel in selections:
            group = sel['group']
            metrics_data = hdf5_manager.get_metrics_from_group(group)
            if not metrics_data:
                continue
            filter_idx = filter_by_signal_ids(metrics_data, set(sel['signal_ids']))
            if filter_idx is None:
                continue
            timestamps = np.array(metrics_data['timestamp'])[filter_idx]
            values = np.array(metrics_data[metric])[filter_idx]
            fig.add_trace(go.Scattergl(
                x=timestamps, y=values, mode='lines+markers',
                marker=dict(size=3, color=sel['color'], opacity=0.8),
                line=dict(color=sel['color'], width=1),
                name=sel['name'], opacity=0.85,
            ))

        fig.update_layout(**dark_layout(
            height=380,
            xaxis=dict(title='Timestamp', gridcolor=C['border'], linecolor=C['border2'], zerolinecolor=C['border2']),
            yaxis=dict(title=metric, gridcolor=C['border'], linecolor=C['border2'], zerolinecolor=C['border2']),
            showlegend=True,
        ))
        return fig

    @atexit.register
    def cleanup():
        if _hdf5_manager:
            _hdf5_manager.close()