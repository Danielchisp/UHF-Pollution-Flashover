# -*- coding: utf-8 -*-
from dash import Input, Output, State, callback_context, dcc, html, no_update, ALL
import plotly.graph_objects as go
import h5py
import numpy as np
from scipy.stats import kurtosis as sp_kurtosis, skew as sp_skew
from scipy.fft import rfft, rfftfreq
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
    'orange':   '#ffaa00',
    'text':     '#c8d0e0',
    'text_dim': '#5a6480',
}

PLOT_THEME = dict(
    template='plotly_dark',
    paper_bgcolor=C['panel'],
    plot_bgcolor=C['bg'],
    font=dict(family="'JetBrains Mono', monospace", color=C['text'], size=10),
    legend=dict(bgcolor='rgba(0,0,0,0)', bordercolor=C['border2'], borderwidth=1,
                font=dict(size=10, color=C['text'])),
    margin=dict(l=50, r=70, t=12, b=40),
)

COLORS = ['#00d4ff', '#ff3b5c', '#00ff9d', '#ffaa00', '#b66dff',
          '#ff6db0', '#5bc8ff', '#aaffcc', '#ffdd88', '#e8e8e8']

SELECTION_COLORS = [
    '#00d4ff', '#ff3b5c', '#00ff9d', '#ffaa00', '#b66dff',
    '#ff6db0', '#5bc8ff', '#ff8855', '#88ff88', '#e8e8e8',
    '#ff55ff', '#55ffff', '#ffff55', '#aa88ff', '#ff8888',
]

AXIS_STYLE = dict(
    gridcolor=C['border'], gridwidth=1,
    linecolor=C['border2'], zerolinecolor=C['border2'],
)

# Parametros FFT para formato 2
SAMPLE_RATE   = 3e9          # 3 GS/s - ajustar si difiere
B0_MAX_HZ     = 100e6        # 0-100 MHz
B1_MIN_HZ     = 100e6        # 100-600 MHz
B1_MAX_HZ     = 600e6


# -----------------------------------------------------------------------------
# Helpers de calculo de metricas desde senales raw (formato 2)
# -----------------------------------------------------------------------------

def _compute_metrics_from_signals(signals: np.ndarray,
                                  timestamps: np.ndarray,
                                  sample_rate: float = SAMPLE_RATE):
    n = len(signals)
    vpp_list          = []
    kurt_list         = []
    skew_list         = []
    crest_list        = []
    energy_list       = []
    eqFreq_list       = []
    eqTime_list       = list(timestamps[:n])
    B0_list           = []
    B1_list           = []

    for i, sig in enumerate(signals):
        sig = sig.astype(np.float64)
        sig_len = len(sig)

        vpp    = float(sig.max() - sig.min())
        rms    = float(np.sqrt(np.mean(sig ** 2)))
        peak   = float(np.max(np.abs(sig)))
        crest  = float(peak / rms) if rms > 0 else 0.0
        energy = float(np.sum(sig ** 2))
        kurt   = float(sp_kurtosis(sig, fisher=False))
        skew   = float(sp_skew(sig))

        freqs   = rfftfreq(sig_len, d=1.0 / sample_rate)
        fft_mag = np.abs(rfft(sig))

        peak_idx = int(np.argmax(fft_mag))
        eq_freq  = float(freqs[peak_idx]) / 1e6   # en MHz

        total_energy = float(np.sum(fft_mag ** 2)) or 1.0
        mask_b0 = freqs <= B0_MAX_HZ
        mask_b1 = (freqs >= B1_MIN_HZ) & (freqs <= B1_MAX_HZ)
        b0 = float(np.sum(fft_mag[mask_b0] ** 2)) / total_energy
        b1 = float(np.sum(fft_mag[mask_b1] ** 2)) / total_energy

        vpp_list.append(vpp)
        kurt_list.append(kurt)
        skew_list.append(skew)
        crest_list.append(crest)
        energy_list.append(energy)
        eqFreq_list.append(eq_freq)
        B0_list.append(b0)
        B1_list.append(b1)

    return {
        'vpp':          vpp_list,
        'kurtosis':     kurt_list,
        'skewness':     skew_list,
        'crest_factor': crest_list,
        'energy':       energy_list,
        'eqFreq':       eqFreq_list,
        'eqTime':       eqTime_list,
        'B0':           B0_list,
        'B1':           B1_list,
        'timestamp':    list(timestamps[:n]),
    }


def _detect_format(group) -> str:
    keys = list(group.keys())
    if 'Metrics' in keys and 'Signals' in keys:
        return 'fmt1'
    for k in keys:
        try:
            sub = group[k]
            if 'signals' in sub and 'data' in sub['signals']:
                return 'fmt2'
        except Exception:
            continue
    return 'fmt1'


# -----------------------------------------------------------------------------
# HDF5Manager
# -----------------------------------------------------------------------------

class HDF5Manager:
    def __init__(self, filepath):
        self.filepath = filepath
        self._file = None
        self._fmt2_cache = {}
        self._precomputed = {}

    @property
    def file(self):
        if self._file is None:
            self._file = h5py.File(self.filepath, 'r')
        return self._file

    def get_all_groups(self):
        return list(self.file.keys())

    # -- Metricas precalculadas (precompute_metrics.py) -----------------------

    def _load_precomputed(self, group_name):
        """Lee el grupo 'Metrics' escrito por precompute_metrics.py.
        Devuelve None si no existe o esta incompleto (entonces se recalcula)."""
        if group_name in self._precomputed:
            return self._precomputed[group_name]
        try:
            grp = self.file[group_name]
            if 'Metrics' not in grp:
                return None
            m = grp['Metrics']
            req = {'vpp', 'eqTime', 'eqFreq', 'energy', 'B0', 'B1',
                   'kurtosis', 'skewness', 'crest_factor', 'timestamp'}
            if not req.issubset(set(m.keys())):
                return None
            # signal_ids es obligatorio para mapear puntos -> senales
            if 'signal_ids' not in m:
                return None

            data = {k: m[k][:].tolist() for k in req}
            raw_ids = m['signal_ids'][:]
            data['signal_ids'] = [s.decode() if isinstance(s, (bytes, np.bytes_)) else str(s)
                                  for s in raw_ids]

            # Humedad (ligera, no toca las senales)
            rh_h, rh_t = [], []
            if 'Relative Humidity' in grp:
                try:
                    rhg = grp['Relative Humidity']
                    rh_h = rhg['humidity'][:].tolist()
                    rh_t = rhg['timestamps'][:].tolist()
                except Exception:
                    pass
            else:
                for cn in sorted(grp.keys()):
                    if cn == 'Metrics':
                        continue
                    try:
                        rhg = grp[cn]['humidity']
                        rh_h.extend(rhg['humidity'][:].tolist())
                        rh_t.extend(rhg['timestamps'][:].tolist())
                    except Exception:
                        continue
                if rh_t:
                    pairs = sorted(zip(rh_t, rh_h))
                    rh_t = [p[0] for p in pairs]
                    rh_h = [p[1] for p in pairs]

            data['rh_humidity']   = rh_h
            data['rh_timestamps'] = rh_t
            data['group']         = group_name
            data['_format']       = 'fmt2'
            self._precomputed[group_name] = data
            print(f"[precomputed] {group_name}: {len(data['signal_ids'])} senales leidas de Metrics")
            return data
        except Exception as e:
            print(f"[precomputed] Error en {group_name}: {e}")
            return None

    # -- Formato 1 ------------------------------------------------------------

    def _get_metrics_fmt1(self, group_name, atbs_window=5):
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

            rh_humidity, rh_timestamps = [], []
            try:
                rh_group      = self.file[group_name]['Relative Humidity']
                rh_humidity   = rh_group['humidity'][:].tolist()
                rh_timestamps = rh_group['timestamps'][:].tolist()
                print(f"[RH] {len(rh_humidity)} pts | "
                      f"t=[{rh_timestamps[0]:.2f}...{rh_timestamps[-1]:.2f}] | "
                      f"h=[{rh_humidity[0]:.2f}...{rh_humidity[-1]:.2f}]")
            except Exception as e:
                print(f"[RH] not available for {group_name}: {e}")

            return {
                'B0':             metrics_group['B0'][:].tolist(),
                'B1':             metrics_group['B1'][:].tolist(),
                'energy':         metrics_group['energy'][:].tolist(),
                'eqFreq':         metrics_group['eqFreq'][:].tolist(),
                'eqTime':         metrics_group['eqTime'][:].tolist(),
                'vpp':            metrics_group['vpp'][:].tolist(),
                'kurtosis':       metrics_group['kurtosis'][:].tolist(),
                'skewness':       metrics_group['skewness'][:].tolist(),
                'crest_factor':   metrics_group['crest_factor'][:].tolist(),
                'timestamp':      timestamps,
                'ATBS':           atbs_values,
                'ATBS_timestamp': atbs_timestamps,
                'signal_ids':     list(signals_group.keys()),
                'group':          group_name,
                'rh_humidity':    rh_humidity,
                'rh_timestamps':  rh_timestamps,
                '_format':        'fmt1',
            }
        except Exception as e:
            print(f"Error leyendo grupo fmt1 {group_name}: {e}")
            return None

    # -- Formato 2 ------------------------------------------------------------

    def _get_metrics_fmt2(self, group_name, atbs_window=5):
        if group_name in self._fmt2_cache:
            return self._fmt2_cache[group_name]

        try:
            group = self.file[group_name]
            chunk_names = sorted(group.keys())

            all_metrics = {k: [] for k in
                           ['vpp', 'kurtosis', 'skewness', 'crest_factor',
                            'energy', 'eqFreq', 'eqTime', 'B0', 'B1', 'timestamp']}
            all_signal_ids    = []
            rh_humidity_all   = []
            rh_timestamps_all = []

            self._fmt2_signal_map = getattr(self, '_fmt2_signal_map', {})
            self._fmt2_signal_map[group_name] = {}

            for chunk_name in chunk_names:
                chunk = group[chunk_name]
                if 'signals' not in chunk or 'data' not in chunk['signals']:
                    continue

                signals_raw    = chunk['signals']['data'][:]
                timestamps_raw = chunk['signals']['timestamps'][:]
                n = len(signals_raw)
                if n == 0:
                    continue

                chunk_metrics = _compute_metrics_from_signals(signals_raw, timestamps_raw)

                for k in all_metrics:
                    all_metrics[k].extend(chunk_metrics[k])

                for idx in range(n):
                    sid = f"{chunk_name}::{idx}"
                    all_signal_ids.append(sid)
                    self._fmt2_signal_map[group_name][sid] = (chunk_name, idx)

                # Leer humedad del chunk si existe
                try:
                    rh_group = chunk['humidity']
                    rh_h     = rh_group['humidity'][:]
                    rh_t     = rh_group['timestamps'][:]
                    rh_humidity_all.extend(rh_h.tolist())
                    rh_timestamps_all.extend(rh_t.tolist())
                except Exception:
                    pass

            if not all_signal_ids:
                print(f"[fmt2] No signals found in {group_name}")
                return None

            timestamps = all_metrics['timestamp']

            atbs_values, atbs_timestamps, atbs_cumulative = [], [], 0
            for i in range(atbs_window - 1, len(timestamps)):
                window = timestamps[i - atbs_window + 1:i + 1]
                dt = (window[-1] - window[0]) / max(len(window) - 1, 1)
                atbs_cumulative += dt
                atbs_values.append(atbs_cumulative)
                atbs_timestamps.append(timestamps[i])

            # Ordenar humedad por timestamp
            if rh_timestamps_all:
                rh_pairs          = sorted(zip(rh_timestamps_all, rh_humidity_all))
                rh_timestamps_all = [p[0] for p in rh_pairs]
                rh_humidity_all   = [p[1] for p in rh_pairs]
                print(f"[fmt2 RH] {group_name}: {len(rh_humidity_all)} pts | "
                      f"t=[{rh_timestamps_all[0]:.2f}...{rh_timestamps_all[-1]:.2f}] | "
                      f"h=[{rh_humidity_all[0]:.2f}...{rh_humidity_all[-1]:.2f}]")

            result = {
                **all_metrics,
                'ATBS':           atbs_values,
                'ATBS_timestamp': atbs_timestamps,
                'signal_ids':     all_signal_ids,
                'group':          group_name,
                'rh_humidity':    rh_humidity_all,
                'rh_timestamps':  rh_timestamps_all,
                '_format':        'fmt2',
            }
            self._fmt2_cache[group_name] = result
            print(f"[fmt2] {group_name}: {len(all_signal_ids)} senales calculadas")
            return result

        except Exception as e:
            print(f"Error leyendo grupo fmt2 {group_name}: {e}")
            import traceback; traceback.print_exc()
            return None

    # -- Interfaz publica -----------------------------------------------------

    def get_format(self, group_name):
        try:
            return _detect_format(self.file[group_name])
        except Exception:
            return 'fmt1'

    def get_metrics_from_group(self, group_name, atbs_window=5):
        if not group_name or group_name == 'all':
            return None
        fmt = self.get_format(group_name)
        if fmt == 'fmt2':
            # PRIORIDAD: usar metricas precalculadas si existen.
            # Evita recalcular FFT por senal en cada seleccion (causa del 'updating').
            pre = self._load_precomputed(group_name)
            if pre is not None:
                ts = pre['timestamp']
                atbs_values, atbs_timestamps, atbs_cumulative = [], [], 0
                for i in range(atbs_window - 1, len(ts)):
                    dt = (ts[i] - ts[i - atbs_window + 1]) / max(atbs_window - 1, 1)
                    atbs_cumulative += dt
                    atbs_values.append(atbs_cumulative)
                    atbs_timestamps.append(ts[i])
                res = dict(pre)
                res['ATBS'] = atbs_values
                res['ATBS_timestamp'] = atbs_timestamps
                return res
            return self._get_metrics_fmt2(group_name, atbs_window)
        return self._get_metrics_fmt1(group_name, atbs_window)

    def get_signal_data(self, group_name, signal_id):
        fmt = self.get_format(group_name)
        if fmt == 'fmt2':
            try:
                sig_map = getattr(self, '_fmt2_signal_map', {})
                if group_name in sig_map and signal_id in sig_map[group_name]:
                    chunk_name, idx = sig_map[group_name][signal_id]
                else:
                    # Reconstruir desde el id "chunk::idx" (modo precalculado)
                    chunk_name, idx_str = signal_id.rsplit('::', 1)
                    idx = int(idx_str)
                return self.file[group_name][chunk_name]['signals']['data'][idx].astype(np.float64)
            except Exception as e:
                print(f"[fmt2] Error get_signal_data {signal_id}: {e}")
                return np.array([])
        else:
            try:
                return self.file[group_name]['Signals'][signal_id][:]
            except Exception as e:
                print(f"Error leyendo senal {signal_id}: {e}")
                return np.array([])

    def get_fft_data(self, group_name, signal_id):
        fmt = self.get_format(group_name)
        if fmt == 'fmt2':
            sig = self.get_signal_data(group_name, signal_id)
            if not len(sig):
                return np.array([])
            return np.abs(rfft(sig))
        else:
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


# -----------------------------------------------------------------------------
# Singleton
# -----------------------------------------------------------------------------

_hdf5_manager = None


def get_hdf5_manager(filepath):
    global _hdf5_manager
    if _hdf5_manager is None:
        _hdf5_manager = HDF5Manager(filepath)
    return _hdf5_manager


# -----------------------------------------------------------------------------
# Helpers genericos
# -----------------------------------------------------------------------------

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


def get_excluded_ids_for_group(store_excluded, group_name):
    if not store_excluded or not group_name:
        return set()
    return set(store_excluded.get(group_name, []))


def apply_exclusion_to_metrics(metrics_data, excluded_ids):
    if not excluded_ids:
        return None
    keep = []
    for i, sid in enumerate(metrics_data['signal_ids']):
        if sid not in excluded_ids:
            keep.append(i)
    return keep if len(keep) < len(metrics_data['signal_ids']) else None


def full_range(arr, padding=0.05):
    mn, mx = np.min(arr), np.max(arr)
    pad = (mx - mn) * padding if mx != mn else 1
    return [mn - pad, mx + pad]


def base_layout(height, **extra):
    layout = dict(**PLOT_THEME)
    layout['height'] = height
    layout.update(extra)
    return layout


def empty_fig(height=460):
    fig = go.Figure()
    fig.update_layout(**base_layout(
        height,
        xaxis=dict(**AXIS_STYLE),
        yaxis=dict(**AXIS_STYLE),
    ))
    return fig


# -----------------------------------------------------------------------------
# Callbacks
# -----------------------------------------------------------------------------

def app_callbacks(app, archivo_hdf5):
    hdf5_manager = get_hdf5_manager(archivo_hdf5)

    band_labels = {
        'B0':   'B0  0-100 MHz',
        'B1':   'B1  100-600 MHz',
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
            fmt = hdf5_manager.get_format(selected_group)
            if fmt == 'fmt2':
                group = hdf5_manager.file[selected_group]
                all_ts = []
                for k in sorted(group.keys()):
                    try:
                        ts = group[k]['signals']['timestamps'][:]
                        all_ts.extend(ts.tolist())
                    except Exception:
                        continue
                if not all_ts:
                    return '00:00:00'
                duration = float(max(all_ts) - min(all_ts))
            else:
                timestamps = hdf5_manager.file[selected_group]['Metrics']['timestamp'][:]
                duration = float(np.max(timestamps) - np.min(timestamps))

            h = int(duration // 3600)
            m = int((duration % 3600) // 60)
            s = int(duration % 60)
            return f'{h:02d}:{m:02d}:{s:02d}'
        except Exception:
            return '00:00:00'

    @app.callback(
        Output('group-rated-voltage', 'children'),
        Output('group-flashover-status', 'children'),
        Output('group-flashover-status', 'style'),
        Input('dropdown-group', 'value'),
    )
    def update_group_attrs(selected_group):
        base_style = {
            'fontFamily': "'JetBrains Mono', monospace",
            'fontSize': '13px',
            'letterSpacing': '0.05em',
            'fontWeight': '600',
        }
        if not selected_group:
            return '--', '--', {**base_style, 'color': C['text_dim']}
        try:
            grp       = hdf5_manager.file[selected_group]
            voltage   = grp.attrs.get('rated_voltage',    '--')
            flashover = grp.attrs.get('flashover_status', '--')

            flashover_str = str(flashover).strip().upper()
            if flashover_str in ('TRUE', '1', 'YES', 'FLASHOVER'):
                flashover_color = C['red']
            elif flashover_str in ('FALSE', '0', 'NO', 'NORMAL'):
                flashover_color = C['green']
            else:
                flashover_color = C['text_dim']

            return str(voltage), str(flashover), {**base_style, 'color': flashover_color}
        except Exception as e:
            print(f"Error leyendo atributos de grupo: {e}")
            return '--', '--', {**base_style, 'color': C['text_dim']}

    # -- EXCLUSION STORE ------------------------------------------------------
    @app.callback(
        Output('store-excluded', 'data'),
        Input('btn-exclude-points',   'n_clicks'),
        Input('btn-reset-exclusions', 'n_clicks'),
        State('store-excluded',        'data'),
        State('dropdown-group',        'value'),
        State('graph-3',               'selectedData'),
        State('graph-4',               'selectedData'),
        prevent_initial_call=True,
    )
    def manage_exclusions(exc_clicks, reset_clicks,
                          current_excluded, selected_group,
                          ts_sel, sc_sel):
        ctx = callback_context
        if not ctx.triggered:
            return no_update

        triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]
        current_excluded = current_excluded or {}

        if triggered_id == 'btn-reset-exclusions':
            if selected_group and selected_group in current_excluded:
                updated = dict(current_excluded)
                del updated[selected_group]
                return updated
            return no_update

        if triggered_id == 'btn-exclude-points':
            if not selected_group:
                return no_update
            selected_ids = (extract_selected_signal_ids(ts_sel)
                            or extract_selected_signal_ids(sc_sel))
            if not selected_ids:
                return no_update
            updated = dict(current_excluded)
            existing = set(updated.get(selected_group, []))
            existing.update(selected_ids)
            updated[selected_group] = list(existing)
            return updated

        return no_update

    @app.callback(
        Output('exclusion-status', 'children'),
        Input('store-excluded',    'data'),
        Input('dropdown-group',    'value'),
    )
    def update_exclusion_status(store_excluded, selected_group):
        excluded_ids = get_excluded_ids_for_group(store_excluded, selected_group)
        n = len(excluded_ids)
        if n == 0:
            return html.Span('no exclusions active', style={
                'color': C['text_dim'], 'fontFamily': "'JetBrains Mono', monospace",
                'fontSize': '10px',
            })
        return html.Span([
            html.Span('o ', style={'color': C['orange'], 'fontSize': '12px'}),
            html.Span(
                f'{n} signal{"s" if n != 1 else ""} excluded from this group',
                style={'color': C['orange'], 'fontFamily': "'JetBrains Mono', monospace",
                       'fontSize': '10px', 'letterSpacing': '0.06em'},
            ),
        ])

    # -- TIME SERIES ----------------------------------------------------------
    @app.callback(
        Output('graph-3', 'figure'),
        Input('dropdown-y-axis',    'value'),
        Input('dropdown-group',     'value'),
        Input('x-axis-mode',        'value'),
        Input('normalize-toggle',   'value'),
        Input('atbs-window-input',  'value'),
        Input('graph-4',            'selectedData'),
        Input('store-excluded',     'data'),
    )
    def update_time_series(y_metrics, selected_group, x_mode, normalize,
                           atbs_window, scatter_selection, store_excluded):
        if not selected_group:
            return empty_fig(460)

        y_metrics   = y_metrics or ['vpp']
        x_mode      = x_mode or 'timestamp'
        atbs_window = max(atbs_window or 5, 2)

        metrics_data = hdf5_manager.get_metrics_from_group(selected_group, atbs_window)
        if not metrics_data:
            return empty_fig(460)

        is_fmt2 = metrics_data.get('_format') == 'fmt2'

        excluded_ids = get_excluded_ids_for_group(store_excluded, selected_group)
        excl_keep    = apply_exclusion_to_metrics(metrics_data, excluded_ids)
        selected_ids = extract_selected_signal_ids(scatter_selection)
        filter_idx   = filter_by_signal_ids(metrics_data, selected_ids)

        def merge_indices(excl_keep, filter_idx):
            if excl_keep is None and filter_idx is None:
                return None
            base = set(excl_keep) if excl_keep is not None else set(range(len(metrics_data['signal_ids'])))
            if filter_idx is not None:
                base = base.intersection(filter_idx)
            result = sorted(base)
            return result if result else []

        effective_idx = merge_indices(excl_keep, filter_idx)

        try:
            base_metric = next((m for m in y_metrics if m != 'ATBS'), None)

            if x_mode == 'timestamp':
                x_data  = metrics_data['timestamp']
                x_label = 'Timestamp (s)'
            else:
                x_data  = list(range(len(metrics_data[base_metric]))) if base_metric else []
                x_label = 'Index'

            all_x, all_y = [], []
            for ym in y_metrics:
                if ym == 'ATBS':
                    xm = metrics_data['ATBS_timestamp'] if x_mode == 'timestamp' else list(range(len(metrics_data['ATBS'])))
                    yv = np.array(metrics_data['ATBS'])
                    if effective_idx is not None:
                        atbs_len  = len(metrics_data['ATBS'])
                        atbs_keep = [i for i in effective_idx if i < atbs_len]
                        xm_eff    = [xm[i] for i in atbs_keep]
                        yv_eff    = yv[atbs_keep]
                    else:
                        xm_eff, yv_eff = xm, yv
                else:
                    yv = np.array(metrics_data[ym])
                    if effective_idx is not None:
                        xm_eff = [x_data[i] for i in effective_idx]
                        yv_eff = yv[effective_idx]
                    else:
                        xm_eff, yv_eff = x_data, yv
                if normalize and len(y_metrics) > 1:
                    yv_eff = normalize_data(yv_eff)
                all_x.extend(xm_eff)
                all_y.extend(yv_eff.tolist() if hasattr(yv_eff, 'tolist') else list(yv_eff))

            x_rng = full_range(np.array(all_x)) if all_x else None
            y_rng = full_range(np.array(all_y)) if all_y else None

            fig = go.Figure()

            for idx, ym in enumerate(y_metrics):
                if ym == 'ATBS':
                    xm   = metrics_data['ATBS_timestamp'] if x_mode == 'timestamp' else list(range(len(metrics_data['ATBS'])))
                    yv   = np.array(metrics_data['ATBS'])
                    sids = metrics_data['signal_ids'][:len(metrics_data['ATBS'])]
                    if effective_idx is not None:
                        atbs_len  = len(metrics_data['ATBS'])
                        atbs_keep = [i for i in effective_idx if i < atbs_len]
                        xm   = [xm[i] for i in atbs_keep]
                        yv   = yv[atbs_keep]
                        sids = [sids[i] for i in atbs_keep]
                else:
                    xm   = x_data
                    yv   = np.array(metrics_data[ym])
                    sids = metrics_data['signal_ids']
                    if effective_idx is not None:
                        xm   = [xm[i] for i in effective_idx]
                        yv   = yv[effective_idx]
                        sids = [sids[i] for i in effective_idx]

                if normalize and len(y_metrics) > 1:
                    yv = normalize_data(yv)

                cd    = np.column_stack((np.array(sids), np.full(len(sids), selected_group)))
                color = COLORS[idx % len(COLORS)]
                marker_opacity = 0.9 if filter_idx is not None else 0

                fig.add_trace(go.Scattergl(
                    x=xm, y=yv.tolist(), mode='lines+markers',
                    marker=dict(size=3, color=color, opacity=marker_opacity),
                    line=dict(color=color, width=1),
                    customdata=cd, name=band_labels.get(ym, ym),
                    opacity=0.85, yaxis='y1',
                ))

            # Relative Humidity - fmt1 y fmt2
            rh_h = metrics_data.get('rh_humidity',   [])
            rh_t = metrics_data.get('rh_timestamps', [])
            rh_y2_range = None

            if rh_h and rh_t:
                rh_h_arr     = np.array(rh_h)
                rh_t_arr     = np.array(rh_t)
                rh_y2_range  = full_range(rh_h_arr)
                rh_t_shifted = rh_t_arr - rh_t_arr[0]

                if x_mode == 'timestamp':
                    rh_x      = rh_t_shifted.tolist()
                    rh_y_plot = rh_h_arr.tolist()
                else:
                    metrics_t = np.array(metrics_data['timestamp'])
                    rh_y_plot = np.interp(metrics_t, rh_t_shifted, rh_h_arr).tolist()
                    rh_x      = list(range(len(metrics_t)))

                fig.add_trace(go.Scattergl(
                    x=rh_x, y=rh_y_plot, mode='lines',
                    line=dict(color='#ffaa00', width=1.5, dash='dot'),
                    name='Rel. Humidity (%)', yaxis='y2', opacity=0.85,
                ))

            y_title = 'Normalized' if (normalize and len(y_metrics) > 1) else 'Value'
            layout = base_layout(
                460,
                hovermode='closest', clickmode='event+select',
                showlegend=True, dragmode='lasso',
                uirevision=selected_group,
            )
            layout['xaxis']  = dict(range=x_rng, title=x_label, **AXIS_STYLE)
            layout['yaxis']  = dict(range=y_rng, title=y_title,  **AXIS_STYLE)
            layout['yaxis2'] = dict(
                title='Humidity (%)',
                overlaying='y', side='right', showgrid=False,
                range=rh_y2_range,
                tickfont=dict(color='#ffaa00', size=9),
                title_font=dict(color='#ffaa00', size=10),
                linecolor=C['border2'], zerolinecolor=C['border2'],
            )
            fig.update_layout(layout)
            return fig

        except Exception as e:
            print(f"Error time series: {e}")
            import traceback; traceback.print_exc()
            return empty_fig(460)

    # -- SCATTER --------------------------------------------------------------
    @app.callback(
        Output('graph-4', 'figure'),
        Input('dropdown-x-axis',        'value'),
        Input('dropdown-y-axis-scatter', 'value'),
        Input('dropdown-z-axis',         'value'),
        Input('dropdown-group',          'value'),
        Input('graph-3',                 'selectedData'),
        Input('animate-mode',            'value'),
        Input('animate-step-input',      'value'),
        Input('animate-speed-input',     'value'),
        Input('store-excluded',          'data'),
    )
    def update_scatter(x_metric, y_metric, z_metric, selected_group,
                       ts_selection, animate_mode, anim_step, anim_speed,
                       store_excluded):
        if not selected_group:
            return empty_fig(460)

        x_metric   = x_metric or 'eqTime'
        y_metric   = y_metric or 'eqFreq'
        anim_step  = max(int(anim_step or 0), 0)
        anim_speed = max(int(anim_speed or 40), 5)
        do_animate = bool(animate_mode)

        metrics_data = hdf5_manager.get_metrics_from_group(selected_group)
        if not metrics_data:
            return empty_fig(460)

        excluded_ids = get_excluded_ids_for_group(store_excluded, selected_group)
        excl_keep    = apply_exclusion_to_metrics(metrics_data, excluded_ids)
        selected_ids = extract_selected_signal_ids(ts_selection)
        filter_idx   = filter_by_signal_ids(metrics_data, selected_ids)

        def effective_index(excl_keep, filter_idx, total):
            if excl_keep is None and filter_idx is None:
                return None
            base = set(excl_keep) if excl_keep is not None else set(range(total))
            if filter_idx is not None:
                base = base.intersection(filter_idx)
            result = sorted(base)
            return result if result else []

        eff_idx = effective_index(excl_keep, filter_idx, len(metrics_data['signal_ids']))

        try:
            x_vals = np.array(metrics_data[x_metric], dtype=float)
            y_vals = np.array(metrics_data[y_metric], dtype=float)
            sids   = np.array(metrics_data['signal_ids'])
            cd     = np.column_stack((sids, np.full(len(sids), selected_group)))

            if eff_idx is not None:
                x_vals = x_vals[eff_idx]
                y_vals = y_vals[eff_idx]
                cd     = cd[eff_idx]

            if z_metric:
                z_vals = np.array(metrics_data[z_metric], dtype=float)
                if eff_idx is not None:
                    z_vals = z_vals[eff_idx]
                fig = go.Figure()
                fig.add_trace(go.Scatter3d(
                    x=x_vals, y=y_vals, z=z_vals, mode='markers',
                    marker=dict(size=2.5, color=C['red'], opacity=0.7, line=dict(width=0)),
                    customdata=cd, name=f'{x_metric}/{y_metric}/{z_metric}',
                ))
                layout = base_layout(460, uirevision=selected_group)
                layout['scene'] = dict(
                    xaxis=dict(title=x_metric, backgroundcolor=C['bg'], gridcolor=C['border2'], showbackground=True),
                    yaxis=dict(title=y_metric, backgroundcolor=C['bg'], gridcolor=C['border2'], showbackground=True),
                    zaxis=dict(title=z_metric, backgroundcolor=C['bg'], gridcolor=C['border2'], showbackground=True),
                    bgcolor=C['bg'],
                )
                fig.update_layout(layout)
                return fig

            x_rng = full_range(x_vals) if len(x_vals) else None
            y_rng = full_range(y_vals) if len(y_vals) else None

            if not do_animate:
                fig = go.Figure()
                fig.add_trace(go.Scattergl(
                    x=x_vals.tolist(), y=y_vals.tolist(), mode='markers',
                    marker=dict(size=3, color=C['red'], opacity=0.65),
                    customdata=cd, name=f'{y_metric} vs {x_metric}',
                ))
                layout = base_layout(460, hovermode='closest', clickmode='event+select',
                                     dragmode='lasso', uirevision=selected_group)
                layout['xaxis'] = dict(range=x_rng, title=x_metric, **AXIS_STYLE)
                layout['yaxis'] = dict(range=y_rng, title=y_metric, **AXIS_STYLE)
                fig.update_layout(layout)
                return fig

            n    = len(x_vals)
            step = anim_step if anim_step > 0 else max(1, n // 200)
            x_list = x_vals.tolist()
            y_list = y_vals.tolist()

            frames = []
            for end in range(step, n + step, step):
                end = min(end, n)
                frames.append(go.Frame(
                    data=[go.Scatter(
                        x=x_list[:end], y=y_list[:end], mode='markers',
                        marker=dict(size=3, color=C['red'], opacity=0.65),
                        name=f'{y_metric} vs {x_metric}',
                    )],
                    name=str(end),
                ))

            fig = go.Figure(
                data=[go.Scatter(
                    x=x_list[:step], y=y_list[:step], mode='markers',
                    marker=dict(size=3, color=C['red'], opacity=0.65),
                    name=f'{y_metric} vs {x_metric}',
                )],
                frames=frames,
            )
            layout = base_layout(460, hovermode='closest', uirevision=selected_group)
            layout['xaxis'] = dict(range=x_rng, title=x_metric, **AXIS_STYLE)
            layout['yaxis'] = dict(range=y_rng, title=y_metric, **AXIS_STYLE)
            layout['updatemenus'] = [dict(
                type='buttons', showactive=False,
                x=0.0, y=1.13, xanchor='left', yanchor='top',
                bgcolor=C['panel'], bordercolor=C['border2'],
                font=dict(color=C['text'], size=11),
                buttons=[
                    dict(label='Play', method='animate',
                         args=[None, {'frame': {'duration': anim_speed, 'redraw': True},
                                      'transition': {'duration': 0},
                                      'fromcurrent': True, 'mode': 'immediate'}]),
                    dict(label='Pause', method='animate',
                         args=[[None], {'frame': {'duration': 0},
                                        'transition': {'duration': 0},
                                        'mode': 'immediate'}]),
                ],
            )]
            layout['sliders'] = [dict(
                active=0,
                currentvalue=dict(prefix='Senales: ',
                                  font=dict(color=C['text_dim'], size=10),
                                  visible=True, xanchor='center'),
                pad=dict(t=45, b=5),
                bgcolor=C['panel'], bordercolor=C['border2'],
                tickcolor=C['border2'], activebgcolor=C['cyan'],
                font=dict(color=C['text_dim'], size=9),
                steps=[
                    dict(method='animate',
                         args=[[f.name], {'frame': {'duration': 0, 'redraw': True},
                                          'transition': {'duration': 0},
                                          'mode': 'immediate'}],
                         label=f.name)
                    for f in frames
                ],
            )]
            fig.update_layout(layout)
            return fig

        except Exception as e:
            print(f"Error scatter: {e}")
            import traceback; traceback.print_exc()
            return empty_fig(460)

    # -- SENAL TEMPORAL -------------------------------------------------------
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
            layout = base_layout(340)
            layout['xaxis'] = dict(title='Time (us)', **AXIS_STYLE)
            layout['yaxis'] = dict(title='Amplitude',  **AXIS_STYLE)
            fig.update_layout(layout)
            return fig
        except Exception as e:
            print(f"Error temporal: {e}")
            return empty_fig(340)

    # -- FFT ------------------------------------------------------------------
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

            fmt = hdf5_manager.get_format(group_name)
            if fmt == 'fmt2':
                sig  = hdf5_manager.get_signal_data(group_name, signal_id)
                freq = rfftfreq(len(sig), d=1.0 / SAMPLE_RATE) / 1e6
                x_data  = freq.tolist()
                x_title = 'Frequency (MHz)'
            else:
                x_data  = list(range(len(fft)))
                x_title = 'Frequency (MHz)'

            fig = go.Figure()
            fig.add_trace(go.Scattergl(
                x=x_data, y=fft.tolist() if hasattr(fft, 'tolist') else list(fft),
                mode='lines', line=dict(color=C['red'], width=1), name='FFT',
            ))
            layout = base_layout(340)
            layout['xaxis'] = dict(title=x_title,    **AXIS_STYLE)
            layout['yaxis'] = dict(title='Magnitude', **AXIS_STYLE)
            fig.update_layout(layout)
            return fig
        except Exception as e:
            print(f"Error FFT: {e}")
            return empty_fig(340)

    # -- DOWNLOADS ------------------------------------------------------------
    @app.callback(
        Output('download-scatter-csv', 'data'),
        Input('btn-download-scatter', 'n_clicks'),
        State('dropdown-x-axis', 'value'),
        State('dropdown-y-axis-scatter', 'value'),
        State('dropdown-z-axis', 'value'),
        State('dropdown-group', 'value'),
        State('graph-3', 'selectedData'),
        State('store-excluded', 'data'),
        prevent_initial_call=True,
    )
    def download_scatter_csv(n_clicks, x_metric, y_metric, z_metric,
                             selected_group, ts_selection, store_excluded):
        if not selected_group or not x_metric or not y_metric:
            return None
        metrics_data = hdf5_manager.get_metrics_from_group(selected_group)
        if not metrics_data:
            return None
        excluded_ids = get_excluded_ids_for_group(store_excluded, selected_group)
        excl_keep    = apply_exclusion_to_metrics(metrics_data, excluded_ids)

        x_vals = np.array(metrics_data[x_metric])
        y_vals = np.array(metrics_data[y_metric])
        z_vals = np.array(metrics_data[z_metric]) if z_metric else None

        selected_ids = extract_selected_signal_ids(ts_selection)
        filter_idx   = filter_by_signal_ids(metrics_data, selected_ids)

        total = len(metrics_data['signal_ids'])
        base  = set(excl_keep) if excl_keep is not None else set(range(total))
        if filter_idx is not None:
            base = base.intersection(filter_idx)
        final_idx = sorted(base) if (excl_keep is not None or filter_idx is not None) else None

        if final_idx is not None:
            x_vals = x_vals[final_idx]
            y_vals = y_vals[final_idx]
            if z_vals is not None:
                z_vals = z_vals[final_idx]

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
        State('store-excluded', 'data'),
        prevent_initial_call=True,
    )
    def download_signals(n_clicks, selected_group, ts_selection, sc_selection,
                         store_excluded):
        if not selected_group:
            return None
        metrics_data = hdf5_manager.get_metrics_from_group(selected_group)
        if not metrics_data:
            return None

        excluded_ids = get_excluded_ids_for_group(store_excluded, selected_group)
        selected_ids = (extract_selected_signal_ids(ts_selection)
                        or extract_selected_signal_ids(sc_selection))

        all_ids = metrics_data['signal_ids']
        if selected_ids:
            signal_ids = [sid for sid in all_ids
                          if sid in selected_ids and sid not in excluded_ids]
        else:
            signal_ids = [sid for sid in all_ids if sid not in excluded_ids]

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
        Output('download-signal-names-csv', 'data'),
        Input('btn-download-signal-names', 'n_clicks'),
        State('dropdown-group', 'value'),
        State('graph-3', 'selectedData'),
        State('graph-4', 'selectedData'),
        State('store-excluded', 'data'),
        prevent_initial_call=True,
    )
    def download_signal_names(n_clicks, selected_group, ts_selection, sc_selection,
                              store_excluded):
        if not selected_group:
            return None
        metrics_data = hdf5_manager.get_metrics_from_group(selected_group)
        if not metrics_data:
            return None

        excluded_ids = get_excluded_ids_for_group(store_excluded, selected_group)
        selected_ids = (extract_selected_signal_ids(ts_selection)
                        or extract_selected_signal_ids(sc_selection))

        all_ids = metrics_data['signal_ids']
        if selected_ids:
            signal_ids = [sid for sid in all_ids
                          if sid in selected_ids and sid not in excluded_ids]
        else:
            signal_ids = [sid for sid in all_ids if sid not in excluded_ids]

        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(['signal_id'])
        for sid in signal_ids:
            w.writerow([sid])
        return dcc.send_string(buf.getvalue(), f'signal_names_{selected_group}.csv')

    # -- SELECTIONS STORE -----------------------------------------------------
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
        triggered_id       = ctx.triggered[0]['prop_id']
        current_selections = current_selections or []
        if 'btn-delete-selection' in triggered_id:
            import json
            idx_info = json.loads(triggered_id.split('.')[0])
            idx = idx_info['index']
            current_selections = [s for i, s in enumerate(current_selections) if i != idx]
            return current_selections
        if not selected_group:
            return no_update
        selected_ids = (extract_selected_signal_ids(ts_sel)
                        or extract_selected_signal_ids(sc_sel))
        if not selected_ids:
            return no_update
        name  = sel_name or f'Selection {len(current_selections) + 1}'
        color = SELECTION_COLORS[len(current_selections) % len(SELECTION_COLORS)]
        current_selections.append({
            'name':       name,
            'group':      selected_group,
            'signal_ids': list(selected_ids),
            'color':      color,
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
                html.Span('*', style={'color': sel['color'], 'fontSize': '14px', 'marginRight': '8px'}),
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
                html.Button('x', id={'type': 'btn-delete-selection', 'index': i}, style={
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

    # -- COMPARE CALLBACKS ----------------------------------------------------
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
            sids  = sel['signal_ids']
            ffts  = []
            for sid in sids:
                fft = hdf5_manager.get_fft_data(group, sid)
                if len(fft):
                    ffts.append(fft)
            if not ffts:
                continue
            max_len = max(len(f) for f in ffts)
            padded  = [np.pad(f, (0, max_len - len(f))) for f in ffts]
            avg_fft = np.mean(padded, axis=0)

            fmt = hdf5_manager.get_format(group)
            if fmt == 'fmt2' and sids:
                sig  = hdf5_manager.get_signal_data(group, sids[0])
                freq = rfftfreq(len(sig), d=1.0 / SAMPLE_RATE) / 1e6
                x_fft = freq[:len(avg_fft)].tolist()
            else:
                x_fft = list(range(len(avg_fft)))

            fig.add_trace(go.Scattergl(
                x=x_fft, y=avg_fft.tolist(), mode='lines',
                line=dict(color=sel['color'], width=1.5),
                name=sel['name'], opacity=0.85,
            ))
        layout = base_layout(380, showlegend=True)
        layout['xaxis'] = dict(title='Frequency (MHz)', **AXIS_STYLE)
        layout['yaxis'] = dict(title='Magnitude (avg)', **AXIS_STYLE)
        fig.update_layout(layout)
        return fig

    @app.callback(
        Output('graph-compare-signals', 'figure'),
        Input('store-selections', 'data'),
        Input('input-random-signals', 'value'),
    )
    def compare_random_signals(selections, n_signals):
        selections = selections or []
        n_signals  = max(int(n_signals or 10), 1)
        if not selections:
            return empty_fig(380)
        fig = go.Figure()
        for sel in selections:
            group  = sel['group']
            sids   = sel['signal_ids']
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
        layout = base_layout(380, showlegend=True)
        layout['xaxis'] = dict(title='Time (us)',  **AXIS_STYLE)
        layout['yaxis'] = dict(title='Amplitude',  **AXIS_STYLE)
        fig.update_layout(layout)
        return fig

    @app.callback(
        Output('graph-compare-fft', 'figure'),
        Input('store-selections', 'data'),
        Input('input-random-signals', 'value'),
    )
    def compare_random_fft(selections, n_signals):
        selections = selections or []
        n_signals  = max(int(n_signals or 10), 1)
        if not selections:
            return empty_fig(380)
        fig = go.Figure()
        for sel in selections:
            group  = sel['group']
            sids   = sel['signal_ids']
            sample = random.sample(sids, min(n_signals, len(sids)))
            for i, sid in enumerate(sample):
                fft = hdf5_manager.get_fft_data(group, sid)
                if not len(fft):
                    continue
                fmt = hdf5_manager.get_format(group)
                if fmt == 'fmt2':
                    sig  = hdf5_manager.get_signal_data(group, sid)
                    freq = rfftfreq(len(sig), d=1.0 / SAMPLE_RATE) / 1e6
                    x_fft = freq[:len(fft)].tolist()
                else:
                    x_fft = list(range(len(fft)))

                fig.add_trace(go.Scattergl(
                    x=x_fft, y=fft.tolist() if hasattr(fft, 'tolist') else list(fft),
                    mode='lines', line=dict(color=sel['color'], width=0.8),
                    name=sel['name'] if i == 0 else None,
                    showlegend=(i == 0), opacity=0.6,
                    legendgroup=sel['name'],
                ))
        layout = base_layout(380, showlegend=True)
        layout['xaxis'] = dict(title='Frequency (MHz)', **AXIS_STYLE)
        layout['yaxis'] = dict(title='Magnitude',        **AXIS_STYLE)
        fig.update_layout(layout)
        return fig

    @app.callback(
        Output('graph-compare-scatter', 'figure'),
        Input('store-selections', 'data'),
        Input('dropdown-compare-x', 'value'),
        Input('dropdown-compare-y', 'value'),
    )
    def compare_scatter(selections, x_metric, y_metric):
        selections = selections or []
        x_metric   = x_metric or 'eqTime'
        y_metric   = y_metric or 'eqFreq'
        if not selections:
            return empty_fig(380)
        fig = go.Figure()
        for sel in selections:
            group        = sel['group']
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
        layout = base_layout(380, showlegend=True)
        layout['xaxis'] = dict(title=x_metric, **AXIS_STYLE)
        layout['yaxis'] = dict(title=y_metric, **AXIS_STYLE)
        fig.update_layout(layout)
        return fig

    @app.callback(
        Output('graph-compare-timeseries', 'figure'),
        Input('store-selections', 'data'),
        Input('dropdown-compare-ts-metric', 'value'),
    )
    def compare_timeseries(selections, metric):
        selections = selections or []
        metric     = metric or 'vpp'
        if not selections:
            return empty_fig(380)
        fig = go.Figure()
        for sel in selections:
            group        = sel['group']
            metrics_data = hdf5_manager.get_metrics_from_group(group)
            if not metrics_data:
                continue
            filter_idx = filter_by_signal_ids(metrics_data, set(sel['signal_ids']))
            if filter_idx is None:
                continue
            timestamps = np.array(metrics_data['timestamp'])[filter_idx]
            values     = np.array(metrics_data[metric])[filter_idx]
            fig.add_trace(go.Scattergl(
                x=timestamps, y=values, mode='lines+markers',
                marker=dict(size=3, color=sel['color'], opacity=0.8),
                line=dict(color=sel['color'], width=1),
                name=sel['name'], opacity=0.85,
            ))
        layout = base_layout(380, showlegend=True)
        layout['xaxis'] = dict(title='Timestamp', **AXIS_STYLE)
        layout['yaxis'] = dict(title=metric,      **AXIS_STYLE)
        fig.update_layout(layout)
        return fig

    # -- COPY GROUP NAME (clientside) -----------------------------------------
    app.clientside_callback(
        """
        function(group_value) {
            return group_value || '';
        }
        """,
        Output('copy-group-name-value', 'children'),
        Input('dropdown-group', 'value'),
    )

    app.clientside_callback(
        """
        function(n_clicks) {
            if (!n_clicks) return window.dash_clientside.no_update;
            var span = document.getElementById('copy-group-name-value');
            var text = span ? span.innerText.trim() : '';
            if (!text) return window.dash_clientside.no_update;
            navigator.clipboard.writeText(text).then(function() {
                var btn = document.getElementById('btn-copy-group-name');
                if (!btn) return;
                var orig = btn.innerText;
                var origColor = btn.style.color;
                var origBorder = btn.style.borderColor;
                btn.innerText = '\\u2713';
                btn.style.color = '#00ff9d';
                btn.style.borderColor = '#00ff9d';
                setTimeout(function() {
                    btn.innerText = orig;
                    btn.style.color = origColor;
                    btn.style.borderColor = origBorder;
                }, 1200);
            });
            return window.dash_clientside.no_update;
        }
        """,
        Output('btn-copy-group-name', 'n_clicks'),
        Input('btn-copy-group-name', 'n_clicks'),
        prevent_initial_call=True,
    )

    @atexit.register
    def cleanup():
        if _hdf5_manager:
            _hdf5_manager.close()
