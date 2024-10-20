import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.signal import find_peaks
import os

### * -- Set page config
st.set_page_config(page_title="CFD Transient 해석", page_icon="🌊", layout="wide",   # centered, wide
                    initial_sidebar_state="expanded", # runOnSave = True,
                    menu_items = {
                        # 'Get Help': 'https://www.extremelycoolapp.com/help',
                        # 'Report a bug': "https://www.extremelycoolapp.com/bug",
                        # 'About': "# This is a header. This is an *extremely* cool app!"
                    })

def shape(fig, typ, x0,y0,x1,y1, fillcolor, color, width, **kargs):
    dash = 'solid'
    if len(kargs) > 0:  dash = kargs['LineStyle']            
    fig.add_shape(
        type=typ, x0=x0, y0=y0, x1=x1, y1=y1, fillcolor=fillcolor,
        line=dict(color=color, width=width, dash=dash, ), )  # dash = solid, dot, dash, longdash, dashdot, longdashdot, '5px 10px'
    
def load_out_files():
    out_files = {}
    current_dir = os.getcwd()
    
    for filename in os.listdir(current_dir):        
        if filename.endswith('.out'):
            file_path = os.path.join(current_dir, filename)
            try:
                # 파일 읽기 (첫 3줄은 건너뛰고, 공백으로 구분된 데이터 읽기)
                df = pd.read_csv(file_path, skiprows=3, delim_whitespace=True, 
                                names=['Time Step', 'cd', 'flow-time'])
                out_files[filename] = df
            except Exception as e:
                st.error(f"Error reading {filename}: {str(e)}")
    
    return out_files

# 모든 .out 파일 로드
out_files = load_out_files()
static_cd = {
    "soil2_0": 2.234,
    "soil2_10": 1.806,
    "soil2_20": 1.639,
    "soil7_0": 2.280,
    "soil7_10": 2.035,
    "soil7_20": 1.913,
    "soil15_0": 2.293,
    "soil15_10": 2.133,
    "soil15_20": 2.046,
    "bridge2_10": 1.622,
    "bridge2_20": 1.460,
    "bridge2_30": 1.418,
    "bridge7_10": 1.768,
    "bridge7_20": 1.519,
    "bridge7_30": 1.421,
    "bridge15_10": 1.965,
    "bridge15_20": 1.667,
    "bridge15_30": 1.526,
}

# st.title('CFD Transient 해석 결과 분석 🌊💻⏳🔄')
with st.sidebar:
    st.title("🌊 CFD Transient 해석 💻")
    st.write('#### :orange[ - Edge 브라우저 사용]')
    st.write('#### :orange[ - 다크모드 사용 [우측 상단의 Deploy 옆의 ⋮ 클릭하고 Settings에서 Theme을 Dark로 설정]]')
    st.write("---")
    analysis_target = st.radio("✨ **:green[분석할 대상을 선택하세요:]**", ('토공부 (soil)', '교량부 (bridge)'), horizontal=True, index=1)
    sound_barrier_height = st.radio("✨ **:green[방음벽 높이를 선택하세요:]**", ('2m', '7m', '15m'), horizontal=True, index=2)
    st.write("---")
    selected_stable_region = st.slider(':green[✨ **수동 안정화 구간을 선택하세요 (sec)**]', 0., 100., (20., 100.), step=0.1, format="%f")
    st.write("---")
    y_range = st.slider(':green[✨ **y축의 범위를 선택하세요**]', 0., 15., (0., 6.), step=0.1, format="%f")

# 분석 대상에 따라 포함할 키워드 설정
target_keywords = []
if 'soil' in analysis_target:
    target_keywords.append('soil')
if 'bridge' in analysis_target:
    target_keywords.append('bridge')

# 방음벽 높이 추가 ('2', '7', '15'만 사용)
target_keywords.append(sound_barrier_height.replace('m', ''))
target_keywords = ''.join(target_keywords[-2:])
filtered_files = {k: v for k, v in out_files.items() if target_keywords in k}

# 데이터 읽기 함수
# @st.cache_data
def load_data(file):
    data = pd.read_csv(file, skiprows=3, delim_whitespace=True, names=['Time Step', 'cd', 'flow-time'])
    return data

# 안정화 구간 찾기 함수
def find_stable_region(data, window_size=50, threshold=0.01):
    rolling_mean = data['cd'].rolling(window=window_size).mean()
    change_rate = np.abs(rolling_mean.diff() / rolling_mean)
    stable_start = change_rate[change_rate <= threshold].index[0]
    return data.iloc[stable_start:]

# 주기 검출 및 안정화 구간 끝 찾기 함수
def detect_periods_and_end(data, prominence=0.1, distance=10):
    peaks, _ = find_peaks(data['cd'].values, prominence=prominence, distance=distance)
    if len(peaks) < 2:
        return peaks, len(data) - 1  # 마지막 완전한 주기의 끝을 안정화 구간의 끝으로 설정    
    stable_end = peaks[-1]
    return peaks[:-1], stable_end  # 마지막 피크는 제외

def extract_height(filename):
    parts = filename.split('_')
    if len(parts) == 2:  # bridge2_20.out 형식
        return parts[-1].split('.')[0]
    elif len(parts) == 3:  # soil15_10_100.out 형식
        return parts[1]
    else:
        return "형식에 맞지 않는 파일명입니다."

label = '⏳' if 'soil' in analysis_target else '🌉'
st.write(f'## {label} :green[{analysis_target}] :blue[[방음벽 높이 : {sound_barrier_height}]]')
for idx, file in enumerate(filtered_files):
    height = extract_height(file)
    data = load_data(file) # 데이터 로드    

    # 선택된 시간 범위에 해당하는 데이터 추출
    selected_data = data[(data['flow-time'] >= selected_stable_region[0]) & (data['flow-time'] <= selected_stable_region[1])]

    col1, col2 = st.columns(2)
    label = '성토 높이' if 'soil' in analysis_target else '교량 고도'
    with col1:
        st.write(f'##### :orange[{idx+1}. {label} : {height} m]')
    with col2:
        with st.expander("원본 데이터를 보시려면 클릭하세요"):
            st.write(data)

    # 원본 데이터 전체 평균
    overall_mean = data['cd'].mean()

    # 안정화 구간 찾기
    initial_stable_data = find_stable_region(data)

    # 주기 검출 및 안정화 구간 끝 찾기
    peaks, stable_end = detect_periods_and_end(initial_stable_data)

    # 최종 안정화 구간 설정
    # stable_data = initial_stable_data.iloc[:stable_end+1]
    stable_data = initial_stable_data

    # 안정화 구간의 전체 평균
    stable_mean = stable_data['cd'].mean()
    selected_mean = selected_data['cd'].mean()

    # 주기별 평균 계산
    period_means = []
    for i in range(len(peaks)):
        start = peaks[i]
        end = peaks[i+1] if i < len(peaks) - 1 else stable_end
        period_mean = stable_data['cd'].iloc[start:end].mean()
        period_means.append(period_mean)

    # 주기별 평균의 평균
    period_mean_avg = np.mean(period_means) if period_means else None

    # 그래프 생성
    fig = go.Figure()

    # 원본 데이터 추가
    fig.add_trace(go.Scatter(
        x=data['flow-time'], y=data['cd'], mode='lines',
        name=f'전체 데이터 (평균 : {overall_mean:.4f})', line=dict(color='orange', width=2)), )

    # 안정화 구간 추가 (자동)
    fig.add_trace(go.Scatter(x=stable_data['flow-time'], y=stable_data['cd'], 
                            mode='lines', name=f'안정화 구간 (평균 : {stable_mean:.4f}) : 자동', line=dict(color='green', width=3)))
    # 안정화 구간 추가 (수동)
    fig.add_trace(go.Scatter(x=selected_data['flow-time'], y=selected_data['cd'], 
                            mode='lines', name=f'안정화 구간 (평균 : {selected_mean:.4f}) : 수동', line=dict(color='yellow', width=3)))

    # 정적해석 항력계수 결과
    # target_keywords와 height를 조합한 키 생성
    target_key = f"{target_keywords}_{height}"
    if target_key in static_cd:
        value = static_cd[target_key]
        fig.add_trace(go.Scatter(x=data['flow-time'], y=[value] * len(data), 
                    mode='lines', name=f'정적해석 결과 (static cd : {value:.4f})', line=dict(color='magenta', width=1)))

    # # 주기 시작점 표시
    # if period_mean_avg is not None:
    #     fig.add_trace(go.Scatter(
    #         x=stable_data['flow-time'].iloc[peaks],
    #         y=stable_data['cd'].iloc[peaks],
    #         mode='markers',
    #         marker=dict(size=10, color='red', symbol='circle'),
    #         name=f'주기 시작점 (평균 : {period_mean_avg:.4f})'
    # ))
    fig.update_traces(hovertemplate='time : %{x:.1f} sec<br>cd : %{y:.4f}<extra></extra>')
    shape(fig, 'rect', 0,y_range[0],data['flow-time'].max()+0.2,y_range[1], 'rgba(0,0,0,0)', 'RoyalBlue', 2)  # 그림 상자 외부 박스

    # 그래프 레이아웃 설정
    fig.update_layout(
        autosize=False, height=400, margin=dict(l=0, r=0, t=0, b=0), #hovermode='x unified',
        hoverlabel=dict(bgcolor='lightgray', font_size=15, font_color='blue'),
        legend=dict(x=0.5, y=0.95, xanchor='center', yanchor='top', font_size=15, borderwidth=2, bordercolor='green'),
        template="plotly_dark",  # 다크 테마 적용
    )
    fig.update_xaxes(
        range=[0, data['flow-time'].max()+0.2], showgrid=True, gridwidth=1, 
        showspikes=True, spikecolor="gray", spikesnap="cursor", spikemode="across", spikethickness=3,
        title = dict(text='flow-time (sec)', standoff=20, font = dict(size=17)),
        tickfont = dict(size=17), tickformat=',.0f', tickmode='linear', dtick=3, )
    fig.update_yaxes(
        # range=[0, data['cd'].max()+0.1], showgrid=True, gridwidth=1,
        range=[y_range[0], y_range[1]], showgrid=True, gridwidth=1,
        showspikes=True, spikecolor="gray", spikesnap="cursor", spikemode="across", spikethickness=3,
        title = dict(text='cd', standoff=20, font = dict(size=17)),
        tickfont = dict(size=17), tickformat=',.0f', tickmode='linear')

    # Streamlit에 그래프 표시
    st.plotly_chart(fig, use_container_width=True)

