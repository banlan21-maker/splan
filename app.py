"""
IronFlow - 조선기자재 생산 자동 스케줄링 시스템
전면 리팩토링 버전: 동적 공정 관리 + 탭 메뉴 구조
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
import random
from streamlit_option_menu import option_menu
import plotly.graph_objects as go
import plotly.express as px
import io
import json

# ============================================================================
# 상수 정의
# ============================================================================
WEEKDAYS = {
    "월": 0, "화": 1, "수": 2, "목": 3, "금": 4, "토": 5, "일": 6
}
WEEKDAY_NAMES = ["월", "화", "수", "목", "금", "토", "일"]

# 버전 정보 (업로드할 때마다 수동으로 업데이트)
APP_VERSION = "v1.01"
APP_AUTHOR = "by.banlan"
DEFAULT_APP_TITLE = "IronFlow - 조선기자재 스마트 스케줄러"
PROCESS_TYPE_LABELS = {"Duration": "기간", "Milestone": "마일스톤"}
PROCESS_TYPE_VALUES = {v: k for k, v in PROCESS_TYPE_LABELS.items()}

# ============================================================================
# 페이지 설정
# ============================================================================
st.set_page_config(
    page_title=DEFAULT_APP_TITLE,
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# Session State 초기화
# ============================================================================
def init_session_state():
    """Session State 초기화"""
    if 'company_info' not in st.session_state:
        st.session_state.company_info = {
            "company_name": "",
            "business_number": "",
            "department": "",
            "contact": ""
        }
    if 'global_holidays' not in st.session_state:
        st.session_state.global_holidays = set()
    
    if 'processes_df' not in st.session_state:
        st.session_state.processes_df = get_default_processes_df()
    
    if 'team_settings' not in st.session_state:
        st.session_state.team_settings = {}
        # 초기 팀 설정 생성
        for _, proc_row in st.session_state.processes_df.iterrows():
            team_code = proc_row['Team Code']
            if team_code not in st.session_state.team_settings:
                st.session_state.team_settings[team_code] = {
                    'work_weekdays': [0, 1, 2, 3, 4, 5],  # 기본값: 월~토
                    'team_holidays': set()
                }
    
    if 'df_raw' not in st.session_state:
        st.session_state.df_raw = None
    
    if 'projects_db' not in st.session_state:
        st.session_state.projects_db = {}  # Key: Project_No, Value: DataFrame (블록 리스트)
    
    if 'project_capa' not in st.session_state:
        st.session_state.project_capa = {}  # Key: (Project_No, Process_Name) 튜플, Value: Monthly_CAPA_Ton

def get_app_title():
    """회사명 기반 앱 타이틀 반환"""
    company_name = st.session_state.get("company_info", {}).get("company_name", "").strip()
    if company_name:
        return f"{company_name} 자동 생산스케줄 생성기"
    return DEFAULT_APP_TITLE

def apply_browser_title(title):
    """브라우저 탭 제목을 동적으로 변경"""
    st.markdown(
        f"<script>document.title = {json.dumps(title)};</script>",
        unsafe_allow_html=True
    )

# ============================================================================
# 기본 공정 데이터프레임 생성
# ============================================================================
def get_default_processes_df():
    """기본 공정 리스트를 데이터프레임으로 반환"""
    default_data = {
        'Process Name': ['절단', '취부', '용접', '사상', '조립검사', '도장', '도장검사', 'PND', '납기'],
        'Type': ['Duration', 'Duration', 'Duration', 'Duration', 'Milestone', 'Duration', 'Milestone', 'Milestone', 'Milestone'],
        'Order': [1, 2, 3, 4, 5, 6, 7, 8, 9],
        'Team Code': ['cutting', 'fitting', 'welding', 'sandblasting', 'assembly_inspection', 'painting', 'painting_inspection', 'pnd', 'final']
    }
    return pd.DataFrame(default_data)

# ============================================================================
# 유틸리티 함수
# ============================================================================
@st.cache_data
def generate_date_list(start_date=None, end_date=None):
    """날짜 리스트 생성 (YYYY-MM-DD (요일) 형식)"""
    if start_date is None:
        start_date = date.today()
    if end_date is None:
        end_2026 = date(2026, 12, 31)
        one_year_later = start_date + timedelta(days=365)
        end_date = max(end_2026, one_year_later)
    
    date_list = []
    current_date = start_date
    
    while current_date <= end_date:
        weekday_name = WEEKDAY_NAMES[current_date.weekday()]
        date_str = f"{current_date.strftime('%Y-%m-%d')} ({weekday_name})"
        date_list.append(date_str)
        current_date += timedelta(days=1)
    
    return date_list

def date_string_to_date(date_str):
    """'2026-05-05 (화)' 형식의 문자열을 date 객체로 변환"""
    date_part = date_str.split(' (')[0]
    return datetime.strptime(date_part, "%Y-%m-%d").date()

def date_to_date_string(d):
    """date 객체를 'YYYY-MM-DD (요일)' 형식의 문자열로 변환"""
    weekday_name = WEEKDAY_NAMES[d.weekday()]
    return f"{d.strftime('%Y-%m-%d')} ({weekday_name})"

# ============================================================================
# NumPy 기반 날짜 계산 함수
# ============================================================================
def work_weekdays_to_weekmask(work_weekdays):
    """근무 요일 리스트를 NumPy weekmask 형식으로 변환"""
    weekmask = ['0'] * 7
    for day in work_weekdays:
        weekmask[day] = '1'
    return ''.join(weekmask)

def holidays_to_numpy_array(global_holidays, team_holidays):
    """휴무일 세트를 NumPy 배열로 변환"""
    all_holidays = global_holidays.union(team_holidays)
    if not all_holidays:
        return np.array([], dtype='datetime64[D]')
    return np.array(sorted(all_holidays), dtype='datetime64[D]')

def is_work_day_numpy(date_np, weekmask, holidays):
    """NumPy를 사용하여 특정 날짜가 작업일인지 확인"""
    if date_np in holidays:
        return False
    weekday = pd.Timestamp(date_np).weekday()
    return weekmask[weekday] == '1'

def add_business_days_numpy(end_date, days, work_weekdays, global_holidays, team_holidays):
    """NumPy를 사용하여 주말과 휴일을 제외하고 지정된 일수만큼 날짜를 역산"""
    if days == 0:
        return end_date
    
    weekmask = work_weekdays_to_weekmask(work_weekdays)
    holidays = holidays_to_numpy_array(global_holidays, team_holidays)
    end_date_np = np.datetime64(end_date.date())
    
    current_date = end_date_np
    days_counted = 0
    max_iterations = 365 * 2
    iteration = 0
    
    while days_counted < days and iteration < max_iterations:
        current_date -= np.timedelta64(1, 'D')
        if is_work_day_numpy(current_date, weekmask, holidays):
            days_counted += 1
        iteration += 1
    
    if iteration >= max_iterations:
        raise ValueError(f"작업일을 찾을 수 없습니다. 날짜 범위를 확인하세요.")
    
    return pd.Timestamp(current_date)

# ============================================================================
# 동적 역산 스케줄링 엔진
# ============================================================================
def calculate_backward_schedule(df, processes_df, team_settings, global_holidays):
    """
    동적 역산 스케줄링 계산 엔진
    - processes_df: 공정 리스트 데이터프레임
    - team_settings: 팀별 설정 딕셔너리 (Team Code를 키로 사용)
    - global_holidays: 공통 휴무일 세트
    """
    df = df.copy()
    
    # 공정 리스트를 Order 순서대로 정렬하고 역순으로 변환
    processes_sorted = processes_df.sort_values('Order').to_dict('records')
    processes_reversed = list(reversed(processes_sorted))
    
    # 각 행에 대해 역산 스케줄링 계산
    for idx, row in df.iterrows():
        # 납기일 찾기
        final_date = pd.to_datetime(row["납기일(Final_Date)"])
        
        # 역순으로 공정 순회
        current_reference_date = final_date
        
        for process in processes_reversed:
            process_name = process['Process Name']
            process_type = process['Type']
            team_code = process['Team Code']
            
            # 납기와 PND는 특별 처리
            if process_name == '납기':
                df.at[idx, "납기일(Final_Date)"] = final_date
                continue
            elif process_name == 'PND':
                pnd_date = final_date - pd.Timedelta(days=1)
                df.at[idx, "PND"] = pnd_date
                current_reference_date = pnd_date
                continue
            
            # 팀 설정 가져오기
            team_setting = team_settings.get(team_code, {
                'work_weekdays': [0, 1, 2, 3, 4, 5],
                'team_holidays': set()
            })
            work_weekdays = team_setting.get('work_weekdays', [0, 1, 2, 3, 4, 5])
            team_holidays = team_setting.get('team_holidays', set())
            
            if process_type == 'Milestone':
                # 마일스톤: 현재 기준일의 전날에서 작업일 기준으로 1일 역산
                prev_day = current_reference_date - pd.Timedelta(days=1)
                milestone_date = add_business_days_numpy(
                    prev_day, 1, work_weekdays, global_holidays, team_holidays
                )
                df.at[idx, f"{process_name}일"] = milestone_date
                current_reference_date = milestone_date
                
            elif process_type == 'Duration':
                # Duration: 종료일 = 현재 기준일의 전날, 시작일 = 종료일에서 소요일수 역산
                days_col = f"{process_name}_Days"
                if days_col not in row or pd.isna(row[days_col]):
                    days = 5  # 기본값
                else:
                    days = int(row[days_col])
                
                end_date = current_reference_date - pd.Timedelta(days=1)
                start_date = add_business_days_numpy(
                    end_date, days, work_weekdays, global_holidays, team_holidays
                )
                
                df.at[idx, f"{process_name}_Start"] = start_date
                df.at[idx, f"{process_name}_End"] = end_date
                current_reference_date = start_date
    
    return df

# ============================================================================
# 페이지 함수들
# ============================================================================
def main_home():
    """홈 페이지"""
    app_title = get_app_title()
    st.markdown(
        f"<h3 style='text-align: left;'>{app_title}</h3>",
        unsafe_allow_html=True
    )
    
    st.divider()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("등록된 공정 수", len(st.session_state.processes_df))
    
    with col2:
        if st.session_state.df_raw is not None:
            st.metric("등록된 프로젝트 수", len(st.session_state.df_raw))
        else:
            st.metric("등록된 프로젝트 수", 0)
    
    with col3:
        total_holidays = len(st.session_state.global_holidays)
        for team_setting in st.session_state.team_settings.values():
            total_holidays += len(team_setting.get('team_holidays', set()))
        st.metric("등록된 휴무일 수", total_holidays)
    
    st.divider()
    
    st.markdown("""
    #### 📖 사용 방법
    
    1. **공정 설정(Admin)**  
       - 현장에서 실제 사용하는 공정 이름과 순서를 등록합니다.  
       - Duration(기간) / Milestone(특정일) 타입을 선택해 주세요.  
       - 팀코드를 지정하면 스케줄링에서 팀별 근무일 설정과 연동됩니다.
    
    2. **기초정보 관리**  
       - 사이드바의 **양식 다운로드**로 최신 공정이 반영된 템플릿을 받습니다.  
       - 프로젝트 기본정보(호선/블록/중량/납기일)를 입력하거나 엑셀로 업로드합니다.  
       - 각 공정의 소요기간(Days)을 입력해 실제 작업 리드타임을 반영합니다.  
       - 필요 시 **실시간 작업 수정 모드**에서 전체 데이터를 빠르게 수정합니다.
    
    3. **스케줄링 메인**  
       - 공통 휴무일과 팀별 근무 요일/휴무일을 설정합니다.  
       - **스케줄 계산**을 누르면 납기일 기준으로 자동 역산 스케줄이 생성됩니다.  
       - 결과 표와 간트 차트로 일정 흐름을 확인하고 다운로드할 수 있습니다.
    
    #### 💡 주요 기능
    
    - ✅ **사업자 정보 & 브랜딩**: 회사명을 입력하면 앱 타이틀에 자동 반영
    - ✅ **동적 공정 관리**: 공정 추가/수정/삭제가 모든 입력 양식에 즉시 반영
    - ✅ **팀별 근무 캘린더**: 팀별 근무 요일과 휴무일을 분리 관리
    - ✅ **자동 역산 스케줄**: 납기일 기준으로 공정별 시작/종료일 자동 계산
    - ✅ **간트 차트 + 부하율 분석**: 일정 시각화와 CAPA 대비 부하 확인
    """)

def page_admin():
    """공정 설정(Admin) 페이지"""
    st.markdown("### ⚙️ 공정 설정 (Admin)")
    st.caption("시스템에서 사용할 공정을 정의하고 관리합니다.")
    
    st.info("💡 공정을 추가/수정/삭제하면 다른 페이지에 즉시 반영됩니다.")

    # 공정 순서 이동(드래그 대체)
    st.markdown("#### 🧭 공정 순서 이동")
    if "process_reorder_select_target" in st.session_state:
        st.session_state.process_reorder_select = st.session_state.pop(
            "process_reorder_select_target"
        )
    processes_df = st.session_state.processes_df.sort_values('Order').reset_index(drop=True)
    if len(processes_df) > 0:
        option_labels = [
            f"{idx + 1}. {row['Process Name']} ({row['Team Code']})"
            for idx, row in processes_df.iterrows()
        ]
        selected_idx = st.selectbox(
            "이동할 공정 선택",
            options=list(range(len(option_labels))),
            format_func=lambda i: option_labels[i],
            key="process_reorder_select"
        )

        col1, col2, col3, col4 = st.columns(4)
        moved = False
        with col1:
            if st.button("⬆️ 위로", use_container_width=True):
                if selected_idx > 0:
                    processes_df.iloc[[selected_idx - 1, selected_idx]] = processes_df.iloc[
                        [selected_idx, selected_idx - 1]
                    ].values
                    selected_idx -= 1
                    moved = True
        with col2:
            if st.button("⬇️ 아래로", use_container_width=True):
                if selected_idx < len(processes_df) - 1:
                    processes_df.iloc[[selected_idx, selected_idx + 1]] = processes_df.iloc[
                        [selected_idx + 1, selected_idx]
                    ].values
                    selected_idx += 1
                    moved = True
        with col3:
            if st.button("⤒ 맨 위로", use_container_width=True):
                if selected_idx > 0:
                    row = processes_df.iloc[[selected_idx]]
                    processes_df = pd.concat(
                        [row, processes_df.drop(index=selected_idx)],
                        ignore_index=True
                    )
                    selected_idx = 0
                    moved = True
        with col4:
            if st.button("⤓ 맨 아래로", use_container_width=True):
                if selected_idx < len(processes_df) - 1:
                    row = processes_df.iloc[[selected_idx]]
                    processes_df = pd.concat(
                        [processes_df.drop(index=selected_idx), row],
                        ignore_index=True
                    )
                    selected_idx = len(processes_df) - 1
                    moved = True

        if moved:
            processes_df['Order'] = range(1, len(processes_df) + 1)
            st.session_state.processes_df = processes_df
            st.session_state.process_reorder_select_target = selected_idx
            st.success("✅ 공정 순서가 변경되었습니다!")
            st.rerun()
    
    # 공정 데이터프레임 편집 (유형 한글 표시)
    processes_display_df = st.session_state.processes_df.copy()
    processes_display_df["Type"] = processes_display_df["Type"].map(
        lambda v: PROCESS_TYPE_LABELS.get(v, v)
    )

    edited_processes_df = st.data_editor(
        processes_display_df,
        num_rows="dynamic",
        column_config={
            "Process Name": st.column_config.TextColumn(
                "공정명",
                required=True,
                help="공정 이름을 입력하세요"
            ),
            "Type": st.column_config.SelectboxColumn(
                "유형",
                options=list(PROCESS_TYPE_VALUES.keys()),
                required=True,
                help="Duration: 기간 공정, Milestone: 특정일 공정"
            ),
            "Order": st.column_config.NumberColumn(
                "순서",
                min_value=1,
                step=1,
                required=True,
                help="공정 순서 (낮을수록 먼저 실행)"
            ),
            "Team Code": st.column_config.TextColumn(
                "팀코드",
                required=True,
                help="팀 설정과 연동할 코드 (영문 소문자 권장)"
            )
        },
        hide_index=True,
        key="processes_editor"
    )

    # 한글 유형을 내부 값으로 변환
    edited_processes_df["Type"] = edited_processes_df["Type"].map(
        lambda v: PROCESS_TYPE_VALUES.get(v, v)
    )
    
    # 공정 리스트 업데이트
    if not edited_processes_df.equals(st.session_state.processes_df):
        st.session_state.processes_df = edited_processes_df.copy()
        # Order 재정렬
        st.session_state.processes_df = st.session_state.processes_df.sort_values('Order').reset_index(drop=True)
        st.session_state.processes_df['Order'] = range(1, len(st.session_state.processes_df) + 1)
        
        # 팀 설정 동적 업데이트
        for _, proc_row in st.session_state.processes_df.iterrows():
            team_code = proc_row['Team Code']
            if team_code not in st.session_state.team_settings:
                st.session_state.team_settings[team_code] = {
                    'work_weekdays': [0, 1, 2, 3, 4, 5],
                    'team_holidays': set()
                }
        
        # 사용하지 않는 팀 설정 제거
        active_team_codes = set(st.session_state.processes_df['Team Code'].tolist())
        st.session_state.team_settings = {
            k: v for k, v in st.session_state.team_settings.items() 
            if k in active_team_codes
        }
        st.success("✅ 공정 리스트가 업데이트되었습니다!")
        st.rerun()
    
    # 현재 공정 리스트 표시
    st.markdown("#### 📋 현재 공정 리스트")
    processes_list_df = st.session_state.processes_df.sort_values('Order').copy()
    processes_list_df["Type"] = processes_list_df["Type"].map(
        lambda v: PROCESS_TYPE_LABELS.get(v, v)
    )
    processes_list_df = processes_list_df.rename(columns={
        "Process Name": "공정명",
        "Type": "유형",
        "Order": "순서",
        "Team Code": "팀코드"
    })
    st.dataframe(
        processes_list_df,
        use_container_width=True,
        hide_index=True
    )

def page_input():
    """기초정보 관리 페이지 - 프로젝트 마스터 등록 및 공정 시수 입력"""
    st.markdown("### 📥 기초정보 관리")
    st.caption("프로젝트 마스터 정보를 등록하고 공정별 소요기간을 입력합니다.")

    # ========================================================================
    # 사업자 정보 설정
    # ========================================================================
    with st.expander("🏢 사업자 정보 설정 (Business Profile)", expanded=False):
        company_info = st.session_state.get("company_info", {})

        company_name = st.text_input(
            "회사명 (필수)",
            value=company_info.get("company_name", ""),
            placeholder="(주)한국야나세"
        )
        business_number = st.text_input(
            "사업자등록번호",
            value=company_info.get("business_number", "")
        )
        department = st.text_input(
            "부서명",
            value=company_info.get("department", "")
        )
        contact = st.text_input(
            "담당자 연락처",
            value=company_info.get("contact", "")
        )

        if st.button("정보 저장 및 적용", type="primary"):
            if not company_name.strip():
                st.error("회사명은 필수 입력 항목입니다.")
            else:
                st.session_state.company_info = {
                    "company_name": company_name.strip(),
                    "business_number": business_number.strip(),
                    "department": department.strip(),
                    "contact": contact.strip()
                }
                st.success("✅ 사업자 정보가 저장되었습니다!")
                st.rerun()

    # ========================================================================
    # 데이터 입력 양식 다운로드 기능 (사이드바)
    # ========================================================================
    with st.sidebar:
        st.divider()
        st.markdown("#### 📂 양식 다운로드")
        st.caption("아래 버튼을 눌러 최신 공정이 반영된 입력 양식을 받으세요.")

        # 최신 공정 설정 데이터프레임 가져오기 (버튼 클릭 시마다 최신 값 반영)
        processes_df = st.session_state.processes_df.copy()

        # Order 순서대로 정렬
        processes_df = processes_df.sort_values('Order').reset_index(drop=True)

        # 고정 컬럼
        fixed_columns = ['Project_No', 'Block_No', 'Weight', 'Delivery_Date']

        # 동적 컬럼 생성 (Order 순서대로)
        dynamic_columns = []
        for _, proc_row in processes_df.iterrows():
            process_name = proc_row['Process Name']
            process_type = proc_row['Type']

            if process_type == 'Duration':
                dynamic_columns.append(f"{process_name}_Days")
            elif process_type == 'Milestone':
                dynamic_columns.append(f"{process_name}_Date")

        # 전체 컬럼 리스트 (고정 컬럼 + 동적 컬럼)
        template_columns = fixed_columns + dynamic_columns

        # 빈 데이터프레임 생성
        template_df = pd.DataFrame(columns=template_columns)

        # 엑셀 파일 생성 (메모리)
        excel_buffer = io.BytesIO()
        with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
            template_df.to_excel(writer, index=False, sheet_name='Input_Data')
        excel_buffer.seek(0)

        # 다운로드 버튼
        st.download_button(
            label="📥 표준 입력 양식 다운로드 (.xlsx)",
            data=excel_buffer.getvalue(),
            file_name="Input_Template.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    
    tab1, tab2, tab3 = st.tabs(["1️⃣ 프로젝트 마스터 등록", "2️⃣ 공정별 소요기간 입력", "3️⃣ 실시간 작업 수정"])
    
    # ========================================================================
    # 1단계: 프로젝트 마스터 등록
    # ========================================================================
    with tab1:
        st.markdown("#### 📋 신규 프로젝트 등록")
        st.info("💡 프로젝트의 기본 정보(호선번호, 블록, 중량, 납기일)를 등록합니다.")
        
        sub_tab1, sub_tab2 = st.tabs(["📤 엑셀 업로드", "✏️ 직접 입력"])
        
        with sub_tab1:
            st.write("**엑셀 파일 업로드**")
            st.caption("💡 위 양식을 다운로드하여 작성 후 업로드하세요.")
            uploaded_file = st.file_uploader(
                "엑셀 파일을 선택하세요",
                type=['xlsx', 'xls', 'csv'],
                help="필수 컬럼: Project_No, Block_No, Weight, Delivery_Date",
                key="master_upload"
            )
            
            if uploaded_file is not None:
                try:
                    if uploaded_file.name.endswith('.csv'):
                        df = pd.read_csv(uploaded_file, encoding='utf-8-sig')
                    else:
                        df = pd.read_excel(uploaded_file)
                    
                    st.success(f"✅ 파일이 성공적으로 읽혔습니다! ({len(df)}행)")
                    
                    # 필수 컬럼 확인 및 변환
                    required_cols = ["Project_No", "Block_No", "Weight", "Delivery_Date"]
                    missing_cols = [col for col in required_cols if col not in df.columns]
                    
                    if missing_cols:
                        st.error(f"❌ 필수 컬럼이 없습니다: {', '.join(missing_cols)}")
                        st.info("필수 컬럼: Project_No (호선번호), Block_No (블록번호), Weight (중량), Delivery_Date (납기일)")
                    else:
                        # 동적 컬럼 유효성 검사 (선택적)
                        processes_df = st.session_state.processes_df
                        expected_dynamic_cols = []
                        for _, proc_row in processes_df.iterrows():
                            process_name = proc_row['Process Name']
                            process_type = proc_row['Type']
                            
                            if process_type == 'Duration':
                                expected_dynamic_cols.append(f"{process_name}_Days")
                            elif process_type == 'Milestone':
                                expected_dynamic_cols.append(f"{process_name}_Date")
                        
                        missing_dynamic_cols = [col for col in expected_dynamic_cols if col not in df.columns]
                        if missing_dynamic_cols:
                            st.warning(f"⚠️ 일부 공정 컬럼이 없습니다: {', '.join(missing_dynamic_cols)}")
                            st.info("💡 이 컬럼들은 나중에 추가할 수 있습니다. 필수 컬럼만 있어도 등록 가능합니다.")
                        
                        # Delivery_Date를 datetime으로 변환
                        df['Delivery_Date'] = pd.to_datetime(df['Delivery_Date'])
                        
                        # 데이터 미리보기 (날짜만 표시)
                        display_df = df.copy()
                        display_df['Delivery_Date'] = pd.to_datetime(display_df['Delivery_Date']).dt.date
                        st.dataframe(display_df, use_container_width=True, hide_index=True)
                        
                        # 프로젝트별로 그룹화하여 저장
                        if st.button("📥 프로젝트 등록", type="primary", use_container_width=True):
                            for project_no in df['Project_No'].unique():
                                project_df = df[df['Project_No'] == project_no].copy()
                                
                                # 기존 프로젝트 확인
                                if project_no in st.session_state.projects_db:
                                    st.warning(f"⚠️ '{project_no}' 프로젝트가 이미 존재합니다. 덮어쓰기를 진행합니다.")
                                
                                st.session_state.projects_db[project_no] = project_df
                            
                            st.success(f"✅ {len(df['Project_No'].unique())}개의 프로젝트가 등록되었습니다!")
                            st.rerun()
                            
                except Exception as e:
                    st.error(f"❌ 파일 읽기 오류: {str(e)}")
        
        with sub_tab2:
            st.write("**직접 입력**")
            
            col1, col2 = st.columns(2)
            with col1:
                project_no = st.text_input("호선번호 (Project_No)", key="input_project_no")
                block_no = st.text_input("블록번호 (Block_No)", key="input_block_no")
            with col2:
                weight = st.number_input("중량 (Weight, Ton)", min_value=0.0, value=100.0, step=0.1, key="input_weight")
                delivery_date = st.date_input("납기일 (Delivery_Date)", value=date(2026, 4, 30), key="input_delivery")
            
            if st.button("➕ 블록 추가", type="primary"):
                if not project_no or not block_no:
                    st.warning("⚠️ 호선번호와 블록번호를 입력해주세요.")
                else:
                    new_row = pd.DataFrame({
                        'Project_No': [project_no],
                        'Block_No': [block_no],
                        'Weight': [weight],
                        'Delivery_Date': [pd.to_datetime(delivery_date)]
                    })
                    
                    if project_no in st.session_state.projects_db:
                        # 기존 프로젝트에 블록 추가
                        existing_df = st.session_state.projects_db[project_no]
                        if block_no in existing_df['Block_No'].values:
                            st.warning(f"⚠️ '{block_no}' 블록이 이미 존재합니다. 업데이트합니다.")
                            existing_df = existing_df[existing_df['Block_No'] != block_no]
                        st.session_state.projects_db[project_no] = pd.concat([existing_df, new_row], ignore_index=True)
                    else:
                        # 신규 프로젝트 생성
                        st.session_state.projects_db[project_no] = new_row
                    
                    st.success(f"✅ '{project_no}' 프로젝트에 '{block_no}' 블록이 추가되었습니다!")
                    st.rerun()
        
        # 등록된 프로젝트 목록 표시
        st.divider()
        st.markdown("#### 📊 등록된 프로젝트 목록")
        
        if len(st.session_state.projects_db) == 0:
            st.info("등록된 프로젝트가 없습니다.")
        else:
            for project_no, project_df in st.session_state.projects_db.items():
                with st.expander(f"📁 {project_no} ({len(project_df)}개 블록)", expanded=False):
                    # 날짜만 표시 (시간 제거)
                    display_df = project_df.copy()
                    if 'Delivery_Date' in display_df.columns:
                        display_df['Delivery_Date'] = pd.to_datetime(display_df['Delivery_Date']).dt.date
                    st.dataframe(display_df, use_container_width=True, hide_index=True)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button(f"🗑️ 삭제", key=f"delete_{project_no}"):
                            del st.session_state.projects_db[project_no]
                            st.success(f"✅ '{project_no}' 프로젝트가 삭제되었습니다!")
                            st.rerun()
    
    # ========================================================================
    # 2단계: 공정별 소요기간 입력
    # ========================================================================
    with tab2:
        st.markdown("#### ⏱️ 공정별 소요기간 (Lead Time) 계획 수립")
        st.info("💡 등록된 프로젝트를 선택하여 각 블록별/공정별 소요 일수를 입력합니다.")
        
        if len(st.session_state.projects_db) == 0:
            st.warning("⚠️ 먼저 [1단계]에서 프로젝트를 등록해주세요.")
        else:
            # 프로젝트 선택
            project_options = list(st.session_state.projects_db.keys())
            selected_project = st.selectbox(
                "프로젝트 선택",
                options=project_options,
                key="selected_project_for_leadtime"
            )
            
            if selected_project:
                # 선택한 프로젝트의 블록 리스트 가져오기
                project_df = st.session_state.projects_db[selected_project].copy()
                
                # Duration 타입 공정만 필터링
                processes_df = st.session_state.processes_df
                duration_processes = processes_df[processes_df['Type'] == 'Duration'].sort_values('Order')
                
                # Duration 공정의 Days 컬럼 추가 (없으면 기본값 5)
                for _, proc_row in duration_processes.iterrows():
                    process_name = proc_row['Process Name']
                    days_col = f"{process_name}_Days"
                    if days_col not in project_df.columns:
                        project_df[days_col] = 5
                
                st.write(f"**프로젝트: {selected_project}** ({len(project_df)}개 블록)")
                
                # 일괄 적용 기능
                with st.expander("🔧 일괄 적용", expanded=False):
                    st.write("모든 블록에 동일한 소요일수를 일괄 적용합니다.")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        selected_process = st.selectbox(
                            "공정 선택",
                            options=duration_processes['Process Name'].tolist(),
                            key="batch_process"
                        )
                    
                    with col2:
                        batch_days = st.number_input(
                            "소요 일수",
                            min_value=1,
                            max_value=100,
                            value=5,
                            step=1,
                            key="batch_days"
                        )
                    
                    if st.button("✅ 일괄 적용", type="primary"):
                        days_col = f"{selected_process}_Days"
                        project_df[days_col] = batch_days
                        st.session_state.projects_db[selected_project] = project_df
                        st.success(f"✅ 모든 블록의 '{selected_process}' 공정을 {batch_days}일로 설정했습니다!")
                        st.rerun()
                
                # 데이터 에디터
                st.write("**블록별/공정별 소요일수 입력**")
                
                # 컬럼 구성: 기본 컬럼 + Duration 공정 Days 컬럼
                display_columns = ['Project_No', 'Block_No', 'Weight', 'Delivery_Date']
                for _, proc_row in duration_processes.iterrows():
                    process_name = proc_row['Process Name']
                    display_columns.append(f"{process_name}_Days")
                
                # 컬럼 설정 딕셔너리 생성
                column_config_dict = {
                    "Project_No": st.column_config.TextColumn("호선번호", disabled=True),
                    "Block_No": st.column_config.TextColumn("블록번호", disabled=True),
                    "Weight": st.column_config.NumberColumn("중량(Ton)", disabled=True),
                    "Delivery_Date": st.column_config.DateColumn("납기일", disabled=True),
                }
                for _, proc_row in duration_processes.iterrows():
                    process_name = proc_row['Process Name']
                    days_col = f"{process_name}_Days"
                    column_config_dict[days_col] = st.column_config.NumberColumn(
                        f"{process_name} (일)",
                        min_value=1,
                        max_value=100,
                        step=1
                    )
                
                edited_df = st.data_editor(
                    project_df[display_columns],
                    num_rows="fixed",
                    column_config=column_config_dict,
                    use_container_width=True,
                    hide_index=True,
                    key=f"leadtime_editor_{selected_project}"
                )
                
                # 저장 버튼
                if st.button("💾 소요기간 저장", type="primary", use_container_width=True):
                    # 원본 데이터프레임에 Days 컬럼 업데이트
                    for col in edited_df.columns:
                        if col.endswith('_Days'):
                            project_df[col] = edited_df[col]
                    
                    st.session_state.projects_db[selected_project] = project_df
                    st.success("✅ 소요기간이 저장되었습니다!")
    
    # ========================================================================
    # 프로젝트별 월 CAPA(생산능력) 설정
    # ========================================================================
    st.divider()
    st.markdown("#### 📊 프로젝트별 월 목표 생산량(CAPA) 설정")
    st.info("💡 각 프로젝트별로 공정별 월간 목표 생산량을 설정합니다. 이 값은 부하율 계산의 기준선으로 사용됩니다.")
    
    if len(st.session_state.projects_db) == 0:
        st.warning("⚠️ 먼저 [1단계]에서 프로젝트를 등록해주세요.")
    else:
        # 프로젝트 선택
        project_options = list(st.session_state.projects_db.keys())
        selected_project_capa = st.selectbox(
            "프로젝트 선택",
            options=project_options,
            key="selected_project_for_capa"
        )
        
        if selected_project_capa:
            # 공정 리스트 가져오기 (Order 순서대로)
            processes_df = st.session_state.processes_df.sort_values('Order').reset_index(drop=True)
            
            # CAPA 데이터프레임 생성
            capa_data = []
            for _, proc_row in processes_df.iterrows():
                process_name = proc_row['Process Name']
                # 기존 값이 있으면 가져오기, 없으면 0
                capa_key = (selected_project_capa, process_name)
                existing_capa = st.session_state.project_capa.get(capa_key, 0)
                
                capa_data.append({
                    'Process Name': process_name,
                    'Monthly CAPA (Ton)': existing_capa
                })
            
            capa_df = pd.DataFrame(capa_data)
            
            # 전체 공정 동일값 적용 기능
            with st.expander("🔧 전체 공정 동일값 적용", expanded=False):
                col1, col2 = st.columns([2, 1])
                with col1:
                    uniform_capa = st.number_input(
                        "월간 CAPA (Ton)",
                        min_value=0.0,
                        value=0.0,
                        step=10.0,
                        key="uniform_capa_input"
                    )
                with col2:
                    st.write("")  # 공간 확보
                    st.write("")  # 공간 확보
                    if st.button("✅ 전체 적용", key="apply_uniform_capa"):
                        capa_df['Monthly CAPA (Ton)'] = uniform_capa
                        st.success(f"✅ 모든 공정의 CAPA를 {uniform_capa} Ton으로 설정했습니다!")
                        st.rerun()
            
            # 데이터 에디터 설정
            column_config_dict = {
                "Process Name": st.column_config.TextColumn("공정명", disabled=True),
                "Monthly CAPA (Ton)": st.column_config.NumberColumn(
                    "월간 목표 생산량 (Ton)",
                    min_value=0.0,
                    step=10.0,
                    format="%.1f"
                )
            }
            
            # 데이터 에디터
            edited_capa_df = st.data_editor(
                capa_df,
                num_rows="fixed",
                column_config=column_config_dict,
                use_container_width=True,
                hide_index=True,
                key=f"capa_editor_{selected_project_capa}"
            )
            
            # 저장 버튼
            if st.button("💾 CAPA 정보 저장", type="primary", use_container_width=True):
                # session_state에 저장
                for _, row in edited_capa_df.iterrows():
                    process_name = row['Process Name']
                    monthly_capa = row['Monthly CAPA (Ton)']
                    
                    # NaN 체크 및 기본값 처리
                    if pd.isna(monthly_capa):
                        monthly_capa = 0.0
                    else:
                        monthly_capa = float(monthly_capa)
                    
                    capa_key = (selected_project_capa, process_name)
                    st.session_state.project_capa[capa_key] = monthly_capa
                
                st.success(f"✅ '{selected_project_capa}' 프로젝트의 CAPA 정보가 저장되었습니다!")
                
                # 저장된 CAPA 정보 요약 표시
                st.info(f"📋 저장된 CAPA 정보: {len([k for k in st.session_state.project_capa.keys() if k[0] == selected_project_capa])}개 공정")
    
    # ========================================================================
    # 3단계: 실시간 작업 수정 모드
    # ========================================================================
    with tab3:
        st.markdown("#### ✏️ 실시간 작업 수정 모드")
        st.info("💡 등록된 데이터를 엑셀처럼 편집하여 소요기간을 실시간으로 수정할 수 있습니다.")
        
        if len(st.session_state.projects_db) == 0:
            st.warning("⚠️ 먼저 [1단계]에서 프로젝트를 등록해주세요.")
        else:
            # 모든 프로젝트 데이터 통합
            all_projects_data = []
            for project_no, project_df in st.session_state.projects_db.items():
                all_projects_data.append(project_df.copy())
            
            if not all_projects_data:
                st.warning("⚠️ 등록된 데이터가 없습니다.")
            else:
                combined_df = pd.concat(all_projects_data, ignore_index=True)
                
                # Duration 공정의 Days 컬럼 추가 (없으면 기본값 5)
                processes_df = st.session_state.processes_df
                duration_processes = processes_df[processes_df['Type'] == 'Duration'].sort_values('Order')
                for _, proc_row in duration_processes.iterrows():
                    process_name = proc_row['Process Name']
                    days_col = f"{process_name}_Days"
                    if days_col not in combined_df.columns:
                        combined_df[days_col] = 5
                
                # ====================================================================
                # 필터링 섹션
                # ====================================================================
                st.markdown("### 🔍 필터 설정")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    # 프로젝트 선택
                    project_options = ["전체"] + sorted(combined_df['Project_No'].unique().tolist())
                    selected_project_filter = st.selectbox(
                        "프로젝트 선택",
                        options=project_options,
                        key="realtime_project_filter"
                    )
                
                with col2:
                    # 공정(팀) 선택
                    process_options = ["전체"] + duration_processes['Process Name'].tolist()
                    selected_process_filter = st.selectbox(
                        "공정(팀) 선택",
                        options=process_options,
                        key="realtime_process_filter"
                    )
                
                with col3:
                    # 블록 검색
                    block_search = st.text_input(
                        "블록 검색",
                        placeholder="블록명 입력 (예: A-101)",
                        key="realtime_block_search"
                    )
                
                # 필터링 적용
                filtered_df = combined_df.copy()
                
                # 프로젝트 필터
                if selected_project_filter != "전체":
                    filtered_df = filtered_df[filtered_df['Project_No'] == selected_project_filter]
                
                # 블록 검색 필터
                if block_search:
                    # Block_No를 문자열로 변환 후 검색
                    filtered_df = filtered_df[
                        filtered_df['Block_No'].astype(str).str.contains(block_search, case=False, na=False)
                    ]
                
                # 공정 필터에 따라 표시할 컬럼 결정
                display_columns = ['Project_No', 'Block_No', 'Weight', 'Delivery_Date']
                
                if selected_process_filter != "전체":
                    # 선택한 공정의 Days 컬럼만 추가
                    selected_days_col = f"{selected_process_filter}_Days"
                    if selected_days_col in filtered_df.columns:
                        display_columns.append(selected_days_col)
                else:
                    # 모든 Duration 공정의 Days 컬럼 추가
                    for _, proc_row in duration_processes.iterrows():
                        process_name = proc_row['Process Name']
                        days_col = f"{process_name}_Days"
                        if days_col in filtered_df.columns:
                            display_columns.append(days_col)
                
                # 존재하는 컬럼만 선택
                display_columns = [col for col in display_columns if col in filtered_df.columns]
                filtered_df = filtered_df[display_columns]
                
                if len(filtered_df) == 0:
                    st.warning("⚠️ 필터 조건에 맞는 데이터가 없습니다.")
                else:
                    st.success(f"✅ {len(filtered_df)}개 블록이 표시됩니다.")
                    
                    # ====================================================================
                    # 데이터 에디터 설정
                    # ====================================================================
                    column_config_dict = {
                        "Project_No": st.column_config.TextColumn("호선번호", disabled=True),
                        "Block_No": st.column_config.TextColumn("블록번호", disabled=True),
                        "Weight": st.column_config.NumberColumn("중량(Ton)", disabled=True),
                        "Delivery_Date": st.column_config.DateColumn("납기일", disabled=True),
                    }
                    
                    # 편집 가능한 Days 컬럼 설정
                    for col in display_columns:
                        if col.endswith('_Days'):
                            process_name = col.replace('_Days', '')
                            column_config_dict[col] = st.column_config.NumberColumn(
                                f"{process_name} (일)",
                                min_value=1,
                                max_value=100,
                                step=1
                            )
                    
                    # 데이터 에디터
                    edited_df = st.data_editor(
                        filtered_df,
                        num_rows="fixed",
                        column_config=column_config_dict,
                        use_container_width=True,
                        hide_index=True,
                        key="realtime_editor"
                    )
                    
                    # ====================================================================
                    # 변경사항 저장 및 스케줄 재계산
                    # ====================================================================
                    col1, col2 = st.columns([1, 1])
                    
                    with col1:
                        if st.button("💾 변경사항 저장", type="primary", use_container_width=True):
                            # 원본 데이터에 변경사항 반영
                            changes_made = False
                            
                            # edited_df와 filtered_df를 비교하여 변경사항 확인
                            for idx in edited_df.index:
                                edited_row = edited_df.loc[idx]
                                
                                # 프로젝트와 블록으로 원본 데이터 찾기
                                project_no = edited_row['Project_No']
                                block_no = edited_row['Block_No']
                                
                                if project_no in st.session_state.projects_db:
                                    project_df = st.session_state.projects_db[project_no]
                                    
                                    # 해당 블록 찾기
                                    block_mask = project_df['Block_No'] == block_no
                                    if block_mask.any():
                                        # Days 컬럼 업데이트
                                        for col in edited_df.columns:
                                            if col.endswith('_Days'):
                                                if col in project_df.columns:
                                                    old_value = project_df.loc[block_mask, col].iloc[0]
                                                    new_value = edited_row[col]
                                                    if pd.notna(new_value) and pd.notna(old_value):
                                                        if float(old_value) != float(new_value):
                                                            project_df.loc[block_mask, col] = int(new_value)
                                                            changes_made = True
                                                    elif pd.notna(new_value) and pd.isna(old_value):
                                                        project_df.loc[block_mask, col] = int(new_value)
                                                        changes_made = True
                                        
                                        st.session_state.projects_db[project_no] = project_df
                            
                            if changes_made:
                                st.success("✅ 변경사항이 저장되었습니다!")
                                st.rerun()
                            else:
                                st.info("ℹ️ 변경된 내용이 없습니다.")
                    
                    with col2:
                        if st.button("🔄 스케줄 재계산", type="secondary", use_container_width=True):
                            # 변경사항 먼저 저장
                            for idx in edited_df.index:
                                edited_row = edited_df.loc[idx]
                                project_no = edited_row['Project_No']
                                block_no = edited_row['Block_No']
                                
                                if project_no in st.session_state.projects_db:
                                    project_df = st.session_state.projects_db[project_no]
                                    block_mask = project_df['Block_No'] == block_no
                                    
                                    if block_mask.any():
                                        for col in edited_df.columns:
                                            if col.endswith('_Days'):
                                                if col in project_df.columns:
                                                    new_value = edited_row[col]
                                                    if pd.notna(new_value):
                                                        project_df.loc[block_mask, col] = int(new_value)
                                        
                                        st.session_state.projects_db[project_no] = project_df
                            
                            # 데이터 통합 (스케줄링을 위해)
                            all_data = []
                            for proj_no, proj_df in st.session_state.projects_db.items():
                                merged_df = proj_df.copy()
                                merged_df['프로젝트명'] = merged_df['Project_No']
                                merged_df['블록명'] = merged_df['Block_No']
                                merged_df['중량(Ton)'] = merged_df['Weight']
                                merged_df['납기일(Final_Date)'] = merged_df['Delivery_Date']
                                all_data.append(merged_df)
                            
                            if all_data:
                                final_df = pd.concat(all_data, ignore_index=True)
                                
                                # Duration 공정의 Days 컬럼이 없으면 기본값 추가
                                for _, proc_row in processes_df.iterrows():
                                    process_name = proc_row['Process Name']
                                    process_type = proc_row['Type']
                                    
                                    if process_type == 'Duration':
                                        days_col = f"{process_name}_Days"
                                        if days_col not in final_df.columns:
                                            final_df[days_col] = 5
                                
                                # 최종 컬럼 선택
                                final_columns = ['프로젝트명', '블록명', '중량(Ton)', '납기일(Final_Date)']
                                for _, proc_row in processes_df.iterrows():
                                    process_name = proc_row['Process Name']
                                    process_type = proc_row['Type']
                                    
                                    if process_type == 'Duration':
                                        days_col = f"{process_name}_Days"
                                        if days_col in final_df.columns:
                                            final_columns.append(days_col)
                                
                                final_df = final_df[final_columns]
                                st.session_state.df_raw = final_df
                            
                            st.success("✅ 변경사항이 저장되었습니다! [스케줄링 메인] 탭에서 재계산하세요.")
    
    # ========================================================================
    # 데이터 통합 및 최종 데이터프레임 생성
    # ========================================================================
    st.divider()
    st.markdown("#### 🔄 최종 계산용 데이터 통합")
    
    if len(st.session_state.projects_db) == 0:
        st.info("등록된 프로젝트가 없습니다.")
    else:
        # 모든 프로젝트의 데이터를 통합
        all_data = []
        for project_no, project_df in st.session_state.projects_db.items():
            # 컬럼명 변환 (스케줄링 엔진 호환)
            merged_df = project_df.copy()
            merged_df['프로젝트명'] = merged_df['Project_No']
            merged_df['블록명'] = merged_df['Block_No']
            merged_df['중량(Ton)'] = merged_df['Weight']
            merged_df['납기일(Final_Date)'] = merged_df['Delivery_Date']
            
            all_data.append(merged_df)
        
        if all_data:
            # 통합 데이터프레임 생성
            final_df = pd.concat(all_data, ignore_index=True)
            
            # Duration 공정의 Days 컬럼이 없으면 기본값 추가
            processes_df = st.session_state.processes_df
            for _, proc_row in processes_df.iterrows():
                process_name = proc_row['Process Name']
                process_type = proc_row['Type']
                
                if process_type == 'Duration':
                    days_col = f"{process_name}_Days"
                    if days_col not in final_df.columns:
                        final_df[days_col] = 5
            
            # 최종 컬럼 선택 (스케줄링 엔진에 필요한 컬럼만)
            final_columns = ['프로젝트명', '블록명', '중량(Ton)', '납기일(Final_Date)']
            for _, proc_row in processes_df.iterrows():
                process_name = proc_row['Process Name']
                process_type = proc_row['Type']
                
                if process_type == 'Duration':
                    days_col = f"{process_name}_Days"
                    if days_col in final_df.columns:
                        final_columns.append(days_col)
            
            final_df = final_df[final_columns]
            st.session_state.df_raw = final_df
            
            st.success(f"✅ {len(final_df)}개 블록의 데이터가 통합되었습니다!")
            # 날짜만 표시 (시간 제거)
            display_final_df = final_df.copy()
            if '납기일(Final_Date)' in display_final_df.columns:
                display_final_df['납기일(Final_Date)'] = pd.to_datetime(display_final_df['납기일(Final_Date)']).dt.date
            st.dataframe(display_final_df, use_container_width=True, hide_index=True)
            
            # 다운로드 버튼
            csv = final_df.to_csv(index=False, encoding='utf-8-sig')
            st.download_button(
                label="📥 통합 데이터 다운로드 (CSV)",
                data=csv,
                file_name=f"통합데이터_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )

def page_schedule():
    """스케줄링 메인 페이지"""
    st.markdown("### 📅 스케줄링 메인")
    st.caption("휴무일을 설정하고 역산 스케줄을 계산합니다.")
    
    # 데이터 확인
    if st.session_state.df_raw is None:
        st.warning("⚠️ 먼저 [기초정보 관리] 페이지에서 데이터를 업로드하거나 생성해주세요.")
        return
    
    # 사이드바: 휴무일 설정
    with st.sidebar:
        st.header("📅 휴무일 설정")
        
        # 공통 휴무일
        st.markdown("#### 🌐 공통 휴무일")
        date_list = generate_date_list()
        
        current_global_holidays_str = [
            date_to_date_string(d) for d in sorted(st.session_state.global_holidays)
        ]
        
        selected_global_holidays = st.multiselect(
            "공통 휴무일 선택",
            options=date_list,
            default=current_global_holidays_str,
            key="global_holidays_multiselect"
        )
        
        selected_global_holidays_set = {
            date_string_to_date(date_str) for date_str in selected_global_holidays
        }
        st.session_state.global_holidays = selected_global_holidays_set
        
        st.divider()
        
        # 팀별 휴무일 설정
        st.markdown("#### 👥 팀별 설정")
        
        # Team Code와 Process Name 매핑
        team_options = ["팀 선택"]
        team_code_to_name = {}
        for _, proc_row in st.session_state.processes_df.iterrows():
            team_code = proc_row['Team Code']
            process_name = proc_row['Process Name']
            team_code_to_name[team_code] = process_name
            if team_code not in ['pnd', 'final']:  # PND와 납기는 제외
                team_options.append(f"{process_name} ({team_code})")
        
        selected_team_option = st.selectbox(
            "설정할 팀 선택",
            options=team_options,
            key="selected_team"
        )
        
        if selected_team_option != "팀 선택":
            team_code = selected_team_option.split(' (')[1].rstrip(')')
            process_name = team_code_to_name.get(team_code, team_code)
            
            if team_code not in st.session_state.team_settings:
                st.session_state.team_settings[team_code] = {
                    'work_weekdays': [0, 1, 2, 3, 4, 5],
                    'team_holidays': set()
                }
            
            st.markdown(f"### {process_name}팀 설정")
            
            # 근무 요일 설정
            current_workdays = st.session_state.team_settings[team_code]['work_weekdays']
            default_selected = [WEEKDAY_NAMES[i] for i in current_workdays]
            
            selected_workdays = st.multiselect(
                "근무 요일 선택",
                options=WEEKDAY_NAMES,
                default=default_selected,
                key=f"workday_{team_code}"
            )
            
            work_weekdays_nums = [WEEKDAYS[day] for day in selected_workdays]
            st.session_state.team_settings[team_code]['work_weekdays'] = work_weekdays_nums
            
            # 팀별 휴무일 설정
            current_team_holidays = st.session_state.team_settings[team_code]['team_holidays']
            current_team_holidays_str = [
                date_to_date_string(d) for d in sorted(current_team_holidays)
            ]
            
            selected_team_holidays = st.multiselect(
                f"{process_name}팀 전용 휴무일",
                options=date_list,
                default=current_team_holidays_str,
                key=f"team_holidays_multiselect_{team_code}"
            )
            
            selected_team_holidays_set = {
                date_string_to_date(date_str) for date_str in selected_team_holidays
            }
            st.session_state.team_settings[team_code]['team_holidays'] = selected_team_holidays_set
    
    # 메인 화면: 스케줄링 계산
    if st.button("🚀 스케줄 계산", type="primary", use_container_width=True):
        with st.spinner("스케줄을 계산하는 중..."):
            df_scheduled = calculate_backward_schedule(
                st.session_state.df_raw,
                st.session_state.processes_df,
                st.session_state.team_settings,
                st.session_state.global_holidays
            )
            
            st.session_state.df_scheduled = df_scheduled
            st.success("✅ 스케줄 계산이 완료되었습니다!")
    
    # 결과 표시
    if 'df_scheduled' in st.session_state and st.session_state.df_scheduled is not None:
        df_scheduled = st.session_state.df_scheduled.copy()
        
        # 1. _Days로 끝나는 컬럼 제외 (입력 시수는 결과표에서 숨김)
        columns_to_keep = [col for col in df_scheduled.columns if not col.endswith("_Days")]
        df_scheduled = df_scheduled[columns_to_keep]
        
        # 날짜 포맷팅 (MM-DD)
        date_columns = [col for col in df_scheduled.columns 
                       if col.endswith("_Start") or col.endswith("_End") 
                       or col.endswith("일") or col == "PND" or col == "납기일(Final_Date)"]
        for col in date_columns:
            if col in df_scheduled.columns:
                df_scheduled[col] = pd.to_datetime(df_scheduled[col]).dt.strftime("%m-%d")
        
        # 2. 컬럼 순서 재정렬 (깔끔한 결과표)
        # 2-1. 기본 정보
        column_order = ["프로젝트명", "블록명", "중량(Ton)", "납기일(Final_Date)"]
        
        # 2-2. 공정 순서(Order)에 따라 결과 날짜만 표시
        processes_sorted = st.session_state.processes_df.sort_values('Order').to_dict('records')
        
        for process in processes_sorted:
            process_name = process['Process Name']
            process_type = process['Type']
            
            # PND와 납기는 별도 처리
            if process_name == 'PND':
                continue  # PND는 맨 뒤에 추가
            elif process_name == '납기':
                continue  # 납기일은 이미 기본 정보에 포함
            
            # Duration: Start, End만 표시 (Days는 제외됨)
            elif process_type == 'Duration':
                if f"{process_name}_Start" in df_scheduled.columns:
                    column_order.append(f"{process_name}_Start")
                if f"{process_name}_End" in df_scheduled.columns:
                    column_order.append(f"{process_name}_End")
            
            # Milestone: Date만 표시
            elif process_type == 'Milestone':
                if f"{process_name}일" in df_scheduled.columns:
                    column_order.append(f"{process_name}일")
        
        # 2-3. PND는 맨 뒤에 배치
        if "PND" in df_scheduled.columns:
            column_order.append("PND")
        
        # 존재하는 컬럼만 선택
        existing_columns = [col for col in column_order if col in df_scheduled.columns]
        remaining_columns = [col for col in df_scheduled.columns if col not in existing_columns]
        df_display = df_scheduled[existing_columns + remaining_columns]
        
        # 결과 테이블
        st.markdown("#### 📊 스케줄 결과")
        st.dataframe(df_display, use_container_width=True, hide_index=True)
        
        # 엑셀 다운로드 버튼 (동일한 형식으로)
        excel_buffer = io.BytesIO()
        with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
            df_display.to_excel(writer, index=False, sheet_name='Schedule_Result')
        excel_buffer.seek(0)
        
        st.download_button(
            label="📥 스케줄 결과 다운로드 (.xlsx)",
            data=excel_buffer.getvalue(),
            file_name=f"스케줄결과_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
        
        # 간트 차트
        st.markdown("#### 📈 간트 차트")
        
        # 간트 차트 데이터 준비
        gantt_data = []
        for idx, row in st.session_state.df_scheduled.iterrows():
            project_name = row.get("프로젝트명", f"프로젝트{idx}")
            block_name = row.get("블록명", f"블록{idx}")
            
            for process in processes_sorted:
                process_name = process['Process Name']
                process_type = process['Type']
                
                if process_name in ['PND', '납기']:
                    continue
                
                if process_type == 'Duration':
                    start_col = f"{process_name}_Start"
                    end_col = f"{process_name}_End"
                    if start_col in row and end_col in row:
                        start_date = pd.to_datetime(row[start_col])
                        end_date = pd.to_datetime(row[end_col])
                        gantt_data.append({
                            'Task': f"{project_name}-{block_name}",
                            'Process': process_name,
                            'Start': start_date,
                            'Finish': end_date,
                            'Duration': (end_date - start_date).days + 1
                        })
                elif process_type == 'Milestone':
                    milestone_col = f"{process_name}일"
                    if milestone_col in row:
                        milestone_date = pd.to_datetime(row[milestone_col])
                        gantt_data.append({
                            'Task': f"{project_name}-{block_name}",
                            'Process': process_name,
                            'Start': milestone_date,
                            'Finish': milestone_date,
                            'Duration': 1
                        })
        
        if gantt_data:
            gantt_df = pd.DataFrame(gantt_data)
            
            # 날짜 범위 계산 (주말 음영 처리를 위해)
            all_dates = []
            for _, row in gantt_df.iterrows():
                all_dates.append(row['Start'])
                all_dates.append(row['Finish'])
            min_date = pd.to_datetime(min(all_dates)).date()
            max_date = pd.to_datetime(max(all_dates)).date()
            
            # Plotly 간트 차트 생성
            fig = px.timeline(
                gantt_df,
                x_start='Start',
                x_end='Finish',
                y='Task',
                color='Process',
                title='생산 스케줄 간트 차트'
            )
            
            # 1. 격자선 및 배경 강화
            # X축: 매주 월요일마다 진한 회색 세로선
            # 먼저 첫 번째 월요일 찾기
            current_date = min_date
            while current_date.weekday() != 0:  # 0 = 월요일
                current_date += timedelta(days=1)
            
            # 모든 월요일에 세로선 추가
            monday_dates = []
            while current_date <= max_date:
                monday_dates.append(pd.Timestamp(current_date))
                current_date += timedelta(days=7)
            
            # 주말 음영 처리 (토요일과 일요일)
            current_date = min_date
            while current_date <= max_date:
                weekday = current_date.weekday()
                if weekday == 5:  # 토요일
                    saturday = pd.Timestamp(current_date)
                    sunday = saturday + pd.Timedelta(days=1)
                    fig.add_vrect(
                        x0=saturday,
                        x1=sunday + pd.Timedelta(days=1),
                        fillcolor="lightgray",
                        opacity=0.2,
                        layer="below",
                        line_width=0
                    )
                current_date += timedelta(days=1)
            
            # 2. 막대 디자인 개선
            # 막대에 테두리 추가 및 스타일 개선
            fig.update_traces(
                marker_line_width=1,
                marker_line_color='darkgray',
                selector=dict(type='bar')
            )
            
            # 3. X축 설정 (1주일 간격, 날짜+요일 형식)
            # 월요일 날짜 리스트 생성 (라벨용)
            monday_labels = []
            monday_positions = []
            current_date = min_date
            while current_date.weekday() != 0:
                current_date += timedelta(days=1)
            
            while current_date <= max_date:
                monday_positions.append(pd.Timestamp(current_date))
                # "05-01(월)" 형식으로 라벨 생성
                weekday_name = WEEKDAY_NAMES[current_date.weekday()]
                label = f"{current_date.strftime('%m-%d')}({weekday_name})"
                monday_labels.append(label)
                current_date += timedelta(days=7)
            
            fig.update_xaxes(
                tickmode='array',
                tickvals=monday_positions,
                ticktext=monday_labels,
                tickangle=-45,
                showgrid=True,
                gridwidth=2,
                gridcolor='darkgray',
                showline=True,
                linewidth=2,
                linecolor='black',
                rangeslider_visible=True,  # 4. Range Slider 추가
                rangeslider_thickness=0.1
            )
            
            # Y축 설정 (가로선 추가)
            fig.update_yaxes(
                autorange="reversed",
                showgrid=True,
                gridwidth=1,
                gridcolor='lightgray',
                showline=True,
                linewidth=1,
                linecolor='black'
            )
            
            # 4. 레이아웃 개선
            fig.update_layout(
                height=600,
                xaxis_title="날짜",
                yaxis_title="프로젝트-블록",
                plot_bgcolor='white',
                paper_bgcolor='white',
                hovermode='closest',
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("간트 차트를 생성할 데이터가 없습니다.")
        
        # ========================================================================
        # 공정 부하율 분석
        # ========================================================================
        st.divider()
        st.markdown("#### 📊 공정 부하율 분석")
        st.info("💡 각 공정별 작업 물량과 생산능력(CAPA)을 비교하여 부하율을 분석합니다.")
        
        # 1. 기간별 보기 선택
        time_scale = st.radio(
            "기간 단위 선택",
            options=["주간(Weekly)", "월간(Monthly)", "연간(Yearly)"],
            index=1,  # 기본값: 월간
            horizontal=True,
            key="load_analysis_time_scale"
        )
        
        # 원본 스케줄 데이터 가져오기 (날짜 포맷팅 전)
        df_original = st.session_state.df_scheduled.copy()
        
        # Duration 타입 공정만 필터링
        duration_processes = st.session_state.processes_df[
            st.session_state.processes_df['Type'] == 'Duration'
        ].sort_values('Order')
        
        if len(duration_processes) == 0:
            st.warning("⚠️ Duration 타입 공정이 없습니다.")
        else:
            # 공정별 부하율 분석
            for _, proc_row in duration_processes.iterrows():
                process_name = proc_row['Process Name']
                start_col = f"{process_name}_Start"
                end_col = f"{process_name}_End"
                days_col = f"{process_name}_Days"
                
                # 해당 공정의 시작일/종료일 컬럼이 있는지 확인
                if start_col not in df_original.columns or end_col not in df_original.columns:
                    continue
                
                # 부하 데이터 수집
                load_data = []
                
                for idx, row in df_original.iterrows():
                    if pd.isna(row[start_col]) or pd.isna(row[end_col]):
                        continue
                    
                    start_date = pd.to_datetime(row[start_col])
                    end_date = pd.to_datetime(row[end_col])
                    weight = float(row.get('중량(Ton)', 0))
                    project_name = row.get('프로젝트명', '')
                    
                    # Days 정보 가져오기
                    if days_col in row and pd.notna(row[days_col]):
                        days = int(row[days_col])
                    else:
                        days = (end_date - start_date).days + 1
                    
                    if days <= 0:
                        continue
                    
                    # 일별 부하 계산 (중량을 작업일수로 나눔)
                    daily_load = weight / days
                    
                    # 각 작업일에 부하 추가
                    current_date = start_date
                    while current_date <= end_date:
                        load_data.append({
                            'Date': current_date,
                            'Process': process_name,
                            'Project': project_name,
                            'Daily_Load': daily_load,
                            'Weight': weight
                        })
                        current_date += pd.Timedelta(days=1)
                
                if not load_data:
                    continue
                
                load_df = pd.DataFrame(load_data)
                
                # 기간별 집계
                if time_scale == "주간(Weekly)":
                    load_df['Period'] = load_df['Date'].dt.to_period('W').astype(str)
                    period_format = "%Y-W%U"
                elif time_scale == "월간(Monthly)":
                    load_df['Period'] = load_df['Date'].dt.to_period('M').astype(str)
                    period_format = "%Y-%m"
                else:  # 연간
                    load_df['Period'] = load_df['Date'].dt.to_period('Y').astype(str)
                    period_format = "%Y"
                
                # 기간별 합계
                aggregated = load_df.groupby('Period')['Daily_Load'].sum().reset_index()
                aggregated.columns = ['Period', 'Total_Load']
                aggregated = aggregated.sort_values('Period')
                
                # CAPA 정보 가져오기 및 변환
                capa_data = {}
                has_capa = False
                
                # 모든 프로젝트의 CAPA 확인
                for project_name in df_original['프로젝트명'].unique():
                    capa_key = (project_name, process_name)
                    monthly_capa = st.session_state.project_capa.get(capa_key, 0)
                    
                    if monthly_capa and monthly_capa > 0:
                        has_capa = True
                        # 기간별 CAPA 변환
                        if time_scale == "주간(Weekly)":
                            period_capa = monthly_capa / 4.3
                        elif time_scale == "월간(Monthly)":
                            period_capa = monthly_capa
                        else:  # 연간
                            period_capa = monthly_capa * 12
                        
                        capa_data[project_name] = period_capa
                
                # 전체 CAPA 계산 (모든 프로젝트 합산)
                total_capa = sum(capa_data.values()) if capa_data else 0
                
                # 차트 생성
                fig = go.Figure()
                
                # 막대 그래프 (부하량)
                colors = []
                for load in aggregated['Total_Load']:
                    if has_capa and total_capa > 0 and load > total_capa:
                        colors.append('red')  # CAPA 초과
                    else:
                        colors.append('steelblue')  # 정상
                
                fig.add_trace(go.Bar(
                    x=aggregated['Period'],
                    y=aggregated['Total_Load'],
                    name='작업 물량',
                    marker_color=colors,
                    text=[f"{load:.1f} Ton" for load in aggregated['Total_Load']],
                    textposition='outside'
                ))
                
                # CAPA 기준선 (CAPA 정보가 있는 경우만)
                if has_capa and total_capa > 0:
                    fig.add_trace(go.Scatter(
                        x=aggregated['Period'],
                        y=[total_capa] * len(aggregated),
                        mode='lines',
                        name=f'CAPA 기준선 ({total_capa:.1f} Ton)',
                        line=dict(color='orange', width=2, dash='dash'),
                        hovertemplate='CAPA: %{y:.1f} Ton<extra></extra>'
                    ))
                
                # 차트 레이아웃 설정
                fig.update_layout(
                    title=f'{process_name} 공정 부하율 분석 ({time_scale})',
                    xaxis_title='기간',
                    yaxis_title='중량 (Ton)',
                    height=400,
                    hovermode='x unified',
                    showlegend=True
                )
                
                # 공정별 차트 표시
                st.plotly_chart(fig, use_container_width=True)
                
                # 요약 정보 표시
                col1, col2, col3 = st.columns(3)
                with col1:
                    max_load = aggregated['Total_Load'].max()
                    st.metric("최대 부하", f"{max_load:.1f} Ton")
                with col2:
                    avg_load = aggregated['Total_Load'].mean()
                    st.metric("평균 부하", f"{avg_load:.1f} Ton")
                with col3:
                    if has_capa and total_capa > 0:
                        max_utilization = (max_load / total_capa * 100) if total_capa > 0 else 0
                        st.metric("최대 부하율", f"{max_utilization:.1f}%", 
                                 delta=f"CAPA: {total_capa:.1f} Ton" if max_utilization > 100 else None,
                                 delta_color="inverse" if max_utilization > 100 else "normal")
                    else:
                        st.metric("CAPA 정보", "미설정")
                
                st.divider()

# ============================================================================
# 메인 실행
# ============================================================================
if __name__ == "__main__":
    # Session State 초기화
    init_session_state()

    app_title = get_app_title()
    apply_browser_title(app_title)

    with st.sidebar:
        st.markdown(f"### {app_title}")

    # 상단 메뉴와 버전 정보
    col1, col2 = st.columns([10, 1])
    
    with col1:
        selected = option_menu(
            menu_title=None,
            options=["홈(Home)", "기초정보 관리", "스케줄링 메인", "공정 설정(Admin)"],
            icons=["house", "cloud-upload", "list-task", "gear"],
            menu_icon="cast",
            default_index=0,
            orientation="horizontal",
            styles={
                "container": {"padding": "0!important", "background-color": "#fafafa"},
                "icon": {"color": "orange", "font-size": "18px"},
                "nav-link": {
                    "font-size": "16px",
                    "text-align": "center",
                    "margin": "0px",
                    "--hover-color": "#eee",
                },
                "nav-link-selected": {"background-color": "#02ab21"},
            }
        )
    
    with col2:
        st.markdown(
            f'<div style="text-align: right; padding-top: 10px; color: #666; font-size: 12px;">'
            f'{APP_VERSION}<br>{APP_AUTHOR}'
            f'</div>',
            unsafe_allow_html=True
        )
    
    # 페이지 라우팅
    if selected == "홈(Home)":
        main_home()
    elif selected == "기초정보 관리":
        page_input()
    elif selected == "스케줄링 메인":
        page_schedule()
    elif selected == "공정 설정(Admin)":
        page_admin()
