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

# ============================================================================
# 페이지 설정
# ============================================================================
st.set_page_config(
    page_title="IronFlow - 생산 스케줄링",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# 상수 정의
# ============================================================================
WEEKDAYS = {
    "월": 0, "화": 1, "수": 2, "목": 3, "금": 4, "토": 5, "일": 6
}
WEEKDAY_NAMES = ["월", "화", "수", "목", "금", "토", "일"]

# ============================================================================
# Session State 초기화
# ============================================================================
def init_session_state():
    """Session State 초기화"""
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
    st.title("🏠 IronFlow")
    st.markdown("### 조선기자재 생산 자동 스케줄링 시스템")
    
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
    ### 📖 사용 방법
    
    1. **공정 설정(Admin)**: 시스템에서 사용할 공정을 정의합니다.
    2. **기초정보 관리**: 엑셀 파일을 업로드하거나 샘플 데이터를 생성합니다.
    3. **스케줄링 메인**: 휴무일을 설정하고 역산 스케줄을 계산합니다.
    
    ### 💡 주요 기능
    
    - ✅ 동적 공정 관리: 공정을 자유롭게 추가/수정/삭제
    - ✅ 팀별 휴무일 설정: 각 팀의 근무 요일과 휴무일을 개별 설정
    - ✅ NumPy 가속 계산: 대용량 데이터도 빠르게 처리
    - ✅ 간트 차트 시각화: 스케줄을 한눈에 확인
    """)

def page_admin():
    """공정 설정(Admin) 페이지"""
    st.title("⚙️ 공정 설정 (Admin)")
    st.caption("시스템에서 사용할 공정을 정의하고 관리합니다.")
    
    st.info("💡 공정을 추가/수정/삭제하면 다른 페이지에 즉시 반영됩니다.")
    
    # 공정 데이터프레임 편집
    edited_processes_df = st.data_editor(
        st.session_state.processes_df,
        num_rows="dynamic",
        column_config={
            "Process Name": st.column_config.TextColumn(
                "공정명",
                required=True,
                help="공정 이름을 입력하세요"
            ),
            "Type": st.column_config.SelectboxColumn(
                "유형",
                options=["Duration", "Milestone"],
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
    st.subheader("📋 현재 공정 리스트")
    st.dataframe(
        st.session_state.processes_df.sort_values('Order'),
        use_container_width=True,
        hide_index=True
    )

def page_input():
    """기초정보 관리 페이지 - 프로젝트 마스터 등록 및 공정 시수 입력"""
    st.title("📥 기초정보 관리")
    st.caption("프로젝트 마스터 정보를 등록하고 공정별 소요기간을 입력합니다.")
    
    # ========================================================================
    # 데이터 입력 양식 다운로드 기능
    # ========================================================================
    st.subheader("📋 표준 입력 양식 다운로드")
    st.info("💡 아래 버튼을 클릭하여 시스템에 맞는 엑셀 템플릿을 다운로드하세요.")
    
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
    
    st.divider()
    
    tab1, tab2 = st.tabs(["1️⃣ 프로젝트 마스터 등록", "2️⃣ 공정별 소요기간 입력"])
    
    # ========================================================================
    # 1단계: 프로젝트 마스터 등록
    # ========================================================================
    with tab1:
        st.subheader("📋 신규 프로젝트 등록")
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
        st.subheader("📊 등록된 프로젝트 목록")
        
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
        st.subheader("⏱️ 공정별 소요기간 (Lead Time) 계획 수립")
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
    # 데이터 통합 및 최종 데이터프레임 생성
    # ========================================================================
    st.divider()
    st.subheader("🔄 최종 계산용 데이터 통합")
    
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
    st.title("📅 스케줄링 메인")
    st.caption("휴무일을 설정하고 역산 스케줄을 계산합니다.")
    
    # 데이터 확인
    if st.session_state.df_raw is None:
        st.warning("⚠️ 먼저 [기초정보 관리] 페이지에서 데이터를 업로드하거나 생성해주세요.")
        return
    
    # 사이드바: 휴무일 설정
    with st.sidebar:
        st.header("📅 휴무일 설정")
        
        # 공통 휴무일
        st.subheader("🌐 공통 휴무일")
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
        st.subheader("👥 팀별 설정")
        
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
        st.subheader("📊 스케줄 결과")
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
        st.subheader("📈 간트 차트")
        
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
            
            # Plotly 간트 차트 생성
            fig = px.timeline(
                gantt_df,
                x_start='Start',
                x_end='Finish',
                y='Task',
                color='Process',
                title='생산 스케줄 간트 차트'
            )
            fig.update_yaxes(autorange="reversed")
            fig.update_layout(
                height=600,
                xaxis_title="날짜",
                yaxis_title="프로젝트-블록"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("간트 차트를 생성할 데이터가 없습니다.")

# ============================================================================
# 메인 실행
# ============================================================================
if __name__ == "__main__":
    # Session State 초기화
    init_session_state()
    
    # 상단 메뉴
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
    
    # 페이지 라우팅
    if selected == "홈(Home)":
        main_home()
    elif selected == "기초정보 관리":
        page_input()
    elif selected == "스케줄링 메인":
        page_schedule()
    elif selected == "공정 설정(Admin)":
        page_admin()
