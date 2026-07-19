from __future__ import annotations

import io
import json
import os
import re
import zipfile
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from advanced_statistics import METHOD_CATEGORIES
from youtube_startup_mining import (
    APP_VERSION,
    AUTHOR_CREDIT,
    YouTubeAPIError,
    analyze,
    api_get,
    collect,
    create_demo_data,
    load_config,
    write_markdown_report,
)


APP_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = APP_DIR / "config.json"
RUNS_DIR = APP_DIR / "runs"
PRODUCT_PROMISE = (
    "키워드와 YouTube API 키만 입력하면 영상·댓글 경향을 통계, 그래프, "
    "Excel 및 연구 보고서로 자동 생성하는 텍스트 마이닝 도구"
)


st.set_page_config(
    page_title=f"YouTube Research Studio {APP_VERSION}",
    page_icon=":material/query_stats:",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
.block-container {padding-top: 1.6rem; padding-bottom: 4rem;}
[data-testid="stMetricValue"] {color: #175A8C;}
.method-note {padding: .8rem 1rem; border-left: 4px solid #2F75B5; background: #F3F8FC; border-radius: 4px;}
.safe-note {padding: .8rem 1rem; border-left: 4px solid #2E8B57; background: #F2FBF6; border-radius: 4px;}
</style>
""",
    unsafe_allow_html=True,
)


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(path, encoding="utf-8-sig")


def split_terms(value: str) -> list[str]:
    return [
        term.strip()
        for term in re.split(r"[,\n;]+", value)
        if term.strip()
    ]


def safe_project_name(value: str) -> str:
    cleaned = re.sub(r"[^0-9A-Za-z가-힣_-]+", "_", value.strip()).strip("_")
    return cleaned[:60] or "youtube_analysis"


def to_api_datetime(day: date, end_of_range: bool = False) -> str:
    target = day + timedelta(days=1) if end_of_range else day
    return datetime.combine(target, time.min, tzinfo=timezone.utc).isoformat().replace(
        "+00:00", "Z"
    )


def topic_frame(config: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"주제": topic, "키워드(쉼표 구분)": ", ".join(words)}
            for topic, words in config.get("keyword_groups", {}).items()
        ]
    )


def topic_dict(frame: pd.DataFrame) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for _, row in frame.fillna("").iterrows():
        topic = str(row.get("주제", "")).strip()
        words = split_terms(str(row.get("키워드(쉼표 구분)", "")))
        if topic and words:
            result[topic] = words
    return result


def discover_runs() -> list[Path]:
    if not RUNS_DIR.exists():
        return []
    return sorted(
        [
            path.parent
            for path in RUNS_DIR.glob("*/analysis/run_manifest.json")
            if path.is_file()
        ],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )


def table_files(analysis_dir: Path) -> dict[str, Path]:
    tables_dir = analysis_dir / "tables"
    return {
        path.stem: path for path in sorted(tables_dir.glob("*.csv")) if path.is_file()
    }


def zip_run(run_dir: Path) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in run_dir.rglob("*"):
            if path.is_file():
                archive.write(path, path.relative_to(run_dir))
    return buffer.getvalue()


def correlation_heatmap(table: pd.DataFrame) -> go.Figure | None:
    if table.empty or not {
        "variable_1",
        "variable_2",
        "method",
        "coefficient",
    }.issubset(table.columns):
        return None
    selected = table[table["method"] == "Spearman"].copy()
    if selected.empty:
        return None
    variables = sorted(set(selected["variable_1"]) | set(selected["variable_2"]))
    matrix = pd.DataFrame(np.eye(len(variables)), index=variables, columns=variables)
    for _, row in selected.iterrows():
        matrix.loc[row["variable_1"], row["variable_2"]] = row["coefficient"]
        matrix.loc[row["variable_2"], row["variable_1"]] = row["coefficient"]
    return px.imshow(
        matrix,
        zmin=-1,
        zmax=1,
        color_continuous_scale="RdBu_r",
        text_auto=".2f",
        aspect="auto",
        title="Spearman 상관계수 행렬",
    )


def cooccurrence_network(table: pd.DataFrame, limit: int = 45) -> go.Figure | None:
    if table.empty or not {"word_1", "word_2", "cooccurrence"}.issubset(table.columns):
        return None
    selected = table.nlargest(limit, "cooccurrence")
    graph = nx.Graph()
    for _, row in selected.iterrows():
        graph.add_edge(
            str(row["word_1"]),
            str(row["word_2"]),
            weight=float(row["cooccurrence"]),
        )
    if graph.number_of_nodes() < 2:
        return None
    positions = nx.spring_layout(graph, seed=42, weight="weight", k=1.1)
    edge_x, edge_y = [], []
    for first, second in graph.edges():
        x0, y0 = positions[first]
        x1, y1 = positions[second]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        mode="lines",
        line={"width": 1, "color": "#B8C5D1"},
        hoverinfo="none",
    )
    weighted_degree = dict(graph.degree(weight="weight"))
    node_trace = go.Scatter(
        x=[positions[node][0] for node in graph.nodes()],
        y=[positions[node][1] for node in graph.nodes()],
        mode="markers+text",
        text=list(graph.nodes()),
        textposition="top center",
        hovertemplate="%{text}<extra></extra>",
        marker={
            "size": [10 + math_value**0.45 * 3 for math_value in weighted_degree.values()],
            "color": list(weighted_degree.values()),
            "colorscale": "Blues",
            "showscale": True,
            "colorbar": {"title": "연결 강도"},
            "line": {"width": 1, "color": "white"},
        },
    )
    return go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title="댓글 단어 동시출현 네트워크",
            showlegend=False,
            hovermode="closest",
            margin={"l": 10, "r": 10, "t": 50, "b": 10},
            xaxis={"showgrid": False, "zeroline": False, "showticklabels": False},
            yaxis={"showgrid": False, "zeroline": False, "showticklabels": False},
            height=650,
        ),
    )


def ensure_markdown_report(
    run_dir: Path,
    analysis_dir: Path,
    tables: dict[str, pd.DataFrame],
    manifest: dict[str, Any],
) -> Path:
    markdown_path = analysis_dir / "startup_youtube_all_results.md"
    if markdown_path.exists():
        return markdown_path
    config_path = run_dir / "run_config.json"
    config = (
        json.loads(config_path.read_text(encoding="utf-8"))
        if config_path.exists()
        else {"queries": manifest.get("queries", [])}
    )
    write_markdown_report(markdown_path, tables, manifest, config)
    return markdown_path


def render_downloads(
    run_dir: Path,
    analysis_dir: Path,
    tables: dict[str, pd.DataFrame],
    manifest: dict[str, Any],
) -> None:
    st.subheader("결과 내려받기")
    first, second, third, fourth = st.columns(4)
    excel_path = analysis_dir / "startup_youtube_analysis.xlsx"
    markdown_path = ensure_markdown_report(run_dir, analysis_dir, tables, manifest)
    html_path = analysis_dir / "startup_youtube_report.html"
    with first:
        if excel_path.exists():
            st.download_button(
                "모든 결과 Excel",
                data=excel_path.read_bytes,
                file_name=excel_path.name,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                width="stretch",
                icon=":material/table_view:",
                on_click="ignore",
            )
    with second:
        st.download_button(
            "모든 결과 Markdown",
            data=markdown_path.read_bytes,
            file_name=markdown_path.name,
            mime="text/markdown; charset=utf-8",
            width="stretch",
            icon=":material/markdown:",
            on_click="ignore",
        )
    with third:
        if html_path.exists():
            st.download_button(
                "HTML 요약 보고서",
                data=html_path.read_bytes,
                file_name=html_path.name,
                mime="text/html",
                width="stretch",
                icon=":material/web:",
                on_click="ignore",
            )
    with fourth:
        st.download_button(
            "원자료 포함 전체 ZIP",
            data=lambda: zip_run(run_dir),
            file_name=f"{run_dir.name}.zip",
            mime="application/zip",
            width="stretch",
            icon=":material/folder_zip:",
            on_click="ignore",
            help="공개 댓글 원문이 포함될 수 있으므로 공유 전 개인정보·연구윤리를 확인하세요.",
        )


def render_dashboard(analysis_dir: Path) -> None:
    manifest_path = analysis_dir / "run_manifest.json"
    if not manifest_path.exists():
        st.error("선택한 폴더에 분석 결과가 없습니다.")
        return
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    run_dir = analysis_dir.parent
    tables = {name: read_csv(path) for name, path in table_files(analysis_dir).items()}
    videos = read_csv(analysis_dir / "videos_enriched.csv")
    comments = read_csv(analysis_dir / "comments_enriched.csv")

    st.header(f"분석 결과: {run_dir.name}")
    if manifest.get("demo_data"):
        st.warning("이 결과는 프로그램 기능 확인용 가상 데이터입니다.")
    metric_columns = st.columns(4)
    metric_columns[0].metric("영상", f"{manifest.get('video_count', 0):,}개")
    metric_columns[1].metric("댓글", f"{manifest.get('comment_count', 0):,}개")
    metric_columns[2].metric("채널", f"{manifest.get('unique_channel_count', 0):,}개")
    metric_columns[3].metric("검색어", ", ".join(manifest.get("queries", [])))

    video_tab, comment_tab, statistics_tab, data_tab = st.tabs(
        ["영상 경향", "댓글·텍스트", "통계 방법론", "자료·다운로드"]
    )

    with video_tab:
        monthly = tables.get("monthly_video_trends", pd.DataFrame())
        topic = tables.get("topic_summary", pd.DataFrame())
        duration = tables.get("duration_summary", pd.DataFrame())
        left, right = st.columns(2)
        with left:
            if not monthly.empty:
                st.plotly_chart(
                    px.line(
                        monthly,
                        x="publish_month",
                        y="video_count",
                        markers=True,
                        title="월별 게시 영상 수",
                        labels={"publish_month": "게시 월", "video_count": "영상 수"},
                    ),
                    width="stretch",
                )
        with right:
            if not topic.empty:
                st.plotly_chart(
                    px.bar(
                        topic.sort_values("video_count"),
                        x="video_count",
                        y="topic_primary",
                        orientation="h",
                        title="사용자 정의 주제별 영상 수",
                        labels={"topic_primary": "주제", "video_count": "영상 수"},
                    ),
                    width="stretch",
                )
        if not videos.empty:
            first, second = st.columns(2)
            with first:
                if {"topic_primary", "views_per_day"}.issubset(videos.columns):
                    st.plotly_chart(
                        px.box(
                            videos,
                            x="topic_primary",
                            y="views_per_day",
                            points="outliers",
                            log_y=True,
                            title="주제별 일평균 조회수 분포",
                            labels={"topic_primary": "주제", "views_per_day": "일평균 조회수"},
                        ),
                        width="stretch",
                    )
            with second:
                if {"channel_subscriber_count", "views_per_day", "topic_primary"}.issubset(
                    videos.columns
                ):
                    scatter = videos[
                        (pd.to_numeric(videos["channel_subscriber_count"], errors="coerce") > 0)
                        & (pd.to_numeric(videos["views_per_day"], errors="coerce") > 0)
                    ]
                    st.plotly_chart(
                        px.scatter(
                            scatter,
                            x="channel_subscriber_count",
                            y="views_per_day",
                            color="topic_primary",
                            hover_name="title",
                            log_x=True,
                            log_y=True,
                            title="채널 규모와 일평균 조회수",
                            labels={
                                "channel_subscriber_count": "채널 구독자 수",
                                "views_per_day": "일평균 조회수",
                            },
                        ),
                        width="stretch",
                    )
        if not duration.empty:
            st.dataframe(duration, width="stretch", hide_index=True)

    with comment_tab:
        sentiment = tables.get("sentiment_summary", pd.DataFrame())
        monthly_comments = tables.get("monthly_comment_trends", pd.DataFrame())
        word_frequency = tables.get("comment_word_frequency", pd.DataFrame())
        cooccurrence = tables.get("comment_cooccurrence", pd.DataFrame())
        left, right = st.columns(2)
        with left:
            if not sentiment.empty:
                st.plotly_chart(
                    px.pie(
                        sentiment,
                        names="sentiment",
                        values="comment_count",
                        title="댓글 감성 구성",
                        hole=0.42,
                    ),
                    width="stretch",
                )
        with right:
            if not word_frequency.empty:
                words = word_frequency.head(25).sort_values("term_frequency")
                st.plotly_chart(
                    px.bar(
                        words,
                        x="term_frequency",
                        y="word",
                        orientation="h",
                        title="댓글 빈출어 상위 25개",
                        labels={"word": "단어", "term_frequency": "빈도"},
                    ),
                    width="stretch",
                )
        if not monthly_comments.empty:
            share_columns = [
                column
                for column in ["positive_share", "negative_share", "neutral_share"]
                if column in monthly_comments
            ]
            melted = monthly_comments.melt(
                id_vars="comment_month",
                value_vars=share_columns,
                var_name="감성",
                value_name="비율",
            )
            st.plotly_chart(
                px.line(
                    melted,
                    x="comment_month",
                    y="비율",
                    color="감성",
                    markers=True,
                    title="월별 댓글 감성 비율",
                ),
                width="stretch",
            )
        network = cooccurrence_network(cooccurrence)
        if network is not None:
            st.plotly_chart(network, width="stretch")
        st.caption(
            "감성분석은 사용자가 설정한 사전 기반의 탐색적 분류입니다. 논문 보고 전에는 사람이 코딩한 검증 표본과 비교하세요."
        )

    with statistics_tab:
        catalog = tables.get("advanced_method_catalog", pd.DataFrame())
        if not catalog.empty:
            executed = int((catalog["status"] == "실행됨").sum())
            st.info(f"방법론 목록 {len(catalog)}개 중 현재 자료에서 {executed}개가 실행되었습니다.")
            st.dataframe(catalog, width="stretch", hide_index=True)

        correlations = tables.get("advanced_full_correlations", pd.DataFrame())
        heatmap = correlation_heatmap(correlations)
        if heatmap is not None:
            st.plotly_chart(heatmap, width="stretch")

        available = {
            "정규성": "advanced_normality_tests",
            "등분산성": "advanced_variance_tests",
            "두 집단 비교": "advanced_two_group_tests",
            "다집단 비교": "advanced_multi_group_tests",
            "사후검정": "advanced_posthoc_tests",
            "상관분석": "advanced_full_correlations",
            "범주형 분석": "advanced_contingency_tests",
            "회귀모형": "advanced_regression_models",
            "회귀진단": "advanced_regression_diagnostics",
            "시계열": "advanced_time_series_tests",
            "PCA 적재량": "advanced_pca_loadings",
            "군집 프로필": "advanced_cluster_profiles",
            "이상치": "advanced_outlier_summary",
            "표본크기": "advanced_power_analysis",
        }
        selected_label = st.selectbox("통계 결과표 선택", list(available))
        selected_table = tables.get(available[selected_label], pd.DataFrame())
        if selected_table.empty:
            st.warning("선택한 통계 방법은 현재 표본 조건에서 실행되지 않았습니다.")
        else:
            st.dataframe(selected_table, width="stretch", hide_index=True)
        st.markdown(
            '<div class="method-note">유의성은 원 p값만 보지 않고 Holm 보정 p값, FDR q값, 효과크기, 신뢰구간을 함께 판단합니다. 검색 결과는 확률표본이 아니므로 모집단 추론에는 제한이 있습니다.</div>',
            unsafe_allow_html=True,
        )

    with data_tab:
        render_downloads(run_dir, analysis_dir, tables, manifest)
        st.subheader("분석 자료 미리보기")
        preview = st.radio("자료", ["영상", "댓글"], horizontal=True)
        target = videos if preview == "영상" else comments
        st.dataframe(target.head(2000), width="stretch", hide_index=True)
        csv_bytes = target.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
        st.download_button(
            f"{preview} CSV 내려받기",
            data=csv_bytes,
            file_name="videos_enriched.csv" if preview == "영상" else "comments_enriched.csv",
            mime="text/csv",
        )


def build_config_form(base: dict[str, Any], api_key: str) -> tuple[dict[str, Any] | None, bool]:
    st.subheader("새 연구 설정")
    with st.form("research_config"):
        project_name = st.text_input("프로젝트 이름", value="startup_research")
        query_text = st.text_area(
            "검색 주제·키워드",
            value="\n".join(base.get("queries", ["스타트업", "startup"])),
            help="쉼표 또는 줄바꿈으로 여러 검색어를 입력합니다.",
        )
        first, second, third = st.columns(3)
        with first:
            start_date = st.date_input(
                "시작일",
                value=date.fromisoformat(
                    str(base.get("published_after", "2025-01-01"))[:10]
                ),
            )
        with second:
            end_date = st.date_input("종료일", value=date.today())
        with third:
            order = st.selectbox(
                "검색 정렬",
                ["date", "relevance", "viewCount", "rating", "title"],
                index=["date", "relevance", "viewCount", "rating", "title"].index(
                    base.get("order", "date")
                ),
            )

        first, second, third, fourth = st.columns(4)
        with first:
            region_code = st.text_input("국가 코드", value=base.get("region_code", "KR"))
        with second:
            language = st.text_input(
                "관련 언어", value=base.get("relevance_language", "ko")
            )
        with third:
            max_videos = st.number_input(
                "검색어별 영상 수",
                min_value=1,
                max_value=1000,
                value=min(int(base.get("max_videos_per_query", 200)), 1000),
                step=10,
            )
        with fourth:
            max_comments = st.number_input(
                "영상별 댓글 수",
                min_value=0,
                max_value=5000,
                value=min(int(base.get("max_comments_per_video", 300)), 5000),
                step=50,
            )

        replies, all_replies, bootstrap = st.columns(3)
        with replies:
            include_replies = st.checkbox(
                "답글 포함", value=bool(base.get("include_replies", True))
            )
        with all_replies:
            collect_all_replies = st.checkbox(
                "모든 답글 추가 수집",
                value=bool(base.get("collect_all_replies", False)),
                help="API 요청량이 크게 증가할 수 있습니다.",
            )
        with bootstrap:
            bootstrap_iterations = st.select_slider(
                "부트스트랩 반복",
                options=[100, 300, 500, 1000, 2000, 5000],
                value=500
                if int(base.get("bootstrap_iterations", 500)) not in [100, 300, 500, 1000, 2000, 5000]
                else int(base.get("bootstrap_iterations", 500)),
            )

        st.markdown("#### 사용자 정의 주제 사전")
        edited_topics = st.data_editor(
            topic_frame(base),
            num_rows="dynamic",
            width="stretch",
            hide_index=True,
        )
        selected_methods = st.multiselect(
            "실행할 통계 방법론 묶음",
            METHOD_CATEGORIES,
            default=METHOD_CATEGORIES,
        )
        with st.expander("감성·불용어 사전 설정"):
            positive_text = st.text_area(
                "긍정 단어",
                value=", ".join(base.get("positive_words", [])),
            )
            negative_text = st.text_area(
                "부정 단어",
                value=", ".join(base.get("negative_words", [])),
            )
            stopword_text = st.text_area(
                "분석 제외 단어",
                value=", ".join(base.get("stopwords", [])),
            )

        first, second = st.columns(2)
        run_actual = first.form_submit_button(
            "실제 YouTube 수집·분석",
            type="primary",
            width="stretch",
        )
        run_demo = second.form_submit_button(
            "가상자료로 시험",
            width="stretch",
        )

    if not run_actual and not run_demo:
        return None, False
    queries = split_terms(query_text)
    topics = topic_dict(edited_topics)
    if not queries:
        st.error("검색어를 하나 이상 입력하세요.")
        return None, False
    if not topics:
        st.error("사용자 정의 주제를 하나 이상 설정하세요.")
        return None, False
    if start_date > end_date:
        st.error("시작일은 종료일보다 늦을 수 없습니다.")
        return None, False
    if run_actual and not api_key:
        st.error("왼쪽 화면에 YouTube API 키를 입력하세요.")
        return None, False

    config = dict(base)
    config.update(
        {
            "project_name": safe_project_name(project_name),
            "queries": queries,
            "published_after": to_api_datetime(start_date),
            "published_before": to_api_datetime(end_date, end_of_range=True),
            "region_code": region_code.strip().upper(),
            "relevance_language": language.strip().lower(),
            "order": order,
            "max_videos_per_query": int(max_videos),
            "max_comments_per_video": int(max_comments),
            "include_replies": include_replies,
            "collect_all_replies": collect_all_replies,
            "bootstrap_iterations": int(bootstrap_iterations),
            "keyword_groups": topics,
            "advanced_methods": selected_methods,
            "positive_words": split_terms(positive_text),
            "negative_words": split_terms(negative_text),
            "stopwords": split_terms(stopword_text),
        }
    )
    return config, run_demo


def execute_research(config: dict[str, Any], api_key: str, demo: bool) -> Path | None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = RUNS_DIR / f"{safe_project_name(config['project_name'])}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    try:
        if demo:
            with st.status("가상자료를 만들고 분석하고 있습니다.", expanded=True) as status:
                st.write("가상 영상과 댓글 생성")
                create_demo_data(run_dir / "raw", config)
                st.write("기초·고급 통계와 보고서 생성")
                analyze(run_dir / "raw", run_dir / "analysis", config, demo=True)
                status.update(label="가상자료 분석 완료", state="complete")
        else:
            with st.status("YouTube 자료를 수집하고 있습니다.", expanded=True) as status:
                st.write("검색 결과와 영상·채널 정보 수집")
                collection = collect(run_dir, config, api_key)
                st.write(
                    f"영상 {collection['video_count']:,}개, 댓글 {collection['comment_count']:,}개 수집"
                )
                st.write("통계, 표, 차트, 보고서 생성")
                analyze(run_dir / "raw", run_dir / "analysis", config)
                status.update(label="YouTube 수집·분석 완료", state="complete")
    except (YouTubeAPIError, ValueError, FileNotFoundError) as exc:
        st.error(f"실행 중 오류가 발생했습니다: {exc}")
        return None
    return run_dir / "analysis"


def main() -> None:
    base_config = load_config(DEFAULT_CONFIG_PATH)
    st.title(f"YouTube Research Studio {APP_VERSION}")
    st.markdown(
        f":material/school: **{AUTHOR_CREDIT}** · :blue-badge[{APP_VERSION}]"
    )
    with st.container(border=True):
        st.markdown(f"### :material/query_stats: {PRODUCT_PROMISE}")
        st.markdown(
            ":blue-badge[1. API 키 입력] "
            ":violet-badge[2. 키워드·주제 설정] "
            ":green-badge[3. 수집·통계 분석] "
            ":orange-badge[4. 보고서 다운로드]"
        )
        st.caption(
            "API 키는 YouTube 요청에만 사용되며 설정 파일과 분석 결과에 저장되지 않습니다."
        )

    with st.sidebar:
        st.header("YouTube 연결")
        api_key = st.text_input(
            "YouTube Data API 키",
            value=os.environ.get("YOUTUBE_API_KEY", ""),
            type="password",
            help="키는 화면 메모리에만 사용되며 설정 파일과 결과물에 저장되지 않습니다.",
        )
        if st.button("API 연결 시험", width="stretch"):
            if not api_key:
                st.warning("API 키를 입력하세요.")
            else:
                try:
                    api_get(
                        "videos",
                        {
                            "part": "id",
                            "chart": "mostPopular",
                            "maxResults": 1,
                            "regionCode": "KR",
                        },
                        api_key,
                    )
                    st.success("YouTube API 연결 성공")
                except YouTubeAPIError as exc:
                    st.error(str(exc))
        st.markdown(
            '<div class="safe-note">API 키는 실행 중에만 사용하며 디스크에 저장하지 않습니다.</div>',
            unsafe_allow_html=True,
        )
        st.divider()
        st.header("기존 실행 결과")
        runs = discover_runs()
        run_options = {path.parent.name: path for path in runs}
        selected_run = st.selectbox(
            "결과 선택",
            ["선택 안 함", *run_options.keys()],
        )
        if st.button("선택 결과 열기", width="stretch"):
            if selected_run != "선택 안 함":
                st.session_state["analysis_dir"] = str(run_options[selected_run])
        st.space("small")
        st.caption(f"{AUTHOR_CREDIT}\n\n{APP_VERSION}")

    config, demo = build_config_form(base_config, api_key)
    if config is not None:
        result_dir = execute_research(config, api_key, demo)
        if result_dir is not None:
            st.session_state["analysis_dir"] = str(result_dir)
            st.success(f"완료: {result_dir}")

    analysis_dir_value = st.session_state.get("analysis_dir")
    if analysis_dir_value:
        st.divider()
        render_dashboard(Path(analysis_dir_value))
    else:
        st.info("가상자료 시험 또는 실제 수집·분석을 실행하면 이 화면에 대시보드가 표시됩니다.")


if __name__ == "__main__":
    main()
