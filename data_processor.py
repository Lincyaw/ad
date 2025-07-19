"""
Data preprocessing module containing data cleaning, feature engineering and graph construction logic.
Optimized version: uses span_name as nodes and trace statistics as features.
"""

import pandas as pd
import numpy as np
import torch
from typing import List, Dict, Any, Tuple, Optional
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from torch_geometric.data import Data

from config import ModelConfig, TraceDataConfig


def extract_path(span_name: str) -> str:
    """Extract path from span name. Simple implementation as fallback."""
    if not isinstance(span_name, str):
        return str(span_name)
    # Simple normalization: remove extra spaces and convert to lowercase
    return span_name.strip().lower()


class DataProcessor:
    """Encapsulates data cleaning, feature engineering and graph construction logic - optimized version."""

    def __init__(
        self, config: ModelConfig, trace_config: Optional[TraceDataConfig] = None
    ) -> None:
        self.config = config
        self.trace_config = trace_config or TraceDataConfig()
        self.ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.span_name_to_idx: Dict[str, int] = {}
        self.service_graph = None

    def fit(self, df: pd.DataFrame) -> None:
        df_copy = self._preprocess_dataframe(df)
        # Save original data reference for graph construction
        self._original_data = df_copy

        unique_span_names = df_copy["span_name"].unique()
        self.span_name_to_idx = {
            name: idx for idx, name in enumerate(unique_span_names)
        }

        self._build_service_graph(df_copy)

        trace_features = self._aggregate_trace_features(df_copy)

        available_categorical = [
            col
            for col in self.config.categorical_features
            if col in trace_features.columns
        ]
        available_numerical = [
            col
            for col in self.config.numerical_features
            if col in trace_features.columns
        ]

        if available_categorical:
            self.ohe.fit(trace_features[available_categorical])
        if available_numerical:
            self.scaler.fit(trace_features[available_numerical])

        self.is_fitted = True
        print("DataProcessor fitted:")
        print(f"  - {len(unique_span_names)} unique span names")
        print(f"  - {len(available_categorical)} categorical features")
        print(f"  - {len(available_numerical)} numerical features")

    def transform(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Transform data to trace-level features."""
        if not self.is_fitted:
            raise RuntimeError("Processor has not been fitted. Call fit() first.")

        df_copy = self._preprocess_dataframe(df)
        # Save current data reference for graph construction
        self._current_data = df_copy

        trace_features = self._aggregate_trace_features(df_copy)

        # Feature encoding
        available_categorical = [
            col
            for col in self.config.categorical_features
            if col in trace_features.columns
        ]
        available_numerical = [
            col
            for col in self.config.numerical_features
            if col in trace_features.columns
        ]

        features_list = []

        if available_categorical:
            cat_encoded = self.ohe.transform(trace_features[available_categorical])
            features_list.append(cat_encoded)

        if available_numerical:
            num_scaled = self.scaler.transform(trace_features[available_numerical])
            features_list.append(num_scaled)

        if features_list:
            features = np.hstack(features_list)
        else:
            features = np.ones((len(trace_features), 1))

        return trace_features, pd.DataFrame(features, index=trace_features.index)

    def _preprocess_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        df_copy = df.copy()

        df_copy["span_name"] = df_copy["span_name"].fillna("unknown_span").astype(str)
        df_copy["span_name"] = df_copy["span_name"].apply(extract_path)
        df_copy["service_name"] = (
            df_copy["service_name"].fillna("unknown_service").astype(str)
        )

        # Process categorical features
        for col in self.config.categorical_features:
            if col in df_copy.columns:
                df_copy[col] = (
                    df_copy[col]
                    .fillna(self.trace_config.categorical_fill_value)
                    .astype(str)
                )

        # Process numerical features
        for col in self.config.numerical_features:
            if col in df_copy.columns:
                # First convert non-numeric strings to NaN
                df_copy[col] = pd.to_numeric(df_copy[col], errors="coerce")
                # Then fill missing values
                df_copy[col] = df_copy[col].fillna(
                    self.trace_config.numerical_fill_value
                )

        return df_copy

    def _build_service_graph(self, df: pd.DataFrame):
        import networkx as nx

        G = nx.DiGraph()

        for span_name in df["span_name"].unique():
            G.add_node(span_name)

        for trace_id, trace_df in df.groupby("trace_id"):
            span_id_to_name = dict(zip(trace_df["span_id"], trace_df["span_name"]))

            for _, row in trace_df.iterrows():
                if (
                    pd.notna(row["parent_span_id"])
                    and row["parent_span_id"] in span_id_to_name
                ):
                    parent_span_name = span_id_to_name[row["parent_span_id"]]
                    current_span_name = row["span_name"]

                    if not G.has_edge(parent_span_name, current_span_name):
                        G.add_edge(parent_span_name, current_span_name, weight=1)
                    else:
                        G[parent_span_name][current_span_name]["weight"] += 1

        self.service_graph = G

    def _aggregate_trace_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate span-level data to trace-level features."""
        trace_features = []

        for trace_id, trace_df in df.groupby("trace_id"):
            # Basic statistical features
            features = {
                "trace_id": trace_id,
                "span_count": len(trace_df),
                "total_duration": trace_df["duration"].sum(),
                "avg_duration": trace_df["duration"].mean(),
                "max_duration": trace_df["duration"].max(),
                "min_duration": trace_df["duration"].min(),
                "duration_std": trace_df["duration"].std()
                if len(trace_df) > 1
                else 0.0,
            }

            # Handle NaN values
            for key in [
                "total_duration",
                "avg_duration",
                "max_duration",
                "min_duration",
                "duration_std",
            ]:
                if pd.isna(features[key]):
                    features[key] = 0.0

            # Service diversity features
            features["unique_services"] = trace_df["service_name"].nunique()
            features["unique_spans"] = trace_df["span_name"].nunique()

            # Error rate feature (if status code available)
            if "attr.http.response.status_code" in trace_df.columns:
                status_codes = pd.to_numeric(
                    trace_df["attr.http.response.status_code"], errors="coerce"
                )
                features["error_rate"] = (
                    (status_codes >= 400).mean()
                    if not status_codes.isna().all()
                    else 0.0
                )
            else:
                features["error_rate"] = 0.0

            features["primary_service"] = (
                trace_df["service_name"].mode().iloc[0]
                if not trace_df["service_name"].empty
                else "unknown"
            )
            features["root_span"] = (
                trace_df[trace_df["parent_span_id"].isna()]["span_name"].iloc[0]
                if not trace_df[trace_df["parent_span_id"].isna()].empty
                else "unknown"
            )

            trace_features.append(features)

        return pd.DataFrame(trace_features).set_index("trace_id")

    def _calculate_span_depth(
        self, span_row: pd.Series, trace_data: pd.DataFrame
    ) -> int:
        """Calculate span depth in call tree."""
        depth = 0
        current_span_id = span_row["span_id"]
        current_parent_id = span_row["parent_span_id"]

        # Traverse upward to root node
        visited = set()  # Prevent circular references
        while (
            pd.notna(current_parent_id)
            and current_parent_id not in visited
            and depth < 50
        ):  # Maximum depth limit
            visited.add(current_span_id)
            parent_rows = trace_data[trace_data["span_id"] == current_parent_id]
            if parent_rows.empty:
                break

            depth += 1
            current_span_id = current_parent_id
            current_parent_id = parent_rows.iloc[0]["parent_span_id"]

        return depth

    def _create_time_windows(
        self, df: pd.DataFrame, window_size_seconds: int = 20
    ) -> Dict[pd.Timestamp, pd.DataFrame]:
        """Group data by time windows."""
        if "time" not in df.columns:
            # If no time column, group by trace_id as fallback
            windows = {}
            for trace_id, trace_df in df.groupby("trace_id"):
                # Use trace_id as pseudo timestamp
                windows[
                    pd.Timestamp.now().replace(microsecond=hash(trace_id) % 1000000)
                ] = trace_df
            return windows

        # Ensure time column is datetime type
        df = df.copy()
        if not pd.api.types.is_datetime64_any_dtype(df["time"]):
            df["time"] = pd.to_datetime(df["time"])

        # Group by time windows
        df = df.sort_values("time")
        start_time = df["time"].min()
        end_time = df["time"].max()

        windows = {}
        current_start = start_time

        while current_start <= end_time:
            current_end = current_start + pd.Timedelta(seconds=window_size_seconds)

            # Get data within current time window
            window_mask = (df["time"] >= current_start) & (df["time"] < current_end)
            window_data = df[window_mask]

            if not window_data.empty:
                windows[current_start] = window_data

            current_start = current_end

        return windows

    def build_graphs(self) -> List[Data]:
        """Build graph dataset based on static call graph, each graph instance contains actual call data within time window.

        Returns:
            List[Data]: List of graph data, each graph represents call instances within a time window
        """
        if not self.is_fitted:
            raise RuntimeError("Processor has not been fitted. Call fit() first.")

        if self.service_graph is None:
            raise RuntimeError("Service graph not built. Call fit() first.")

        trace_data = self._get_current_trace_data()
        if trace_data is None:
            raise RuntimeError("No trace data available for graph building.")

        graphs = []

        all_span_names = list(self.service_graph.nodes())

        if not all_span_names:
            raise RuntimeError("No span names found in service graph.")

        span_to_idx = {span: idx for idx, span in enumerate(all_span_names)}
        num_nodes = len(all_span_names)

        # 按20秒时间窗口分组trace数据
        time_windows = self._create_time_windows(trace_data, window_size_seconds=20)

        for window_start, window_data in time_windows.items():
            if window_data.empty:
                continue

            # 构建节点特征：每个span的实际调用统计
            node_features = []
            for span_name in all_span_names:
                span_data = window_data[window_data["span_name"] == span_name]
                node_feat = self._build_span_node_features(span_data, window_data)
                node_features.append(node_feat)

            node_features = torch.tensor(node_features, dtype=torch.float32)

            # 构建边特征和边索引：只包含实际存在的调用关系
            edge_list = []
            edge_features = []

            # 统计实际调用关系
            span_id_to_name = dict(
                zip(window_data["span_id"], window_data["span_name"])
            )
            call_stats = {}  # (parent_span, child_span) -> statistics

            for _, row in window_data.iterrows():
                if (
                    pd.notna(row["parent_span_id"])
                    and row["parent_span_id"] in span_id_to_name
                ):
                    parent_span = span_id_to_name[row["parent_span_id"]]
                    current_span = row["span_name"]
                    call_key = (parent_span, current_span)

                    if call_key not in call_stats:
                        call_stats[call_key] = {
                            "call_count": 0,
                            "total_duration": 0.0,
                            "error_count": 0,
                            "durations": [],
                        }

                    call_stats[call_key]["call_count"] += 1
                    call_stats[call_key]["total_duration"] += row.get("duration", 0)
                    call_stats[call_key]["durations"].append(row.get("duration", 0))

                    # 检查是否为错误调用
                    status_code = row.get("attr.http.response.status_code", 200)
                    if pd.notna(status_code) and status_code >= 400:
                        call_stats[call_key]["error_count"] += 1

            # 构建边索引和边特征
            for (parent_span, child_span), stats in call_stats.items():
                if parent_span in span_to_idx and child_span in span_to_idx:
                    src_idx = span_to_idx[parent_span]
                    dst_idx = span_to_idx[child_span]

                    edge_list.append([src_idx, dst_idx])

                    # 边特征：调用统计信息
                    avg_duration = stats["total_duration"] / stats["call_count"]
                    error_rate = stats["error_count"] / stats["call_count"]
                    duration_std = (
                        np.std(stats["durations"])
                        if len(stats["durations"]) > 1
                        else 0.0
                    )

                    edge_feat = [
                        stats["call_count"],  # 调用次数
                        avg_duration,  # 平均持续时间
                        duration_std,  # 持续时间标准差
                        error_rate,  # 错误率
                        stats["total_duration"],  # 总持续时间
                    ]
                    edge_features.append(edge_feat)

            # 创建边索引张量
            if edge_list:
                edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
                edge_attr = torch.tensor(edge_features, dtype=torch.float32)
            else:
                edge_index = torch.empty((2, 0), dtype=torch.long)
                edge_attr = torch.empty((0, 5), dtype=torch.float32)  # 5个边特征

            # 添加自环（可选）
            if self.trace_config.add_self_loops and num_nodes > 0:
                self_loops = torch.arange(num_nodes).unsqueeze(0).repeat(2, 1)
                edge_index = torch.cat([edge_index, self_loops], dim=1)

                # 为自环添加默认特征
                self_loop_features = torch.zeros((num_nodes, 5), dtype=torch.float32)
                edge_attr = torch.cat([edge_attr, self_loop_features], dim=0)

            # 创建图数据对象
            graph_data = Data(
                x=node_features,
                edge_index=edge_index,
                edge_attr=edge_attr,
                num_nodes=num_nodes,
                window_start=window_start,  # 直接作为属性保存
            )

            # 为了调试目的，可以将额外信息保存为简单属性
            graph_data._metadata = {
                "span_names": all_span_names,
                "num_call_relationships": len(call_stats),
            }

            graphs.append(graph_data)

        return graphs

    def _get_current_trace_data(self) -> pd.DataFrame | None:
        if hasattr(self, "_current_data") and self._current_data is not None:
            return self._current_data
        elif hasattr(self, "_original_data") and self._original_data is not None:
            return self._original_data
        else:
            return None

    def _build_span_node_features(
        self, span_data: pd.DataFrame, trace_data: pd.DataFrame
    ) -> List[float]:
        if span_data.empty:
            return [0.0] * 8

        features = []

        call_count = len(span_data)
        features.append(float(call_count))

        # 2. 总持续时间
        total_duration = span_data["duration"].sum()
        features.append(float(total_duration) if pd.notna(total_duration) else 0.0)

        # 3. 平均持续时间
        avg_duration = span_data["duration"].mean()
        features.append(float(avg_duration) if pd.notna(avg_duration) else 0.0)

        # 4. 持续时间标准差
        duration_std = span_data["duration"].std() if len(span_data) > 1 else 0.0
        features.append(float(duration_std) if pd.notna(duration_std) else 0.0)

        # 5. 错误率
        if "attr.http.response.status_code" in span_data.columns:
            status_codes = pd.to_numeric(
                span_data["attr.http.response.status_code"], errors="coerce"
            )
            error_rate = (
                (status_codes >= 400).mean() if not status_codes.isna().all() else 0.0
            )
        else:
            error_rate = 0.0
        features.append(float(error_rate))

        # 6. 在trace中的相对频率
        trace_span_count = len(trace_data)
        relative_frequency = (
            call_count / trace_span_count if trace_span_count > 0 else 0.0
        )
        features.append(float(relative_frequency))

        # 7. 最大持续时间
        max_duration = span_data["duration"].max()
        features.append(float(max_duration) if pd.notna(max_duration) else 0.0)

        # 8. 最小持续时间
        min_duration = span_data["duration"].min()
        features.append(float(min_duration) if pd.notna(min_duration) else 0.0)

        return features

    def get_node_feature_dim(self) -> int:
        """获取节点特征的维度"""
        return 8  # 对应_build_span_node_features中定义的8个特征

    def get_feature_info(self) -> Dict[str, Any]:
        if not self.is_fitted:
            raise RuntimeError("Processor has not been fitted. Call fit() first.")

        info = {
            "categorical_features": self.config.categorical_features,
            "numerical_features": self.config.numerical_features,
            "categorical_encoder_features": getattr(
                self.ohe, "feature_names_in_", None
            ),
            "numerical_scaler_features": getattr(
                self.scaler, "feature_names_in_", None
            ),
            "span_name_mapping": self.span_name_to_idx,
            "service_graph_info": {
                "num_nodes": len(self.service_graph.nodes())
                if self.service_graph
                else 0,
                "num_edges": len(self.service_graph.edges())
                if self.service_graph
                else 0,
            },
        }

        return info
