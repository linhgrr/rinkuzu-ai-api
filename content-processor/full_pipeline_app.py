"""
Streamlit app để chạy FULL pipeline: PDF → Concepts → Graph.

Usage:
    streamlit run full_pipeline_app.py
"""
import sys
from pathlib import Path

# Thêm src vào sys.path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import streamlit as st
import tempfile
import hashlib
import pickle
import json
from io import BytesIO
import networkx as nx
import plotly.graph_objects as go

from config import settings as config_settings
from processors.loaders.pdf_loader import PDFLoader
from processors.chunkers.text_chunker import TextChunker
from llm import get_llm
from llm.extract_chain import ExtractionChain
from llm.schemas import Relation
from embed.embedding_client import EmbeddingClient
from embed.embeddings import compute_embedding_for_concepts
from embed.prereq_ranking import rank_prerequisites
from merge import merge_by_name, deduplicate_by_embedding
from graph.builder import KnowledgeGraphBuilder
from graph.reduction import apply_transitive_reduction
from graph.cycle_removal import make_dag_with_llm


# Cache directory setup
CACHE_DIR = Path(__file__).parent / ".cache" / "chunks"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Page config
st.set_page_config(
    page_title="Full Pipeline - Knowledge Graph Builder",
    page_icon="🚀",
    layout="wide"
)

st.title("🚀 Full Pipeline: PDF → Knowledge Graph")
st.markdown("Tự động xử lý PDF và build Knowledge Graph với đầy đủ các bước.")
st.markdown("---")

# ==================== CACHE FUNCTIONS ====================


def get_cache_key(file_name: str, file_size: int, chunk_size: int, chunk_overlap: int) -> str:
    """Tạo cache key dựa trên thông tin file và chunking settings."""
    key_string = f"{file_name}_{file_size}_{chunk_size}_{chunk_overlap}"
    return hashlib.md5(key_string.encode()).hexdigest()


def save_chunks_to_cache(cache_key: str, chunks: list, content: dict):
    """Lưu chunks và content vào cache."""
    cache_file = CACHE_DIR / f"{cache_key}.pkl"
    try:
        with open(cache_file, 'wb') as f:
            pickle.dump({'chunks': chunks, 'content': content}, f)
        return True
    except Exception as e:
        st.warning(f"⚠️ Không thể lưu cache: {str(e)}")
        return False


def load_chunks_from_cache(cache_key: str):
    """Load chunks và content từ cache."""
    cache_file = CACHE_DIR / f"{cache_key}.pkl"
    if cache_file.exists():
        try:
            with open(cache_file, 'rb') as f:
                data = pickle.load(f)
            return data['chunks'], data['content']
        except Exception as e:
            st.warning(f"⚠️ Không thể đọc cache: {str(e)}")
            return None, None
    return None, None


def get_cache_info():
    """Lấy thông tin về cache hiện có."""
    cache_files = list(CACHE_DIR.glob("*.pkl"))
    total_size = sum(f.stat().st_size for f in cache_files)
    return len(cache_files), total_size


def clear_cache():
    """Xóa toàn bộ cache."""
    cache_files = list(CACHE_DIR.glob("*.pkl"))
    for f in cache_files:
        try:
            f.unlink()
        except Exception as e:
            st.warning(f"⚠️ Không thể xóa {f.name}: {str(e)}")
    return len(cache_files)


# ==================== GRAPH VISUALIZATION ====================

def create_hierarchical_layout(graph: nx.DiGraph) -> dict:
    """
    Tạo hierarchical layout cho DAG theo dạng top-down.
    Root nodes (không có prerequisites) ở trên cùng, build xuống dưới theo levels.

    Returns:
        dict: {node_id: (x, y)}
    """
    if graph.number_of_nodes() == 0:
        return {}

    # Compute levels using topological generation
    # Level 0 = root nodes (không có incoming edges)
    levels = {}

    # Find all nodes at each level
    for node in graph.nodes():
        # Level = longest path from any root to this node
        try:
            # Get all predecessors (ancestors)
            ancestors = nx.ancestors(graph, node)
            if not ancestors:
                # No ancestors = root node
                levels[node] = 0
            else:
                # Level = max level of all parents + 1
                max_parent_level = -1
                for parent in graph.predecessors(node):
                    if parent in levels:
                        max_parent_level = max(
                            max_parent_level, levels[parent])

                if max_parent_level >= 0:
                    levels[node] = max_parent_level + 1
                else:
                    # Fallback: compute from ancestors
                    max_level = 0
                    for ancestor in ancestors:
                        if ancestor in levels:
                            max_level = max(max_level, levels[ancestor])
                    levels[node] = max_level + 1
        except:
            levels[node] = 0

    # If no levels computed yet, do BFS from nodes with no incoming edges
    if not levels:
        root_nodes = [n for n in graph.nodes() if graph.in_degree(n) == 0]
        if not root_nodes:
            # Graph has cycles or all nodes have incoming edges
            # Use any node as root
            root_nodes = [list(graph.nodes())[0]]

        for root in root_nodes:
            levels[root] = 0

        # BFS to assign levels
        visited = set(root_nodes)
        queue = [(root, 0) for root in root_nodes]

        while queue:
            node, level = queue.pop(0)
            for successor in graph.successors(node):
                if successor not in visited:
                    levels[successor] = level + 1
                    visited.add(successor)
                    queue.append((successor, level + 1))
                else:
                    # Update level if we found a longer path
                    levels[successor] = max(levels[successor], level + 1)

    # Group nodes by level
    level_nodes = {}
    for node, level in levels.items():
        if level not in level_nodes:
            level_nodes[level] = []
        level_nodes[level].append(node)

    # Compute positions
    pos = {}
    max_level = max(levels.values()) if levels else 0

    for level, nodes in level_nodes.items():
        # Y coordinate: top-down (root ở trên = y cao nhất)
        # Root nodes (level 0) sẽ có y = max_level (cao nhất)
        y = max_level - level

        # X coordinates: spread evenly
        num_nodes = len(nodes)
        if num_nodes == 1:
            x_positions = [0]
        else:
            # Spread from -1 to 1
            x_positions = [(-1 + 2 * i / (num_nodes - 1))
                           for i in range(num_nodes)]

        # Sort nodes by name for consistent positioning
        sorted_nodes = sorted(
            nodes, key=lambda n: graph.nodes[n].get('name', n))

        for i, node in enumerate(sorted_nodes):
            pos[node] = (x_positions[i], y)

    return pos


def find_all_prerequisite_paths(graph: nx.DiGraph, target_node: str) -> list:
    """
    Tìm tất cả các path cần học trước khi đến một node.

    Args:
        graph: NetworkX DiGraph
        target_node: Node ID đích

    Returns:
        list: Danh sách các paths, mỗi path là list của node IDs từ root đến target
    """
    if target_node not in graph.nodes():
        return []

    # Tìm tất cả các root nodes (không có prerequisites)
    root_nodes = [n for n in graph.nodes() if graph.in_degree(n) == 0]

    # Tìm tất cả ancestors (nodes có thể đến được target)
    ancestors = nx.ancestors(graph, target_node)

    # Nếu không có ancestors, node này là root
    if not ancestors:
        return [[target_node]]

    # Tìm các root nodes có thể đến được target
    reachable_roots = [
        r for r in root_nodes if r in ancestors or r == target_node]

    # Nếu không có root nào đến được, tìm các nodes không có predecessor trong ancestors
    if not reachable_roots:
        # Tìm nodes trong ancestors không có predecessor nào cũng trong ancestors
        possible_starts = []
        for node in ancestors:
            preds = set(graph.predecessors(node))
            if not preds.intersection(ancestors):
                possible_starts.append(node)
        reachable_roots = possible_starts if possible_starts else list(ancestors)[
            :1]

    # Tìm tất cả simple paths từ mỗi root đến target
    all_paths = []
    for root in reachable_roots:
        if root == target_node:
            all_paths.append([target_node])
        else:
            try:
                paths = list(nx.all_simple_paths(graph, root, target_node))
                all_paths.extend(paths)
            except nx.NetworkXNoPath:
                continue

    # Sắp xếp paths theo độ dài
    all_paths.sort(key=len)

    return all_paths


def format_learning_paths(graph: nx.DiGraph, paths: list) -> str:
    """
    Format các learning paths thành text để hiển thị.

    Args:
        graph: NetworkX DiGraph
        paths: List of paths (mỗi path là list of node IDs)

    Returns:
        str: Formatted text
    """
    if not paths:
        return "Không có prerequisite paths (đây là root concept)"

    lines = []
    lines.append(f"**Tìm thấy {len(paths)} learning path(s):**\n")

    for i, path in enumerate(paths, 1):
        # Get node names
        node_names = []
        for node_id in path:
            node_data = graph.nodes[node_id]
            node_name = node_data.get('name', node_id)
            node_names.append(node_name)

        # Format path
        path_str = " → ".join(node_names)
        lines.append(f"{i}. {path_str}")
        lines.append(f"   _(Độ dài: {len(path)} concepts)_\n")

    return "\n".join(lines)


def create_graph_visualization(graph: nx.DiGraph):
    """
    Tạo visualization cho graph bằng Plotly với hierarchical top-down layout.

    Args:
        graph: NetworkX DiGraph
    """
    if graph.number_of_nodes() == 0:
        return None

    # Compute hierarchical layout
    pos = create_hierarchical_layout(graph)

    # Create edge traces with hover info and arrows
    edge_trace = []
    edge_annotations = []  # For arrow heads

    for edge in graph.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]

        # Get edge data
        edge_data = graph.edges[edge]
        source_name = graph.nodes[edge[0]].get('name', edge[0])
        target_name = graph.nodes[edge[1]].get('name', edge[1])

        # Build hover text
        hover_parts = []
        hover_parts.append(f"<b>{source_name}</b> → <b>{target_name}</b>")

        relation_type = edge_data.get('relation_type', 'UNKNOWN')
        hover_parts.append(f"<br>Type: {relation_type}")

        evidence = edge_data.get('evidence')
        if evidence:
            if isinstance(evidence, list) and evidence:
                # Show first evidence
                ev_text = evidence[0]
                if len(ev_text) > 150:
                    ev_text = ev_text[:150] + '...'
                hover_parts.append(f"<br><br>Evidence: {ev_text}")
                if len(evidence) > 1:
                    hover_parts.append(
                        f"<br>+ {len(evidence) - 1} more evidence(s)")
            elif isinstance(evidence, str):
                if len(evidence) > 150:
                    evidence = evidence[:150] + '...'
                hover_parts.append(f"<br><br>Evidence: {evidence}")

        hover_text = "".join(hover_parts)

        edge_trace.append(
            go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                mode='lines',
                line=dict(width=1.5, color='#888'),
                hovertext=hover_text,
                hoverinfo='text',
                showlegend=False
            )
        )

        # Add arrow annotation
        # Calculate arrow position (80% along the edge towards target)
        arrow_x = x0 + 0.8 * (x1 - x0)
        arrow_y = y0 + 0.8 * (y1 - y0)

        edge_annotations.append(
            dict(
                x=arrow_x,
                y=arrow_y,
                ax=x0 + 0.7 * (x1 - x0),
                ay=y0 + 0.7 * (y1 - y0),
                xref='x',
                yref='y',
                axref='x',
                ayref='y',
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor='#888',
                standoff=0
            )
        )

    # Create node trace
    node_x = []
    node_y = []
    node_labels = []  # Text hiển thị trên node
    node_hover = []   # Text hiển thị khi hover
    node_color = []

    for node in graph.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)

        node_data = graph.nodes[node]
        node_name = node_data.get('name', node)
        node_labels.append(node_name)

        # Build hover text with detailed information
        hover_parts = []
        hover_parts.append(f"<b>{node_name}</b>")
        hover_parts.append(f"<i>ID: {node}</i>")

        # Definition
        definition = node_data.get('definition', '')
        if definition:
            # Truncate long definitions
            if len(definition) > 200:
                definition = definition[:200] + '...'
            hover_parts.append(f"<br><br>{definition}")

        # Examples
        examples = node_data.get('examples', [])
        if examples:
            hover_parts.append(f"<br><br><b>Examples:</b>")
            for i, example in enumerate(examples[:3]):  # Show max 3 examples
                if len(example) > 100:
                    example = example[:100] + '...'
                hover_parts.append(f"<br>• {example}")
            if len(examples) > 3:
                hover_parts.append(f"<br>• ... and {len(examples) - 3} more")

        # Relations (incoming and outgoing)
        num_predecessors = graph.in_degree(node)
        num_successors = graph.out_degree(node)
        hover_parts.append(f"<br><br><b>Relations:</b>")
        hover_parts.append(f"<br>↓ Prerequisites: {num_predecessors}")
        hover_parts.append(f"<br>↑ Leads to: {num_successors}")

        # Is placeholder
        is_placeholder = node_data.get('is_placeholder', False)
        if is_placeholder:
            hover_parts.append(f"<br><br><i>⚠️ Placeholder node</i>")

        node_hover.append("".join(hover_parts))

        # Color by node properties
        node_color.append('#ff7f0e' if is_placeholder else '#1f77b4')

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode='markers+text',
        text=node_labels,
        textposition='top center',
        hovertext=node_hover,
        hoverinfo='text',
        marker=dict(
            size=20,
            color=node_color,
            line=dict(width=2, color='white')
        ),
        showlegend=False
    )

    # Calculate height based on number of levels
    max_y = max(y for _, y in pos.values()) if pos else 0
    # 150px per level, min 600, max 1200
    height = max(600, min(1200, 150 * (max_y + 2)))

    # Create figure
    fig = go.Figure(
        data=edge_trace + [node_trace],
        layout=go.Layout(
            showlegend=False,
            hovermode='closest',
            margin=dict(b=0, l=0, r=0, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            height=height,
            hoverlabel=dict(
                bgcolor="white",
                font_size=12,
                font_family="monospace",
                align="left"
            ),
            annotations=edge_annotations
        )
    )

    return fig


# ==================== SIDEBAR CONFIGURATION ====================

st.sidebar.header("⚙️ Configuration")

# Pipeline Settings
st.sidebar.subheader("📝 Chunking")
chunk_size = st.sidebar.slider("Chunk Size", 500, 3000, 1500, 100)
chunk_overlap = st.sidebar.slider("Chunk Overlap", 0, 500, 200, 50)

st.sidebar.subheader("⚡ Processing")
batch_size = st.sidebar.slider(
    "Batch Size", 1, 10, 3, 1, help="Số chunk gộp trong 1 prompt")
max_workers = st.sidebar.slider(
    "Max Workers", 1, 8, 4, 1, help="Số worker xử lý song song")
max_previous_concepts = st.sidebar.slider(
    "Prev. Concepts Window", 0, 50, 20, 5,
    help="Số concept đã trích xuất (gần nhất) đưa vào prompt của mỗi batch để tránh trùng lặp. 0 = tắt.")

st.sidebar.subheader("🔀 Merging")
enable_name_merge = st.sidebar.checkbox("Enable Name-based Merge", value=True)
# Bỏ embedding-based merge - sẽ dùng LLM để merge thông qua verification
# enable_embedding_merge = st.sidebar.checkbox("Enable Embedding-based Merge", value=True)
# similarity_threshold = st.sidebar.slider("Similarity Threshold", 0.7, 1.0, 0.9, 0.05)

st.sidebar.subheader("🔗 Prerequisite Ranking")
prs_threshold = st.sidebar.slider(
    "PRS Threshold", 0.5, 1.0, config_settings.prs_threshold, 0.05)

st.sidebar.subheader("🔍 Verification")
min_confidence = st.sidebar.slider(
    "Min Confidence for Verification", 0.0, 1.0, 0.5, 0.1)

st.sidebar.subheader("📊 Graph")
apply_reduction = st.sidebar.checkbox("Apply Transitive Reduction", value=True)

# Cache Management
st.sidebar.subheader("💾 Cache Management")
num_cached, cache_size = get_cache_info()
st.sidebar.info(
    f"📦 **{num_cached}** file(s) cached\n\n💽 **{cache_size / 1024:.2f}** KB")
if st.sidebar.button("🗑️ Clear Cache", use_container_width=True):
    cleared = clear_cache()
    st.sidebar.success(f"✅ Đã xóa {cleared} file(s) cache!")
    st.rerun()

# ==================== INITIALIZE COMPONENTS ====================


@st.cache_resource
def init_extraction_chain():
    """Initialize extraction chain with LLM."""
    try:
        llm = get_llm(
            temperature=0.1,
        )
        chain = ExtractionChain(client=llm)
        return chain
    except Exception as e:
        st.error(f"❌ Error initializing extraction chain: {str(e)}")
        return None


@st.cache_resource
def init_embeddings():
    """Initialize embeddings model."""
    try:
        embedding_client = EmbeddingClient()
        st.sidebar.success(
            f"✅ Embeddings: {embedding_client.model_name} on {embedding_client.device}"
        )
        return embedding_client
    except Exception as e:
        st.error(f"❌ Error initializing embeddings: {str(e)}")
        return None


chain = init_extraction_chain()
embeddings = init_embeddings()

# ==================== MAIN APP ====================

# Initialize session state for storing results
if 'graph' not in st.session_state:
    st.session_state.graph = None
if 'concepts' not in st.session_state:
    st.session_state.concepts = None
if 'all_concepts' not in st.session_state:
    st.session_state.all_concepts = None
if 'qualifying_relations' not in st.session_state:
    st.session_state.qualifying_relations = None
if 'final_stats' not in st.session_state:
    st.session_state.final_stats = None
if 'initial_stats' not in st.session_state:
    st.session_state.initial_stats = None
if 'subject_id' not in st.session_state:
    st.session_state.subject_id = None

# File upload
uploaded_file = st.file_uploader(
    "📄 Upload PDF file",
    type=['pdf'],
    help="Upload a PDF document to build knowledge graph"
)

if uploaded_file:
    # Save to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    col1, col2 = st.columns([3, 1])

    with col1:
        st.success(f"✅ Uploaded: **{uploaded_file.name}**")
        st.caption(f"Size: {uploaded_file.size / 1024:.2f} KB")

    with col2:
        # Generate cache key
        cache_key = get_cache_key(
            uploaded_file.name,
            uploaded_file.size,
            chunk_size,
            chunk_overlap
        )

        # Check cache
        cached_chunks, cached_content = load_chunks_from_cache(cache_key)
        if cached_chunks is not None:
            st.success("💾 Cache available")

    # Chunk selection
    if cached_chunks is not None:
        total_chunks = len(cached_chunks)
    else:
        total_chunks = 100  # Estimate

    num_chunks = st.number_input(
        "📊 Number of chunks to process",
        min_value=1,
        max_value=total_chunks,
        value=min(9, total_chunks),
        help="Processing more chunks will take longer"
    )

    subject_id = st.text_input(
        "🎓 Subject ID",
        value=uploaded_file.name.replace('.pdf', ''),
        help="ID của môn học cho knowledge graph"
    )

    st.markdown("---")

    # Run pipeline button
    run_pipeline_btn = st.button(
        "🚀 RUN FULL PIPELINE",
        type="primary",
        use_container_width=True
    )

    if run_pipeline_btn:
        if not chain or not embeddings:
            st.error("❌ Components not initialized!")
        else:
            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()

            try:
                # ========== STEP 1: Load & Chunk PDF ==========
                status_text.text("📄 Step 1/7: Loading PDF...")
                progress_bar.progress(1/7)

                # Try to load from cache
                if cached_chunks is not None and cached_content is not None:
                    chunks = cached_chunks
                    content = cached_content
                    st.info(f"💾 Loaded {len(chunks)} chunks from cache")
                else:
                    # Load and chunk
                    loader = PDFLoader()
                    content = loader.load(tmp_path)

                    chunker = TextChunker(
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap
                    )
                    chunks = chunker.chunk(content, doc_id=uploaded_file.name)

                    # Save to cache
                    save_chunks_to_cache(cache_key, chunks, content)
                    st.info(f"📦 Created {len(chunks)} chunks")

                # ========== STEP 2: Extract Concepts ==========
                status_text.text("🔬 Step 2/7: Extracting concepts...")
                progress_bar.progress(2/7)

                chunk_texts = [
                    chunk.page_content for chunk in chunks[:num_chunks]]

                extractions = chain.extract_from_batch(
                    chunks=chunk_texts,
                    subject_id=subject_id,
                    batch_size=batch_size,
                    max_workers=max_workers,
                    max_previous_concepts=max_previous_concepts,
                )

                all_concepts = []
                for extraction in extractions:
                    all_concepts.extend(extraction.concepts)

                st.success(
                    f"✅ Extracted {len(all_concepts)} concepts from {num_chunks} chunks")

                # ========== STEP 3: Generate Embeddings ==========
                status_text.text("📊 Step 3/7: Generating embeddings...")
                progress_bar.progress(3/7)

                compute_embedding_for_concepts(
                    concepts=all_concepts,
                    client=embeddings
                )

                st.success(
                    f"✅ Generated embeddings for {len(all_concepts)} concepts")

                # ========== STEP 4: Merge by Name ==========
                status_text.text("🔀 Step 4/7: Merging by name...")
                progress_bar.progress(4/7)

                concepts = all_concepts.copy()

                if enable_name_merge:
                    concepts = merge_by_name(concepts)
                    reduction = len(all_concepts) - len(concepts)
                    reduction_pct = (reduction / len(all_concepts)
                                     ) * 100 if all_concepts else 0
                    st.success(
                        f"✅ Name merge: reduced from {len(all_concepts)} to {len(concepts)} concepts ({reduction_pct:.1f}% reduction)")
                else:
                    st.info("ℹ️ Name merge disabled")

                # ========== STEP 5: Find Prerequisite Pairs ==========
                status_text.text("🔗 Step 5/7: Finding prerequisite pairs...")
                progress_bar.progress(5/7)

                prereq_pairs = rank_prerequisites(
                    concepts=concepts,
                    prs_threshold=prs_threshold
                )

                st.success(
                    f"✅ Found {len(prereq_pairs)} prerequisite candidate pairs")

                # ========== STEP 6: Verify Relations ==========
                status_text.text("🔍 Step 6/7: Verifying relations...")
                progress_bar.progress(6/7)

                if prereq_pairs:
                    concept_map = {c.concept_id: c for c in concepts}

                    # Prepare pairs for verification
                    concept_pairs = []
                    pair_metadata = []

                    for id1, id2 in prereq_pairs:
                        c1 = concept_map.get(id1)
                        c2 = concept_map.get(id2)

                        if c1 and c2:
                            concept_pairs.append((c1.name, c2.name))
                            pair_metadata.append({
                                'concept_a_id': id1,
                                'concept_b_id': id2,
                                'concept_a_name': c1.name,
                                'concept_b_name': c2.name
                            })

                    # Verify all pairs in parallel
                    verifications = chain.verify_relations_batch(
                        concept_pairs=concept_pairs,
                        max_workers=max_workers
                    )

                    # Combine results
                    verification_results = []
                    for metadata, verification in zip(pair_metadata, verifications):
                        verification_results.append({
                            **metadata,
                            'verification': verification
                        })

                    # Separate same_concept pairs from relation pairs
                    same_concept_pairs = [
                        r for r in verification_results
                        if r['verification'].has_relation
                        and r['verification'].direction == "same_concept"
                        and r['verification'].confidence >= min_confidence
                    ]

                    qualifying_relations = [
                        r for r in verification_results
                        if r['verification'].has_relation
                        and r['verification'].direction != "same_concept"
                        and r['verification'].confidence >= min_confidence
                    ]

                    st.success(
                        f"✅ Verified {len(verification_results)} pairs: {len(same_concept_pairs)} same concepts, {len(qualifying_relations)} relations (confidence ≥ {min_confidence:.1f})")
                else:
                    qualifying_relations = []
                    same_concept_pairs = []
                    st.warning("⚠️ No prerequisite pairs to verify")

                # ========== STEP 6.5: Merge Same Concepts ==========
                if same_concept_pairs:
                    status_text.text(
                        "🔀 Step 6.5/7: Merging same concepts identified by LLM...")
                    progress_bar.progress(6.5/7)

                    # Build merge groups using union-find
                    concept_map = {c.concept_id: c for c in concepts}
                    parent = {c.concept_id: c.concept_id for c in concepts}

                    def find(x):
                        if parent[x] != x:
                            parent[x] = find(parent[x])
                        return parent[x]

                    def union(a, b):
                        ra, rb = find(a), find(b)
                        if ra != rb:
                            parent[rb] = ra

                    # Union same concept pairs
                    for result in same_concept_pairs:
                        concept_a_id = result['concept_a_id']
                        concept_b_id = result['concept_b_id']
                        union(concept_a_id, concept_b_id)

                    # Group concepts by root
                    merge_groups = {}
                    for concept_id in parent.keys():
                        root = find(concept_id)
                        if root not in merge_groups:
                            merge_groups[root] = []
                        merge_groups[root].append(concept_id)

                    # Merge each group
                    id_map = {}
                    merged_concepts = []

                    for root, group_ids in merge_groups.items():
                        if len(group_ids) == 1:
                            # No merge needed
                            c = concept_map[group_ids[0]]
                            id_map[c.concept_id] = c.concept_id
                            merged_concepts.append(c)
                        else:
                            # Merge group
                            group_concepts = [concept_map[cid]
                                              for cid in group_ids]
                            # Use existing merge logic from name_merge
                            from merge.name_merge import _merge_concepts
                            merged, group_id_map = _merge_concepts(
                                group_concepts)
                            merged_concepts.append(merged)
                            id_map.update(group_id_map)

                    # Remap relations in all merged concepts
                    from merge.name_merge import _remap_relations
                    final_concepts = []
                    for c in merged_concepts:
                        c = c.copy(deep=True)
                        c.relations = _remap_relations(
                            c.relations, id_map, self_id=c.concept_id)
                        final_concepts.append(c)

                    num_before = len(concepts)
                    concepts = final_concepts
                    num_after = len(concepts)
                    reduction = num_before - num_after

                    st.success(
                        f"✅ LLM-based merge: reduced from {num_before} to {num_after} concepts ({reduction} merged)")

                    # Remap qualifying_relations to use new concept_ids
                    remapped_qualifying_relations = []
                    for result in qualifying_relations:
                        concept_a_id = id_map.get(
                            result['concept_a_id'], result['concept_a_id'])
                        concept_b_id = id_map.get(
                            result['concept_b_id'], result['concept_b_id'])

                        # Skip self-loops
                        if concept_a_id == concept_b_id:
                            continue

                        remapped_qualifying_relations.append({
                            **result,
                            'concept_a_id': concept_a_id,
                            'concept_b_id': concept_b_id
                        })

                    qualifying_relations = remapped_qualifying_relations

                # ========== STEP 7: Build Knowledge Graph ==========
                status_text.text("📊 Step 7/7: Building knowledge graph...")
                progress_bar.progress(7/7)

                # Initialize graph builder
                builder = KnowledgeGraphBuilder(subject_id=subject_id)

                # Create concept map (needed for adding relations)
                concept_map = {c.concept_id: c for c in concepts}

                # Add verified relations to concepts
                for result in qualifying_relations:
                    verification = result['verification']
                    concept_a_id = result['concept_a_id']
                    concept_b_id = result['concept_b_id']

                    # Skip same_concept (already handled in merge step)
                    if verification.direction == "same_concept":
                        continue

                    # Determine direction and add relation
                    if verification.direction == "A_to_B":
                        source_id = concept_a_id
                        target_id = concept_b_id
                    elif verification.direction == "B_to_A":
                        source_id = concept_b_id
                        target_id = concept_a_id
                    else:
                        continue

                    # Add relation to concept
                    if source_id in concept_map:
                        source_concept = concept_map[source_id]
                        existing_targets = {
                            rel.target_id for rel in source_concept.relations}

                        if target_id not in existing_targets:
                            new_relation = Relation(
                                type="PREREQUISITE",
                                target_id=target_id,
                                confidence=verification.confidence,
                                evidence="\n".join(
                                    verification.evidences) if verification.evidences else None
                            )
                            source_concept.relations.append(new_relation)

                # Add all concepts to graph
                builder.add_concepts(list(concept_map.values()))

                # Get initial stats
                graph = builder.get_graph()
                initial_stats = builder.get_stats()

                # Convert to DAG by removing cycles (using LLM)
                dag_stats = None
                if not nx.is_directed_acyclic_graph(graph):
                    st.info("🔄 Graph contains cycles, using LLM to remove them...")
                    graph, dag_stats = make_dag_with_llm(graph, llm=chain.llm)
                    builder.graph = graph

                    if dag_stats.get('edges_removed', 0) > 0:
                        st.success(
                            f"✅ Removed {dag_stats['edges_removed']} edge(s) from "
                            f"{dag_stats['cycles_removed']} cycle(s) to create DAG"
                        )

                # Apply transitive reduction if requested
                if apply_reduction:
                    graph = apply_transitive_reduction(graph)
                    builder.graph = graph

                # Get final stats
                final_stats = builder.get_stats()

                # Store DAG stats
                if dag_stats:
                    final_stats['dag_conversion'] = dag_stats

                st.success("✅ Knowledge graph built successfully!")

                # ========== SAVE TO SESSION STATE ==========
                st.session_state.graph = graph
                st.session_state.concepts = concepts
                st.session_state.all_concepts = all_concepts
                st.session_state.qualifying_relations = qualifying_relations
                st.session_state.final_stats = final_stats
                st.session_state.initial_stats = initial_stats
                st.session_state.subject_id = subject_id

                # Clear progress
                progress_bar.empty()
                status_text.empty()

            except Exception as e:
                st.error(f"❌ Pipeline error: {str(e)}")
                st.exception(e)

# ========== DISPLAY RESULTS (Outside of button click) ==========
# This section uses session_state, so it persists across reruns
if st.session_state.graph is not None:
    graph = st.session_state.graph
    concepts = st.session_state.concepts
    all_concepts = st.session_state.all_concepts
    qualifying_relations = st.session_state.qualifying_relations
    final_stats = st.session_state.final_stats
    initial_stats = st.session_state.initial_stats
    subject_id = st.session_state.subject_id

    st.markdown("---")
    st.header("📊 Pipeline Results")

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Concepts Extracted", len(all_concepts))

    with col2:
        st.metric("After Merging", len(concepts))

    with col3:
        st.metric("Relations Verified", len(qualifying_relations))

    with col4:
        st.metric("Graph Edges", final_stats['num_edges'])

    # Graph Statistics
    st.markdown("---")
    st.subheader("📈 Graph Statistics")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Nodes", final_stats['num_nodes'])

    with col2:
        initial_edges = initial_stats.get(
            'num_edges', final_stats['num_edges'])
        current_edges = final_stats['num_edges']
        delta = current_edges - initial_edges if initial_edges != current_edges else None
        st.metric("Edges", current_edges, delta=delta)

    with col3:
        st.metric("Density", f"{final_stats['density']:.4f}")

    with col4:
        has_cycle = final_stats['has_cycle']
        cycle_status = "⚠️ Yes" if has_cycle else "✅ No"
        st.metric("Has Cycles", cycle_status)

    # Edge types
    if final_stats['edge_types']:
        st.markdown("**Edge Types:**")
        for edge_type, count in final_stats['edge_types'].items():
            st.caption(f"- {edge_type}: {count}")

    # DAG conversion info
    dag_conversion = final_stats.get('dag_conversion')
    if dag_conversion and dag_conversion.get('had_cycles'):
        cycles_removed = dag_conversion.get('cycles_removed', 0)
        edges_removed = dag_conversion.get('edges_removed', 0)
        iterations = dag_conversion.get('iterations', 0)

        st.warning(
            f"⚠️ **DAG Conversion**: Graph had cycles! LLM removed **{edges_removed}** edge(s) "
            f"from **{cycles_removed}** cycle(s) in {iterations} iteration(s)"
        )

    # Reduction info
    if initial_stats != final_stats and initial_stats.get('num_edges', 0) > final_stats['num_edges']:
        # Calculate reduction excluding DAG conversion
        dag_edges_removed = dag_conversion.get(
            'edges_removed', 0) if dag_conversion else 0
        initial_after_dag = initial_stats.get(
            'num_edges', 0) - dag_edges_removed

        if initial_after_dag > final_stats['num_edges']:
            reduced_count = initial_after_dag - final_stats['num_edges']
            reduction_pct = (reduced_count / initial_after_dag) * \
                100 if initial_after_dag > 0 else 0
            st.info(
                f"🔄 **Transitive Reduction**: Removed **{reduced_count}** redundant edges "
                f"({reduction_pct:.1f}% reduction)"
            )

    # Total optimization
    total_removed = initial_stats.get(
        'num_edges', 0) - final_stats['num_edges']
    if total_removed > 0:
        total_pct = (total_removed / initial_stats['num_edges']) * 100
        st.success(
            f"✅ **Total Optimization**: Removed **{total_removed}** edges total "
            f"({total_pct:.1f}% reduction from original graph)"
        )

    # Graph Visualization
    st.markdown("---")
    st.subheader("🔍 Graph Visualization")

    if graph.number_of_nodes() > 0:
        # Legend
        col1, col2 = st.columns([1, 3])
        with col1:
            st.markdown("**Node Colors:**")
        with col2:
            st.caption("🔵 Complete concepts | 🟠 Placeholder concepts")

        # Create visualization
        fig = create_graph_visualization(graph)
        if fig:
            st.plotly_chart(fig, use_container_width=True)

            st.info(
                "💡 **Top-Down Hierarchical Layout**: Root concepts (không có prerequisites) ở **trên cùng**. "
                "Concepts build xuống dưới theo dependency levels. Hover để xem chi tiết."
            )

        # Node selection for prerequisite paths
        st.markdown("---")
        st.subheader("🎯 Prerequisite Learning Paths")
        st.caption(
            "Chọn một concept để xem tất cả các path cần học trước khi đến được nó")

        # Create list of nodes with names
        node_options = {}
        for node_id in graph.nodes():
            node_data = graph.nodes[node_id]
            node_name = node_data.get('name', node_id)
            node_options[f"{node_name} (ID: {node_id})"] = node_id

        # Sort by name
        sorted_options = sorted(node_options.keys())

        # Node selector
        selected_display = st.selectbox(
            "Chọn concept:",
            options=["-- Chọn một concept --"] + sorted_options,
            key="node_selector"
        )

        if selected_display != "-- Chọn một concept --":
            selected_node_id = node_options[selected_display]
            selected_node_data = graph.nodes[selected_node_id]
            selected_node_name = selected_node_data.get(
                'name', selected_node_id)

            # Display selected node info
            st.markdown(f"### 📌 Selected: **{selected_node_name}**")

            # Show definition if available
            definition = selected_node_data.get('definition', '')
            if definition:
                st.info(f"**Definition:** {definition}")

            # Find and display prerequisite paths
            with st.spinner("Đang tìm learning paths..."):
                paths = find_all_prerequisite_paths(graph, selected_node_id)

            if paths:
                # Display paths
                st.markdown("### 📚 Learning Paths")
                formatted_paths = format_learning_paths(graph, paths)
                st.markdown(formatted_paths)

                # Show path statistics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Paths", len(paths))
                with col2:
                    avg_length = sum(len(p) for p in paths) / len(paths)
                    st.metric("Avg Path Length", f"{avg_length:.1f}")
                with col3:
                    max_length = max(len(p) for p in paths)
                    st.metric("Longest Path", max_length)

                # Show detailed paths in expander
                with st.expander("📋 View Detailed Paths", expanded=False):
                    for i, path in enumerate(paths, 1):
                        st.markdown(f"**Path {i}:**")
                        for j, node_id in enumerate(path):
                            node_data = graph.nodes[node_id]
                            node_name = node_data.get('name', node_id)
                            is_target = (j == len(path) - 1)

                            # Indent based on position
                            indent = "  " * j
                            arrow = "└→" if j > 0 else "●"
                            marker = "🎯" if is_target else "📖"

                            st.markdown(
                                f"{indent}{arrow} {marker} **{node_name}**")

                            # Show definition for intermediate nodes
                            if not is_target and node_data.get('definition'):
                                def_text = node_data['definition']
                                if len(def_text) > 100:
                                    def_text = def_text[:100] + "..."
                                st.caption(f"{indent}   _{def_text}_")

                        if i < len(paths):
                            st.divider()
            else:
                st.success(
                    "✅ Đây là **root concept** - không cần học trước concept nào!")
                st.info("Bạn có thể bắt đầu học concept này ngay.")
    else:
        st.warning("⚠️ Graph không có nodes")

    # Export Options
    st.markdown("---")
    st.subheader("💾 Export Results")

    col1, col2, col3 = st.columns(3)

    with col1:
        # Export concepts as JSON
        concepts_json = json.dumps(
            [c.model_dump() for c in concepts],
            indent=2,
            ensure_ascii=False
        )
        st.download_button(
            "📥 Concepts (JSON)",
            data=concepts_json,
            file_name=f"{subject_id}_concepts.json",
            mime="application/json",
            use_container_width=True
        )

    with col2:
        # Export graph as JSON
        graph_json = json.dumps(
            nx.node_link_data(graph),
            indent=2,
            ensure_ascii=False
        )
        st.download_button(
            "📥 Graph (JSON)",
            data=graph_json,
            file_name=f"{subject_id}_graph.json",
            mime="application/json",
            use_container_width=True
        )

    with col3:
        # Export as GraphML (convert lists to strings for GraphML compatibility)
        graphml_buffer = BytesIO()
        
        # Create a copy of the graph with serialized attributes
        graph_copy = graph.copy()
        for node_id, data in graph_copy.nodes(data=True):
            for key, value in list(data.items()):
                if isinstance(value, list):
                    # Convert list to JSON string
                    data[key] = json.dumps(value, ensure_ascii=False)
                elif isinstance(value, dict):
                    # Convert dict to JSON string
                    data[key] = json.dumps(value, ensure_ascii=False)
        
        for u, v, data in graph_copy.edges(data=True):
            for key, value in list(data.items()):
                if isinstance(value, list):
                    data[key] = json.dumps(value, ensure_ascii=False)
                elif isinstance(value, dict):
                    data[key] = json.dumps(value, ensure_ascii=False)
        
        nx.write_graphml(graph_copy, graphml_buffer)
        graphml_data = graphml_buffer.getvalue()

        st.download_button(
            "📥 Graph (GraphML)",
            data=graphml_data,
            file_name=f"{subject_id}_graph.graphml",
            mime="application/xml",
            use_container_width=True
        )

    # Concept List
    st.markdown("---")
    st.subheader("📚 Extracted Concepts")

    with st.expander(f"View {len(concepts)} concepts", expanded=False):
        for idx, concept in enumerate(concepts):
            st.markdown(f"**{idx+1}. {concept.name}**")
            st.caption(f"ID: `{concept.concept_id}`")
            if concept.definition:
                st.caption(f"_{concept.definition[:200]}..._" if len(
                    concept.definition) > 200 else f"_{concept.definition}_")

            if concept.relations:
                rel_text = ", ".join(
                    [f"{rel.target_id}" for rel in concept.relations])
                st.caption(f"Relations: {rel_text}")

            if idx < len(concepts) - 1:
                st.divider()

else:
    st.info("👆 Upload a PDF file to start building knowledge graph")

    # Show example workflow
    st.markdown("---")
    st.subheader("📋 Pipeline Workflow")

    steps = [
        "📄 **Load PDF** - Upload and chunk document",
        "🔬 **Extract Concepts** - LLM-based extraction from chunks",
        "📊 **Generate Embeddings** - Create vector embeddings for concepts",
        "🔀 **Merge by Name** - Merge concepts with same normalized name",
        "🔗 **Find Prerequisites** - Use PRS to find potential relations",
        "🔍 **Verify Relations** - LLM verifies relations and identifies same concepts",
        "🔀 **LLM-based Merge** - Merge concepts identified as identical by LLM",
        "📊 **Build Graph** - Create NetworkX knowledge graph",
        "🔄 **DAG Conversion** - LLM removes cycles to create DAG",
        "✂️ **Transitive Reduction** - Remove redundant edges"
    ]

    for step in steps:
        st.markdown(f"- {step}")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
        <small>Built with ❤️ using Streamlit | LangChain | Gemini | NetworkX</small>
    </div>
    """,
    unsafe_allow_html=True
)
