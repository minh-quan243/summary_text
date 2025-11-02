# app.py
import streamlit as st
import torch
from transformers import LEDTokenizer, LEDForConditionalGeneration
import pandas as pd
import json
import time
from rouge_score import rouge_scorer
import nltk
import plotly.graph_objects as go
import plotly.express as px

# Setup page
st.set_page_config(
    page_title="Scientific Paper Summarizer",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Download nltk data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')


@st.cache_resource
def load_model():
    """Load model với cache để tăng tốc độ"""
    try:
        tokenizer = LEDTokenizer.from_pretrained("D:\Quân\project\summary_text\checkpoints\led-scientific")
        model = LEDForConditionalGeneration.from_pretrained("D:\Quân\project\summary_text\checkpoints\led-scientific")

        # Move to GPU nếu có
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        return tokenizer, model, device
    except Exception as e:
        st.error(f"❌ Lỗi khi load model: {e}")
        return None, None, None


def summarize_text(model, tokenizer, device, text, max_length=512, num_beams=4):
    """Tóm tắt văn bản"""
    try:
        # Tokenize input
        inputs = tokenizer(
            text,
            return_tensors="pt",
            max_length=8192,
            truncation=True
        ).to(device)

        # Generate summary
        with torch.no_grad():
            summary_ids = model.generate(
                inputs["input_ids"],
                max_length=max_length,
                min_length=150,
                num_beams=num_beams,
                length_penalty=2.0,
                early_stopping=True,
                no_repeat_ngram_size=3
            )

        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary
    except Exception as e:
        st.error(f"❌ Lỗi khi tóm tắt: {e}")
        return None


def calculate_rouge(original_summary, generated_summary):
    """Tính ROUGE scores"""
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(original_summary, generated_summary)

    return {
        'rouge1': scores['rouge1'].fmeasure,
        'rouge2': scores['rouge2'].fmeasure,
        'rougeL': scores['rougeL'].fmeasure
    }


def plot_rouge_scores(scores):
    """Vẽ biểu đồ ROUGE scores"""
    fig = go.Figure(data=[
        go.Bar(name='ROUGE Scores',
               x=list(scores.keys()),
               y=[scores[k] for k in scores],
               marker_color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
    ])

    fig.update_layout(
        title="ROUGE Scores Evaluation",
        xaxis_title="Metrics",
        yaxis_title="Score",
        yaxis=dict(range=[0, 1]),
        template="plotly_white"
    )

    return fig


def main():
    # Header
    st.title("📚 Scientific Paper Summarizer")
    st.markdown("""
    Sử dụng LED model được fine-tuned trên PubMed dataset để tóm tắt paper khoa học.
    Model hỗ trợ documents dài đến 16,384 tokens.
    """)

    # Sidebar
    st.sidebar.title("⚙️ Cài đặt")

    # Load model
    with st.sidebar:
        st.subheader("Model Information")
        with st.spinner("Đang load model..."):
            tokenizer, model, device = load_model()

        if model is not None:
            st.success(f"✅ Model loaded on: {device}")
            st.info(f"Model: LED-Scientific")
        else:
            st.error("❌ Không thể load model")
            return

    # Main content
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📝 Nhập văn bản")

        # Input options
        input_method = st.radio(
            "Chọn cách nhập:",
            ["Nhập trực tiếp", "Upload file", "Demo với sample text"]
        )

        input_text = ""

        if input_method == "Nhập trực tiếp":
            input_text = st.text_area(
                "Dán nội dung paper khoa học:",
                height=300,
                placeholder="Dán toàn văn paper khoa học vào đây..."
            )

        elif input_method == "Upload file":
            uploaded_file = st.file_uploader(
                "Upload file text hoặc PDF",
                type=['txt', 'pdf'],
                help="Hỗ trợ file .txt và .pdf"
            )

            if uploaded_file is not None:
                if uploaded_file.type == "text/plain":
                    input_text = str(uploaded_file.read(), "utf-8")
                else:
                    st.warning("⚠️ PDF support coming soon. Please use text file.")

        else:  # Demo với sample text
            sample_text = """
            Deep learning has revolutionized many fields of artificial intelligence, including 
            natural language processing, computer vision, and speech recognition. 
            This paper presents a comprehensive survey of recent advances in deep learning 
            architectures and their applications. We discuss convolutional neural networks, 
            recurrent neural networks, and transformer architectures, highlighting their 
            strengths and limitations. Our analysis shows that transformer-based models 
            have achieved state-of-the-art performance in many NLP tasks, but require 
            significant computational resources. We also explore emerging trends such as 
            efficient transformers and multimodal learning. The paper concludes with 
            discussions on challenges and future research directions in deep learning.

            In recent years, attention mechanisms have become a crucial component in 
            many deep learning models. The self-attention mechanism, in particular, 
            allows models to weigh the importance of different parts of the input sequence. 
            This has led to significant improvements in machine translation, text summarization, 
            and other sequence-to-sequence tasks. However, the quadratic complexity of 
            self-attention with respect to sequence length remains a major limitation 
            for processing long documents.
            """
            input_text = sample_text
            st.text_area("Sample text:", value=sample_text, height=300)

    with col2:
        st.subheader("🎛️ Tùy chỉnh parameters")

        col2a, col2b = st.columns(2)

        with col2a:
            max_length = st.slider(
                "Độ dài tóm tắt (tokens):",
                min_value=100,
                max_value=800,
                value=512,
                help="Độ dài tối đa của bản tóm tắt"
            )

            min_length = st.slider(
                "Độ dài tối thiểu:",
                min_value=50,
                max_value=300,
                value=150,
                help="Độ dài tối thiểu của bản tóm tắt"
            )

        with col2b:
            num_beams = st.selectbox(
                "Number of beams:",
                options=[1, 2, 4, 6, 8],
                index=2,
                help="Số beams cho beam search (cao hơn = chất lượng tốt hơn nhưng chậm hơn)"
            )

            length_penalty = st.slider(
                "Length penalty:",
                min_value=0.1,
                max_value=3.0,
                value=2.0,
                help="Penalty cho độ dài (cao hơn = khuyến khích summary dài hơn)"
            )

    # Summary section
    st.markdown("---")
    st.subheader("📄 Kết quả tóm tắt")

    if input_text and model is not None:
        # Calculate text stats
        word_count = len(input_text.split())
        char_count = len(input_text)

        col_stats1, col_stats2, col_stats3 = st.columns(3)

        with col_stats1:
            st.metric("Số từ input", f"{word_count:,}")
        with col_stats2:
            st.metric("Số ký tự", f"{char_count:,}")
        with col_stats3:
            if word_count > 16384:
                st.warning("⚠️ Văn bản quá dài, sẽ bị cắt bớt")
            else:
                st.success("✅ Độ dài phù hợp")

        # Generate summary
        if st.button("🚀 Tóm tắt", type="primary", use_container_width=True):
            with st.spinner("Đang tóm tắt... (có thể mất vài phút cho văn bản dài)"):
                start_time = time.time()

                summary = summarize_text(
                    model, tokenizer, device, input_text,
                    max_length=max_length,
                    num_beams=num_beams
                )

                end_time = time.time()
                processing_time = end_time - start_time

            if summary:
                # Display results
                col_result1, col_result2 = st.columns([2, 1])

                with col_result1:
                    st.success("✅ Tóm tắt thành công!")
                    st.text_area(
                        "Bản tóm tắt:",
                        value=summary,
                        height=200,
                        key="summary_output"
                    )

                    # Summary stats
                    summary_words = len(summary.split())
                    compression_ratio = (1 - summary_words / word_count) * 100

                    col_sum1, col_sum2, col_sum3 = st.columns(3)
                    with col_sum1:
                        st.metric("Số từ summary", summary_words)
                    with col_sum2:
                        st.metric("Thời gian xử lý", f"{processing_time:.2f}s")
                    with col_sum3:
                        st.metric("Tỉ lệ nén", f"{compression_ratio:.1f}%")

                with col_result2:
                    # ROUGE evaluation (nếu có reference)
                    st.subheader("📊 Đánh giá")

                    reference_summary = st.text_area(
                        "Reference summary (tùy chọn):",
                        height=100,
                        placeholder="Dán reference summary để đánh giá ROUGE scores..."
                    )

                    if reference_summary:
                        rouge_scores = calculate_rouge(reference_summary, summary)

                        # Display scores
                        for metric, score in rouge_scores.items():
                            st.metric(metric.upper(), f"{score:.4f}")

                        # Plot
                        fig = plot_rouge_scores(rouge_scores)
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("ℹ️ Thêm reference summary để xem ROUGE scores")

                # Export options
                st.markdown("---")
                st.subheader("📤 Export kết quả")

                col_export1, col_export2, col_export3 = st.columns(3)

                with col_export1:
                    # Download summary as text
                    st.download_button(
                        label="📥 Tải summary",
                        data=summary,
                        file_name="paper_summary.txt",
                        mime="text/plain"
                    )

                with col_export2:
                    # Download as JSON
                    result_data = {
                        "original_length": word_count,
                        "summary_length": len(summary.split()),
                        "processing_time": processing_time,
                        "summary": summary,
                        "parameters": {
                            "max_length": max_length,
                            "num_beams": num_beams,
                            "length_penalty": length_penalty
                        }
                    }

                    if reference_summary:
                        result_data["rouge_scores"] = rouge_scores

                    st.download_button(
                        label="📊 Tải JSON",
                        data=json.dumps(result_data, indent=2),
                        file_name="summary_results.json",
                        mime="application/json"
                    )

                with col_export3:
                    # Copy to clipboard
                    if st.button("📋 Copy summary"):
                        st.code(summary)
                        st.success("✅ Đã copy vào clipboard!")

    else:
        if not input_text:
            st.info("👈 Vui lòng nhập văn bản cần tóm tắt")
        else:
            st.error("❌ Model chưa sẵn sàng")

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center'>
        <p>🔬 Scientific Paper Summarizer | Built with LED Model & Streamlit</p>
    </div>
    """, unsafe_allow_html=True)


# Batch processing tab
def batch_processing():
    st.header("🔄 Batch Processing")

    st.info("""
    Tính năng này cho phép xử lý hàng loạt nhiều papers cùng lúc.
    Upload file CSV với cột chứa văn bản cần tóm tắt.
    """)

    uploaded_csv = st.file_uploader("Upload CSV file", type=['csv'])

    if uploaded_csv is not None:
        df = pd.read_csv(uploaded_csv)
        st.write("Preview data:", df.head())

        text_column = st.selectbox("Chọn cột chứa văn bản:", df.columns)

        if st.button("Process Batch"):
            progress_bar = st.progress(0)
            status_text = st.empty()

            tokenizer, model, device = load_model()
            summaries = []

            for i, text in enumerate(df[text_column]):
                status_text.text(f"Processing {i + 1}/{len(df)}...")

                if pd.notna(text) and str(text).strip():
                    summary = summarize_text(model, tokenizer, device, str(text))
                    summaries.append(summary)
                else:
                    summaries.append("")

                progress_bar.progress((i + 1) / len(df))

            # Add summaries to dataframe
            df['summary'] = summaries

            # Download results
            st.success(f"✅ Đã xử lý xong {len(df)} documents")
            st.download_button(
                label="📥 Tải kết quả CSV",
                data=df.to_csv(index=False),
                file_name="batch_summaries.csv",
                mime="text/csv"
            )


# Model information tab
def model_info():
    st.header("ℹ️ Thông tin Model")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("LED Model (Longformer)")
        st.markdown("""
        **LED (Longformer-Encoder-Decoder)** là model transformer được thiết kế đặc biệt cho:

        - 📚 **Xử lý documents dài**: Hỗ trợ đến 16,384 tokens
        - 🔬 **Scientific text**: Được pre-train trên arXiv papers
        - 🎯 **Abstractive summarization**: Tạo bản tóm tắt mới

        **Kiến trúc:**
        - Encoder: Longformer với attention hiệu quả
        - Decoder: Transformer decoder thông thường
        - Parameters: ~400M parameters

        **Training data:**
        - PubMed dataset (khoa học y tế)
        - arXiv papers (vật lý, toán, CS)
        """)

    with col2:
        st.subheader("Hiệu năng")
        st.metric("Max input length", "16,384 tokens")
        st.metric("Max output length", "512 tokens")
        st.metric("Model size", "~1.5GB")
        st.metric("Supported tasks", "Summarization")

        st.subheader("Requirements")
        st.code("""
        - GPU: >= 8GB VRAM
        - RAM: >= 16GB  
        - Storage: >= 2GB
        """)


# Create tabs
tab1, tab2, tab3 = st.tabs(["📝 Single Document", "🔄 Batch Processing", "ℹ️ Model Info"])

with tab1:
    main()

with tab2:
    batch_processing()

with tab3:
    model_info()