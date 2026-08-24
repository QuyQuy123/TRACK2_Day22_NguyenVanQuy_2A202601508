# Báo cáo thực nghiệm và phân tích kết quả — Day 22

Dự án: **Day 22: LangSmith + Prompt Versioning + RAGAS Evaluation + Guardrails AI**ọc viên: **Nguyễn Văn Quý**

---

## 1. Danh mục các tệp bằng chứng (Evidence Deliverables)

| Tên tệp | Mô tả | Trạng thái |
|---|---|:---:
|
`010_langsmith_traces.png` | Ảnh chụp màn hình LangSmith dashboard hiển thị ≥ 50 traces truy vấn RAG | ✅ Đầy đủ |
| `02_prompt_hub.png` | ÆĞ�chụp màn hình 2 prompt templates trên LangSmith Prompt Hub |
✅ Đầy đủ |
| `02_ab_routing_log.txt` | File log phân phối A/B routing tất định (50 câu hỏi gắn nhãn `v1`/`v2`) | ✅ Đầy đủ |
| `03_ragas_scores.png` | ÆĞ�chụp bảng điểm so sánh 4 chỉ số REGAS trên Terminal | ✅ Đầy đủ |
| `03_ragas_report.json` | File JSON xuất kết quả đánh giá RAGAS cho 2 phiên bản prompt | ✅ Đầy đủ |
| `04_pii_demo_log.txt` | Log kiểm thử PII Detector (che Email, Phone, SSN, Credit Card) | ✅ Đầy đủ |
| `04_json_demo_log.txt` | Log kiểm thử JSON Formatter (sữa fences, nháy đơn, trailing commas) | ✅ Đầy đủ |

---

## 2. Phân tích kết quả A/B Testing & Đánh giá RAGAS (V1 vs V2)

Cả 2 phiên bản đều đạt chỉ số faithfulness cao (≥ 0.85 - 1.0) và trả lời chính xác theo context.
