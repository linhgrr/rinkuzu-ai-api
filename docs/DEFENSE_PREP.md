# RinKuzu — Audit nghiệp vụ + Bộ phòng thủ phản biện DATN

> Tài liệu chuẩn bị phản biện. Mọi phát hiện đã đọc code xác minh, kèm `file:line`.

## Context
- Phản biện: **9h thứ 4, 08/07, phòng 602 B1** — demo sản phẩm + giải trình **nghiệp vụ / sản phẩm / công nghệ–thuật toán**.
- 3 finding "CRITICAL" của quá trình quét tự động đã bị **bác bỏ** sau khi đọc code (mục 1.4).
- Scope: `rinkuzu/` (Next.js FE + Mongo) + `rinkuzu-ai-api/` (FastAPI). Bỏ `shin-rinkuzu/` (starter trống).

---

# PHẦN 1 — AUDIT NGHIỆP VỤ

## 1.1 Kiến trúc
2 domain, 2 MongoDB, nối qua proxy `/api/adaptive/*`. FE (Next.js): tài khoản/quiz/SRS/payment/admin/gamification. BE (FastAPI): content pipeline (PDF→knowledge graph), adaptive (SAINT + D3QN), quiz-extract, tutor. Quiz "hybrid": BE tạo draft (LLM) → **FE validate & tạo Quiz thật** → BE `mark_submitted` dọn PDF. 3 model train sẵn (`config.py:34-36`): `saint_best.pt` (KT), `dqn_best.pt` (RL), `prereq_vimath_bgem3_namedef_concat_rich.pth` (phân loại prerequisite train trên **ViMath**).

## 1.2 ĐIỂM MẠNH (chủ động khoe)
- **Adaptive là KT thực thụ**: SAINT transformer encoder–decoder, output `torch.sigmoid` ⇒ mastery ∈ (0,1) (`models.py:108-208`).
- **Chọn bài D3QN concept-agnostic** (Dueling + Double DQN) + **action masking** cứng theo prerequisite + bloom-sequential (`environment.py:476-511`, `models.py:211-264`). Một policy dùng cho **mọi số concept / mọi môn**.
- **Prereq graph nhiều nguồn + kiểm chứng**: model PRS (ViMath, BGE-M3 name+def) + candidate từ concept-extraction → merge → verification (`min_confidence`) → **khử chu trình→DAG** → **quality gate**; graph hỏng ⇒ job `FAILED` message rõ, không tạo session lỗi (`relation_engine.py`, `graph_optimization.py`, `finalization.py:140-152`).
- **Auth chắc**: bắt buộc verify email (`lib/auth.ts:209-219`), lockout 5 lần/30′, revoke session khi đổi mật khẩu (`:372-374`), Argon2id.
- **PayOS chắc**: HMAC-SHA256 timing-safe, idempotent theo `transactionReference`, kích hoạt atomic (transaction), entitlement check hết hạn realtime (`SubscriptionService.ts:159`).
- **Authorization nhất quán**: ownership check quiz/attempt/folder; admin gate API; validate quiz bằng Zod mọi đường; chấm điểm server-side (`QuizPlayService.ts:238-257`). **SRS SM-2 chuẩn**.

## 1.3 PHÁT HIỆN (đã xác minh)
| Mức | Vấn đề | Bằng chứng |
|---|---|---|
| 🔴 **H1** | **Adaptive KHÔNG có điều kiện HOÀN THÀNH theo mục tiêu** — chỉ dừng khi `step>=max_steps`, default **9999**; không có "master hết ⇒ xong". | `environment.py:422,51`; `exercise_service.py:225-229`; `session.py:435` |
| 🟠 M1 | Graph phụ thuộc 2 giả định chưa kiểm chứng ngoài miền: concept **do LLM trích** (sót/ảo); PRS **train trên toán ViMath** → tổng quát sang môn khác chưa đánh giá | `relation_engine.py`, `config.py:36` |
| 🟠 M2 | Unlock **chỉ dựa Bloom-3 (Apply)** của tiên quyết — cần giải thích sư phạm | `environment.py:182,243-244` |
| 🟠 M3 | Chấm tự luận bằng LLM **không fallback khi timeout** ⇒ hỏng vòng lặp cốt lõi | `exercise_service.py:424-435` |
| 🟠 M4 | Attempt **không idempotent** ⇒ double-submit thổi phồng số liệu/SRS | `QuizPlayService.ts:270` |
| 🟠 M5 | Gói **free không giới hạn số quiz** (chỉ private=premium) | `QuizCrudService.ts:80-91` |
| 🟡 L1 | `isUserPremium` nhánh `endDate` rỗng ⇒ premium vĩnh viễn | `SubscriptionService.ts:159` |
| 🟡 L2-L7 | admin FE không guard role; admin cấp sub không audit log; race reap↔cancel (single-instance); multi-choice all-or-nothing; dead code `environment.py:432`; draft hết hạn nghi không dọn | — |

## 1.4 Đừng để bị "úp" (đã kiểm chứng KHÔNG phải lỗi)
- ❌ "Mastery không bound [0,1]" — SAI, `torch.sigmoid` (`models.py:164,202`).
- ❌ "Publish quiz không validate" — SAI, `questionListSchema` bắt ≥1 câu, 2-20 option không rỗng, `correctIndex`/`correctIndexes` trong range/không trùng; cả create & update.
- ❌ "Chấm điểm tin client" — SAI, đáp án lấy từ DB, so server-side.
- ❌ "Hết hạn vẫn dùng premium" — SAI, check `!isSubscriptionExpired` realtime.
- ❌ "SM-2 gian lận được" / "prereq kẹt vì thiếu Bloom-3" — thổi phồng / không xảy ra (mọi concept đủ 6 Bloom).

---

# PHẦN 2 — BỘ PHÒNG THỦ Q&A

## A. CÔNG NGHỆ / THUẬT TOÁN (khó nhất)
**A1 — SAINT là gì, sao dùng?** KT transformer encoder–decoder: encoder = chuỗi bài (concept + Bloom embedding), decoder = chuỗi đúng/sai; self-attention bắt phụ thuộc dài hạn tốt hơn RNN/DKT, hơn BKT (1 tham số/kỹ năng). Mở rộng embedding Bloom ⇒ dự đoán mastery theo từng (concept, Bloom), output sigmoid = P(đúng) (`models.py:108-208`).

**A2 — D3QN train sao? "Mô phỏng thì tin được không?" (bẫy):** train trong env Gymnasium, **"học sinh ảo" chính là SAINT** (`environment.py:408-413`: đúng/sai ~ Binomial(p từ SAINT)). State=[hidden SAINT|tiến độ|coverage|per-concept mastery 6 Bloom/visited/prereq_ok]; action=(concept,Bloom); reward = khám phá mới + tăng mastery (decay khi lặp) + coverage & learning-gain cuối (`environment.py:424-448`). Dueling (tách V/A) + Double (giảm overestimate) + masking (ép ràng buộc). **Thừa nhận:** phụ thuộc độ trung thực simulator; sẽ tinh chỉnh khi có dữ liệu thật.

**A3 — "Làm sao biết A tiên quyết B?" (M1):** KHÔNG phải LLM đoán tự do: (1) model phân loại prereq **train trên ViMath** (BGE-M3 tên+định nghĩa, ngưỡng `prs_threshold` hiệu chỉnh) chấm từng cặp; (2) + quan hệ từ bước trích concept; (3) verify theo confidence; (4) khử chu trình→DAG; (5) quality gate. **Thừa nhận:** PRS train toán → môn khác là hướng phát triển; concept LLM trích cần rà.

**A4 — Pipeline:** 14 trạng thái LOADING→…→OPTIMIZING→quality gate→COMPLETED (`domain/jobs.py:41-57`); OCR LandingAI; embedding `vietnamese-sbert`; reaper/recovery/retry-from-S3/cancel.

**A5 — "Sinh/chấm bài LLM đáng tin?":** MCQ/fill/T-F **chấm bằng luật xác định**; chỉ tự luận dùng LLM; payload validate Pydantic. (Yếu: M3 chưa fallback.)

## B. NGHIỆP VỤ
- **B-H1 "Khi nào hoàn thành?"** → nếu vá: "mọi concept mastery ≥ 0.75, dashboard hiện đã master k/N"; chưa vá: framing "luyện tập liên tục theo mastery, hiển thị tiến độ tới khi master hết". Nên vá để mạnh hơn.
- **B-M2 "Sao chỉ Apply?"** → "Apply = hiểu đủ để vận dụng & học tiếp, chuẩn sư phạm phổ biến; Bloom cao hơn vẫn luyện sau."
- **B-M4/M5/L*** → thừa nhận là hạng mục hoàn thiện; nêu hướng sửa ngắn.

## C. SẢN PHẨM
- Vấn đề: có PDF nhưng thiếu lộ trình cá nhân hoá theo mức thành thạo.
- 2 mode: (A) PDF→Quiz; (B) PDF→Lộ trình thích ứng (concept + knowledge graph + chuỗi bài master từng concept). Khác biệt = **adaptive theo knowledge graph tự sinh từ tài liệu của user**.
- Kiếm tiền: premium (quiz private) qua PayOS + gamification (streak/freeze, XP). So Quizlet/Anki: SRS + adaptive theo graph tự sinh.

## D. CÂU "TỬ HUYỆT" — PHẢI CHUẨN BỊ SỐ LIỆU THẬT ⚠️
Không tìm thấy dữ liệu/notebook train trong repo serving (chỉ có checkpoint). **Bắt buộc tự biết & nói chính xác:**
1. **SAINT train trên dữ liệu tương tác nào?** (EdNet/ASSISTments công khai? synthetic? ViMath?) — nếu synthetic, sinh bằng cơ chế gì, vì sao hợp lý.
2. **DQN train bao nhiêu bước, reward hội tụ ra sao** (`mean_reward` lưu trong ckpt, `models.py:313`).
3. **Đánh giá offline adaptive vs baseline (random/tuần tự)?** Nếu chưa → nói "hướng đánh giá tiếp theo", không khẳng định sai.

---

# PHẦN 3 — KỊCH BẢN DEMO AN TOÀN
1. **Không demo live pipeline** (chạy vài phút, có thể bị quality-gate loại). Chuẩn bị sẵn 1 subject đã xử lý + 1 quiz đã publish.
2. Luồng: Login → Mode A (làm quiz → màn chấm điểm) → Mode B (mở subject → **hiện knowledge graph + prereq** → làm 3-4 bài → **mastery dashboard cập nhật + concept mở khoá dần**) → SRS/streak.
3. **Tránh mìn:** mạng ổn (LLM realtime); ưu tiên MCQ, tránh tự luận nếu M3 chưa vá; **giữ đĩa trống** (đừng để `/tmp` đầy lại); quay sẵn video backup luồng adaptive.

---

# PHẦN 4 — VÁ NHANH TRƯỚC DEMO (tùy chọn)
1. **H1** — điều kiện hoàn thành theo mastery (diff nhỏ `environment.py:step`) → trả lời "khi nào hoàn thành" cực mạnh.
2. **M3** — fallback chấm tự luận (retry/heuristic/"chưa chấm" thay vì fail) → chống hỏng demo.
3. **M4** — idempotency attempt (ít gấp).

---

# PHẦN 5 — VIỆC EM TỰ LÀM
- ⚠️ Giữ đĩa trống (đã dọn — theo dõi thêm).
- ⚠️ Nắm chính xác nguồn dữ liệu train SAINT/DQN (Phần D) — tử huyệt.
- Quyết định có vá H1/M3 trước demo không.

## Verification
Mỗi finding có `file:line` để đối chiếu. Chứng minh H1: chạy 1 session tới khi `avg_mastery` cao mà session vẫn `active`. Checkpoint: `ls rinkuzu-ai-api/models/`.
