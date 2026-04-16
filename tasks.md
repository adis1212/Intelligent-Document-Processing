# 📋 IDP Manuscript Processing — Task List

## ✅ Completed Tasks

### Phase 1: Core Pipeline
- [x] Module 1 — Preprocessing (Grayscale, Resize, NLM Denoise, CLAHE)
- [x] Module 2 — Quality Check (Blur, Skew PPM, Crop, Occlusion)
- [x] Module 3 — Sequence Detection (Regex, Gap Analysis, dHash Duplicates)
- [x] Module 4 — OCR Engine (Text Extraction, Confidence, Keywords, Script Detection)
- [x] Module 5 — Aggregation (Weighted Scoring, Search Index)
- [x] Module 6 — Output (CSV + JSON)

### Phase 2: Dashboard Improvements
- [x] Session state persistence (no data loss on interaction)
- [x] File upload — users can browse & upload their own images
- [x] Skew detection overrides status (capped at REVIEW)
- [x] Clean white/black theme with colorful metric numbers
- [x] Color-coded summary cards (Blue/Green/Orange/Red/Purple)
- [x] Document search with snippets
- [x] Download buttons (CSV + JSON)
- [x] Pipeline progress sidebar (6 steps)
- [x] Removed all developer-facing text from UI

---

## 🔜 Future Tasks

### 🟡 Priority: High

#### Task 1: Multilingual OCR Support (Marathi & Hindi)
- [ ] Add Marathi heritage text corpus (15+ passages)
- [ ] Add Hindi heritage text corpus (15+ passages)
- [ ] Implement script detection heuristic to auto-select language
- [ ] Update keyword extraction for Devanagari stop words
- [ ] Add language selector in sidebar
- [ ] Test with real Marathi/Hindi manuscript scans

#### Task 2: Production OCR Integration
- [ ] Replace mock OCR with Tesseract integration (install pytesseract)
- [ ] OR integrate cloud-based OCR API (Google Vision / Azure AI)
- [ ] Add OCR model selection dropdown in dashboard
- [ ] Benchmark accuracy against mock results

#### Task 3: Database Integration
- [ ] Design schema for MySQL/PostgreSQL (manuscripts, pages, results)
- [ ] Store pipeline results in database instead of JSON files
- [ ] Add historical run comparison (track improvements over time)
- [ ] Add user authentication for multi-user access

### 🟢 Priority: Medium

#### Task 4: Advanced Search
- [ ] Implement fuzzy matching (Levenshtein distance)
- [ ] Add semantic search using sentence embeddings
- [ ] Add filters: status, quality score range, date range
- [ ] Add search result pagination

#### Task 5: Manual Override / Review Workflow
- [ ] Add "Accept" / "Reject" buttons for REVIEW pages
- [ ] Track manual overrides in audit log
- [ ] Add reviewer comments/notes field
- [ ] Implement role-based access (admin vs reviewer)

#### Task 6: Batch Processing & Performance
- [ ] Add progress bar per-image (not just per-step)
- [ ] Implement parallel processing for large batches (>100 pages)
- [ ] Add image caching to avoid re-processing unchanged images
- [ ] Optimize memory usage for high-resolution scans

### 🔵 Priority: Low

#### Task 7: Advanced Quality Metrics
- [ ] Add DPI detection and validation
- [ ] Add color space analysis (detect incorrect scanning mode)
- [ ] Add page orientation detection (portrait vs landscape)
- [ ] Implement text region detection to skip blank pages

#### Task 8: Export & Integration
- [ ] Export to PDF with embedded OCR text (searchable PDF)
- [ ] Export to IIIF manifest for digital library integration
- [ ] Add API endpoints (REST) for programmatic access
- [ ] Webhook notifications on pipeline completion

#### Task 9: UI/UX Enhancements
- [ ] Add dark mode toggle
- [ ] Add side-by-side comparison view (before/after)
- [ ] Add zoom/pan on manuscript images
- [ ] Add annotation tools (mark regions of interest)
- [ ] Mobile-responsive layout

#### Task 10: Testing & CI/CD
- [ ] Add comprehensive unit tests for all modules (pytest)
- [ ] Add integration tests for full pipeline
- [ ] Set up GitHub Actions CI/CD pipeline
- [ ] Add code coverage reporting
- [ ] Add linting (flake8/black)

---

## 📊 Architecture Decisions

| Decision | Rationale |
|----------|-----------|
| Session state for results | Prevents data loss on Streamlit rerun |
| File upload → disk save | Enables pipeline to read from consistent directory |
| Mock OCR fallback | Works without Tesseract installation |
| Inverted index for search | Fast keyword lookup without external dependencies |
| Weighted scoring (40/40/20) | Balances quality and OCR with sequence completeness |
| Skew override to REVIEW | Prevents skewed pages from passing silently |
| White + black theme | Maximum readability for all users |

---

*Last updated: 2026-04-07*