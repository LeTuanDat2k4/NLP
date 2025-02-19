import os

os.makedirs("./documents", exist_ok=True)
from typing import List, Dict, Any, Union, Optional, Tuple, Callable
from dataclasses import dataclass
from pathlib import Path
import logging
import json
import torch
from threading import Thread
import numpy as np
import re
import functools
from sentence_transformers import SentenceTransformer, util
from rank_bm25 import BM25Okapi
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
import fitz  # PyMuPDF
import pdfplumber

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ThongTinTrichXuat:
    ten_file: str
    dinh_dang: str
    cac_ben_lien_quan: List[Dict[str, Any]]
    ngay_thang: List[Dict[str, str]]
    dieu_khoan: List[Dict[str, str]]
    nghia_vu: List[Dict[str, str]]
    phan_loai: str
    van_de_tiem_tang: List[Dict[str, str]]
    relevant_sections: List[Dict[str, str]] = None
    risk_score: Dict[str, Any] = None
    confidence_score: float = 0.0

    def __post_init__(self):
        if self.relevant_sections is None:
            self.relevant_sections = []
        if self.risk_score is None:
            self.risk_score = {"overall_score": 0.0, "category_breakdown": {}}


class RAGProcessor:
    def __init__(self, embedding_model: str = "nlpaueb/legal-bert-base-uncased"):
        try:
            self.embedding_model = SentenceTransformer(embedding_model)
        except Exception as e:
            logger.warning(f"Failed to load {embedding_model}: {e}. Falling back to general model.")
            self.embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

        self.document_store = {}
        self.bm25 = None
        self.document_embeddings = None

    def process_document(self, text: str) -> List[str]:
        chunks = self._split_into_chunks(text)
        if not chunks:
            logger.error("No valid chunks extracted from document!")
            return []
        self.document_store = {i: chunk for i, chunk in enumerate(chunks)}
        self.bm25 = BM25Okapi([chunk.split() for chunk in chunks]) if chunks else None
        self.document_embeddings = self.embedding_model.encode(list(self.document_store.values()))
        return chunks

    def retrieve_relevant_sections(self, query: str, k: int = 5) -> List[Dict[str, str]]:
        if not self.document_store:
            logger.error("Document store is empty!")
            return []
        if self.bm25 is None:
            logger.error("BM25 model is not initialized!")
            return []

        dense_scores = self._get_dense_scores(query)
        sparse_scores = self._get_sparse_scores(query) if self.bm25 else np.zeros(len(dense_scores))
        combined_scores = 0.7 * dense_scores + 0.3 * sparse_scores
        top_indices = np.argsort(combined_scores)[-k:][::-1]  # Reversed to get highest scores first
        top_indices = [idx for idx in top_indices if idx < len(self.document_store)]

        return [
            {
                "content": self.document_store[idx],
                "score": float(combined_scores[idx]),
                "index": int(idx)
            }
            for idx in top_indices if idx < len(combined_scores)
        ]

    @functools.lru_cache(maxsize=100)
    def _encode_query(self, query: str) -> np.ndarray:
        return self.embedding_model.encode([query])[0]

    def _get_dense_scores(self, query: str) -> np.ndarray:
        query_embedding = self._encode_query(query)
        if self.document_embeddings is None:
            logger.warning("Document embeddings not precomputed, computing now...")
            doc_embeddings = self.embedding_model.encode(list(self.document_store.values()))
        else:
            doc_embeddings = self.document_embeddings
        return util.dot_score(doc_embeddings, query_embedding).numpy().flatten()

    def _get_sparse_scores(self, query: str) -> np.ndarray:
        return np.array(self.bm25.get_scores(query.split())) if self.bm25 else np.zeros(len(self.document_store))

    def _split_into_chunks(self, text: str) -> List[str]:
        # Try section-based splitting first
        sections = re.split(r'\n\s*(?:SECTION|ARTICLE|CLAUSE)\s+\d+\.', text)
        if len(sections) > 1:
            # Clean up sections and remove empty ones
            sections = [s.strip() for s in sections if s.strip()]
            return sections

        # Fall back to paragraph-based splitting
        paragraphs = re.split(r'\n{2,}', text)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]

        # For very short paragraphs, merge them
        return self._merge_short_paragraphs(paragraphs)

    def _merge_short_paragraphs(self, paragraphs: List[str], min_length: int = 100) -> List[str]:
        result = []
        current_chunk = ""

        for p in paragraphs:
            if len(current_chunk) + len(p) < min_length:
                current_chunk += " " + p if current_chunk else p
            else:
                if current_chunk:
                    result.append(current_chunk)
                current_chunk = p

        if current_chunk:
            result.append(current_chunk)

        return result


class XuLyTaiLieuPhapLy:
    def __init__(self, model_path: str = "deepseek-ai/deepseek-coder-6.7b-base"):
        logger.info("Initializing document processing system...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True
            )
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.padding_side = "left"

            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
            self.streamer = TextIteratorStreamer(self.tokenizer)
            self.rag_processor = RAGProcessor()
            self.precedent_library = {}
            logger.info("System initialization complete")
        except Exception as e:
            logger.error(f"Error initializing model: {str(e)}")
            raise

    def goi_model(self, prompt: str, max_tokens: int = 1024) -> str:
        try:
            logger.info(f"Sending prompt to model ({len(prompt)} characters)...")

            encoded = self.tokenizer.encode_plus(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=4096  # Increased context length
            )

            input_ids = encoded["input_ids"].to(self.model.device)
            attention_mask = encoded["attention_mask"].to(self.model.device)

            generation_kwargs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "max_new_tokens": max_tokens,
                "do_sample": True,
                "temperature": 0.5,  # Lowered for more deterministic outputs
                "top_p": 0.95,  # Added nucleus sampling
                "pad_token_id": self.tokenizer.eos_token_id,
                "streamer": self.streamer
            }

            thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
            thread.start()

            response_text = ""
            for text in self.streamer:
                response_text += text

            if not response_text.strip():
                logger.error("Model returned empty response")
                return ""

            return response_text.strip()

        except Exception as e:
            logger.error(f"Error in model generation: {str(e)}")
            return ""

    def doc_file(self, duong_dan_file: str) -> tuple[str, str]:
        duong_dan = Path(duong_dan_file)
        dinh_dang = duong_dan.suffix.lower()

        try:
            if dinh_dang == ".pdf":
                text = self.extract_text_from_pdf(str(duong_dan))
            elif dinh_dang == ".docx":
                text = self.extract_text_from_docx(str(duong_dan))
            else:
                with open(duong_dan, 'r', encoding='utf-8', errors='ignore') as file:
                    text = file.read()
            return text, dinh_dang
        except Exception as e:
            logger.error(f"Error reading file {duong_dan}: {str(e)}")
            raise

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from a PDF file using PyMuPDF or pdfplumber with fallback"""
        text = []
        try:
            with fitz.open(pdf_path) as doc:
                for page in doc:
                    text.append(page.get_text("text"))
        except Exception as e:
            logger.warning(f"PyMuPDF failed, trying pdfplumber: {e}")
            try:
                with pdfplumber.open(pdf_path) as pdf:
                    text = [page.extract_text() for page in pdf.pages if page.extract_text()]
            except Exception as e2:
                logger.error(f"Both PDF extractors failed: {e2}")
                raise

        return "\n".join(text).strip()

    def extract_legal_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract legal entities using regex patterns"""
        patterns = {
            "parties": r"(?:between|among)\s+([^,\.]+)(?:,|\sand)\s+([^,\.]+)",
            "dates": r"(?:dated|effective|as of|on)\s+(\d{1,2}(?:st|nd|rd|th)?\s+\w+\s+\d{4})",
            "monetary": r"(\$\d{1,3}(?:,\d{3})*(?:\.\d{2})?)",
            "durations": r"(\d+)\s+(day|week|month|year)s?",
            "jurisdictions": r"(?:governed by|laws of|jurisdiction of)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)"
        }

        entities = {}
        for entity_type, pattern in patterns.items():
            matches = re.findall(pattern, text)
            if entity_type == "parties":
                # Handle multi-group matches
                parties = []
                for match in matches:
                    parties.extend([party.strip() for party in match if party.strip()])
                entities[entity_type] = parties
            else:
                entities[entity_type] = [match if isinstance(match, str) else match[0] for match in matches]

        return entities

    def trich_xuat_thong_tin(self, text: str) -> Dict[str, Any]:
        # Process document with RAG
        chunks = self.rag_processor.process_document(text)
        logger.info(f"Document split into {len(chunks)} chunks")

        # Extract preliminary entities
        entities = self.extract_legal_entities(text)
        logger.info(
            f"Extracted entities: {len(entities.get('parties', []))} parties, {len(entities.get('dates', []))} dates")

        # Get relevant sections for different aspects
        query_categories = {
            "parties": "identify all parties and their roles",
            "terms": "key terms, conditions, and obligations",
            "dates": "all important dates, deadlines, and timeframes",
            "risks": "identify risks, issues, unusual clauses, and potential problems",
            "obligations": "key obligations, requirements, and responsibilities for each party",
            "termination": "termination conditions, penalties, and dispute resolution"
        }

        sections = {}
        for category, query in query_categories.items():
            sections[category] = self.rag_processor.retrieve_relevant_sections(query)

        # Create enhanced prompt with retrieved content
        prompt = self._create_enhanced_prompt(text, sections, entities)

        response = self.goi_model(prompt)

        try:
            json_data = json.loads(response)
            # Calculate confidence and risk scores
            json_data["confidence_score"] = self._calculate_confidence_score(sections)
            json_data["risk_score"] = self._calculate_risk_score(sections["risks"])
            json_data["relevant_sections"] = self._get_key_sections(sections)
            return json_data
        except json.JSONDecodeError:
            logger.error(f"Failed to parse JSON response: {response}")
            # Attempt to extract JSON from response
            json_match = re.search(r'({[\s\S]*})', response)
            if json_match:
                try:
                    json_data = json.loads(json_match.group(1))
                    return json_data
                except:
                    pass
            return {
                "cac_ben": [],
                "ngay_thang": [],
                "dieu_khoan": [],
                "nghia_vu": [],
                "phan_loai": "Error: Could not parse",
                "van_de": ["Error extracting information from document"],
                "confidence_score": 0.0
            }

    def _create_enhanced_prompt(self, text: str, sections: Dict[str, List[Dict]],
                                entities: Dict[str, List[str]]) -> str:
        # Format entities for prompt
        entity_section = ""
        for entity_type, items in entities.items():
            if items:
                entity_section += f"{entity_type.upper()}: {', '.join(items[:5])}\n"

        return f"""
        You are a specialized legal document analyzer. Analyze this document with precision.

        <document_metadata>
        {entity_section}
        </document_metadata>

        <key_sections>
        PARTIES INFORMATION:
        {self._format_sections(sections.get('parties', []))}

        KEY TERMS:
        {self._format_sections(sections.get('terms', []))}

        DATES AND TIMEFRAMES:
        {self._format_sections(sections.get('dates', []))}

        OBLIGATIONS:
        {self._format_sections(sections.get('obligations', []))}

        TERMINATION CONDITIONS:
        {self._format_sections(sections.get('termination', []))}

        RISK FACTORS:
        {self._format_sections(sections.get('risks', []))}
        </key_sections>

        Analyze this legal document with focus on:
        1. Identify all parties precisely with their roles and relationships
        2. Extract all critical dates (execution, effective, termination, renewal)
        3. Identify unusual, ambiguous, or non-standard terms
        4. Evaluate balanced vs one-sided obligations
        5. Classify document type with highest confidence
        6. Flag potential legal issues or risks

        Return a detailed JSON with:
        {{
            "cac_ben": [
                {{"name": "Full legal name", "role": "Role in agreement", "obligations": ["Key obligations"]}}
            ],
            "ngay_thang": [
                {{"date": "Date string", "event": "Type of date (effective, termination, etc.)"}}
            ],
            "dieu_khoan": [
                {{"clause": "Clause name/number", "summary": "Brief description", "concern_level": "high/medium/low"}}
            ],
            "nghia_vu": [
                {{"party": "Obligated party", "obligation": "What they must do", "consequence": "Result of non-compliance"}}
            ],
            "phan_loai": "Document classification with confidence (e.g. 'Employment Agreement (95% confidence)')",
            "van_de": [
                {{"issue": "Potential problem", "impact": "Why it matters", "recommendation": "Suggested action"}}
            ]
        }}

        Focus on accuracy and completeness, using exact text from the document where possible.
        """

    def _format_sections(self, sections: List[Dict]) -> str:
        formatted = []
        for i, section in enumerate(sections):
            # Get first 300 chars with ellipsis if needed
            text = section['content'][:300]
            if len(section['content']) > 300:
                text += "..."
            # Add section number and score
            formatted.append(f"[{i + 1}] (score: {section['score']:.2f}) {text}")

        return "\n\n".join(formatted)

    def _get_key_sections(self, sections: Dict[str, List[Dict]]) -> List[Dict[str, str]]:
        """Extract the most important sections across all categories"""
        all_sections = []
        for category, section_list in sections.items():
            for section in section_list[:2]:  # Take top 2 from each category
                all_sections.append({
                    "category": category,
                    "content": section["content"],
                    "score": section["score"]
                })

        # Sort by score and take top 10
        all_sections.sort(key=lambda x: x["score"], reverse=True)
        return all_sections[:10]

    def _calculate_confidence_score(self, sections: Dict[str, List[Dict]]) -> float:
        """Calculate overall confidence based on retrieval scores"""
        scores = []
        for category, section_list in sections.items():
            if section_list:
                # Average the scores for top 3 sections in each category
                category_score = sum(section["score"] for section in section_list[:3]) / min(3, len(section_list))
                scores.append(category_score)

        return sum(scores) / len(scores) if scores else 0.0

    def _calculate_risk_score(self, risk_sections: List[Dict]) -> Dict[str, Any]:
        """Calculate detailed risk scores by category"""
        risk_categories = {
            "liability": ["indemnify", "hold harmless", "unlimited liability", "disclaimer", "waiver"],
            "termination": ["terminate", "cancellation", "rescind", "break", "breach"],
            "dispute": ["arbitration", "jurisdiction", "governing law", "litigation", "court"],
            "compliance": ["comply", "regulation", "violation", "sanction", "requirement"],
            "confidentiality": ["confidential", "disclose", "proprietary", "trade secret", "nda"],
            "payment": ["payment", "invoice", "late fee", "interest", "penalty"]
        }

        category_scores = {}
        key_terms_found = {}

        for category, terms in risk_categories.items():
            score = 0
            terms_found = []

            for section in risk_sections:
                content = section["content"].lower()
                for term in terms:
                    if term in content:
                        count = content.count(term)
                        score += count
                        if count > 0 and term not in terms_found:
                            terms_found.append(term)

            # Normalize score between 0 and 1
            category_scores[category] = min(1.0, score / 10.0)
            key_terms_found[category] = terms_found

        # Calculate overall risk score (weighted average)
        weights = {
            "liability": 0.25,
            "termination": 0.2,
            "dispute": 0.15,
            "compliance": 0.15,
            "confidentiality": 0.15,
            "payment": 0.1
        }

        overall_score = sum(category_scores[cat] * weights[cat] for cat in category_scores)

        return {
            "overall_score": overall_score,
            "category_breakdown": category_scores,
            "key_terms_found": key_terms_found,
            "risk_level": "High" if overall_score > 0.7 else "Medium" if overall_score > 0.4 else "Low"
        }

    def analyze_temporal_elements(self, text: str) -> Dict[str, List[str]]:
        """Analyze time-sensitive elements in contracts"""
        periods = re.findall(r'(\d+)\s+(day|month|year|week)s?', text, re.IGNORECASE)
        time_frames = {
            "notice_periods": [],
            "cure_periods": [],
            "renewal_terms": [],
            "payment_terms": []
        }

        # Get context window for each time period
        for count, unit in periods:
            context_pattern = r'.{0,100}' + re.escape(f"{count} {unit}") + r'.{0,100}'
            context_matches = re.findall(context_pattern, text)

            for context in context_matches:
                context = context.lower()
                period = f"{count} {unit}(s)"

                if any(word in context for word in ["notice", "notify", "notification"]):
                    time_frames["notice_periods"].append(period)
                elif any(word in context for word in ["cure", "remedy", "correct", "fix"]):
                    time_frames["cure_periods"].append(period)
                elif any(word in context for word in ["renew", "extend", "extension", "continuation"]):
                    time_frames["renewal_terms"].append(period)
                elif any(word in context for word in ["pay", "payment", "invoice", "bill"]):
                    time_frames["payment_terms"].append(period)

        return time_frames

    def phan_tich_tai_lieu(self, duong_dan_file: str,
                           progress_callback: Optional[Callable[[float, str], None]] = None) -> ThongTinTrichXuat:
        """Process a legal document with optional progress reporting"""
        logger.info(f"Bắt đầu phân tích file: {duong_dan_file}")

        # Report progress
        if progress_callback:
            progress_callback(0.1, "Loading document")

        # Extract text from file
        noi_dung, dinh_dang = self.doc_file(duong_dan_file)

        if progress_callback:
            progress_callback(0.3, "Processing document content")

        # Additional analysis
        temporal_elements = self.analyze_temporal_elements(noi_dung)

        if progress_callback:
            progress_callback(0.5, "Retrieving relevant information")

        # Extract main information
        thong_tin = self.trich_xuat_thong_tin(noi_dung)

        if progress_callback:
            progress_callback(0.9, "Finalizing analysis")

        ten_file = Path(duong_dan_file).name

        # Construct result object
        return ThongTinTrichXuat(
            ten_file=ten_file,
            dinh_dang=dinh_dang,
            cac_ben_lien_quan=thong_tin.get("cac_ben", []),
            ngay_thang=thong_tin.get("ngay_thang", []),
            dieu_khoan=thong_tin.get("dieu_khoan", []),
            nghia_vu=thong_tin.get("nghia_vu", []),
            phan_loai=thong_tin.get("phan_loai", "Uncategorized"),
            van_de_tiem_tang=thong_tin.get("van_de", []),
            relevant_sections=thong_tin.get("relevant_sections", []),
            risk_score=thong_tin.get("risk_score", {"overall_score": 0.0}),
            confidence_score=thong_tin.get("confidence_score", 0.0)
        )

    def batch_process_documents(self, directory: str) -> Dict[str, ThongTinTrichXuat]:
        """Process all documents in a directory"""
        results = {}
        files = list(Path(directory).glob("**/*.pdf")) + list(Path(directory).glob("**/*.docx"))

        logger.info(f"Found {len(files)} documents to process")

        for i, file_path in enumerate(files):
            logger.info(f"Processing {i + 1}/{len(files)}: {file_path.name}")
            try:
                ket_qua = self.phan_tich_tai_lieu(
                    str(file_path),
                    progress_callback=lambda prog, msg: logger.info(
                        f"Progress {file_path.name}: {prog * 100:.0f}% - {msg}")
                )
                results[file_path.name] = ket_qua

                # Save individual result
                result_path = file_path.with_suffix('.json')
                with open(result_path, 'w', encoding='utf-8') as f:
                    json.dump(dataclasses.asdict(ket_qua), f, ensure_ascii=False, indent=2)

            except Exception as e:
                logger.error(f"Error processing {file_path.name}: {str(e)}")
                results[file_path.name] = ThongTinTrichXuat(
                    ten_file=file_path.name,
                    dinh_dang=file_path.suffix,
                    cac_ben_lien_quan=[],
                    ngay_thang=[],
                    dieu_khoan=[],
                    nghia_vu=[],
                    phan_loai="Error",
                    van_de_tiem_tang=[{"issue": f"Processing error: {str(e)}"}]
                )

        return results

    def extract_text_from_docx(self, param):
        pass


def main():
    try:
        # Use a specific model from Hugging Face Hub
        model_path = "deepseek-ai/deepseek-coder-6.7b-base"  # or another suitable model
        xu_ly = XuLyTaiLieuPhapLy(model_path=model_path)
        thu_muc_tai_lieu = "./documents/"
        os.makedirs(thu_muc_tai_lieu, exist_ok=True)

        logger.info(f"Model loaded successfully from {model_path}")
        logger.info(f"Scanning directory: {thu_muc_tai_lieu}")

        # Process all documents in batch mode
        results = xu_ly.batch_process_documents(thu_muc_tai_lieu)

        # Save summary report
        with open(os.path.join(thu_muc_tai_lieu, "analysis_summary.json"), 'w', encoding='utf-8') as f:
            summary = {filename: {
                "risk_score": result.risk_score["overall_score"] if result.risk_score else 0.0,
                "confidence": result.confidence_score,
                "document_type": result.phan_loai,
                "parties_count": len(result.cac_ben_lien_quan),
                "issues_count": len(result.van_de_tiem_tang)
            } for filename, result in results.items()}
            json.dump(summary, f, ensure_ascii=False, indent=2)

        logger.info(f"Analysis complete. Processed {len(results)} documents.")

    except Exception as e:
        logger.error(f"Critical error in main: {str(e)}")
        raise


if __name__ == "__main__":
    # Import needed in main
    import dataclasses

    main()