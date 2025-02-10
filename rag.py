import spacy
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
import torch
from typing import List, Dict, Any, Union
import pandas as pd
from dataclasses import dataclass
import re
import os
import PyPDF2
from docx import Document
from pathlib import Path
import pytesseract
from pdf2image import convert_from_path
import logging
import json 
from threading import Thread
from queue import Queue

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ThongTinTrichXuat:
    ten_file: str
    dinh_dang: str
    cac_ben_lien_quan: List[str]
    ngay_thang: List[str]
    dieu_khoan: List[str]
    nghia_vu: List[str]
    phan_loai: str
    van_de_tiem_tang: List[str]

class XuLyTaiLieuPhapLy:
    def __init__(self, model_path: str = "./models"):
        logger.info("Đang khởi tạo hệ thống xử lý tài liệu...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            local_files_only=True,
            pad_token_id=self.tokenizer.eos_token_id
        )
        self.streamer = TextIteratorStreamer(self.tokenizer)
        logger.info("Model initialized successfully")

    def doc_file(self, duong_dan_file: str) -> tuple[str, str]:
        duong_dan = Path(duong_dan_file)
        dinh_dang = duong_dan.suffix.lower()
        noi_dung = ""
        try:
            if dinh_dang == '.pdf':
                noi_dung = self.doc_pdf(duong_dan)
            elif dinh_dang == '.docx':
                noi_dung = self.doc_docx(duong_dan)
            else:
                raise ValueError(f"Định dạng file không được hỗ trợ: {dinh_dang}")
            return noi_dung, dinh_dang
        except Exception as e:
            logger.error(f"Lỗi khi đọc file {duong_dan}: {str(e)}")
            raise

    def doc_pdf(self, duong_dan: Path) -> str:
        noi_dung = ""
        try:
            with open(duong_dan, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for trang in pdf_reader.pages:
                    noi_dung += trang.extract_text() + "\n"
            if not noi_dung.strip():
                logger.info("Đang sử dụng OCR để xử lý PDF...")
                images = convert_from_path(duong_dan)
                for image in images:
                    noi_dung += pytesseract.image_to_string(image) + "\n"
        except Exception as e:
            logger.error(f"Lỗi khi xử lý PDF: {str(e)}")
            raise
        return noi_dung

    def doc_docx(self, duong_dan: Path) -> str:
        try:
            doc = Document(duong_dan)
            return "\n".join([paragraph.text for paragraph in doc.paragraphs])
        except Exception as e:
            logger.error(f"Lỗi khi đọc file DOCX: {str(e)}")
            raise

    def goi_model(self, prompt: str, max_tokens: int = 512) -> str:
        try:
            logger.info(f"Gửi prompt tới Deepseek ({len(prompt)} ký tự)...")

            encoded = self.tokenizer.encode_plus(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048  
            )
            
            input_ids = encoded["input_ids"].to(self.model.device)
            attention_mask = encoded["attention_mask"].to(self.model.device)

            generation_kwargs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "max_new_tokens": max_tokens,
                "do_sample": True,
                "temperature": 0.7,
                "pad_token_id": self.tokenizer.eos_token_id,
                "streamer": self.streamer
            }

            thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
            thread.start()

            response_text = ""
            for text in self.streamer:
                response_text += text
                logger.info(f"Generated text chunk: {text}")

            if not response_text.strip():
                logger.error("Deepseek không trả về dữ liệu.")
                return ""

            logger.info(f"Phản hồi từ Deepseek ({len(response_text)} ký tự): {response_text[:200]}...")
            return response_text.strip()

        except Exception as e:
            logger.error(f"Lỗi khi gọi Deepseek: {str(e)}")
            return ""

    def trich_xuat_thong_tin(self, text: str) -> Dict[str, Any]:
        max_length = 1000

        prompt = f"""
            Analyze the following legal document and return the results in valid JSON format.

            CONTENT:
            {text[:max_length]}

            Required JSON format:
            {{
                "cac_ben": ["Party A", "Party B"],
                "ngay_thang": ["01/01/2024"],
                "dieu_khoan": ["Clause 1", "Clause 2"],
                "nghia_vu": ["Party A's obligations"],
                "phan_loai": "Lease Agreement",
                "van_de": ["Controversial clauses"]
            }}

            Ensure the result is valid JSON without any additional text.
            """

        response = self.goi_model(prompt)

        if not response:
            logger.error("No valid response from Deepseek.")
            return {}

        try:
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                logger.error("No JSON found in response")
                return {}
        except json.JSONDecodeError:
            logger.error(f"JSONDecodeError: Invalid response: {response}")
            return {}

    def phan_tich_tai_lieu(self, duong_dan_file: str) -> ThongTinTrichXuat:
        logger.info(f"Bắt đầu phân tích file: {duong_dan_file}")
        noi_dung, dinh_dang = self.doc_file(duong_dan_file)
        thong_tin = self.trich_xuat_thong_tin(noi_dung)
        ten_file = Path(duong_dan_file).name
        return ThongTinTrichXuat(
            ten_file=ten_file,
            dinh_dang=dinh_dang,
            cac_ben_lien_quan=thong_tin.get("cac_ben", []),
            ngay_thang=thong_tin.get("ngay_thang", []),
            dieu_khoan=thong_tin.get("dieu_khoan", []),
            nghia_vu=thong_tin.get("nghia_vu", []),
            phan_loai=thong_tin.get("phan_loai", "Uncategorized"),
            van_de_tiem_tang=thong_tin.get("van_de", [])
        )


def main():
    try:
        model_path = "./models"  
        xu_ly = XuLyTaiLieuPhapLy(model_path=model_path)
        thu_muc_tai_lieu = "./documents/"
        os.makedirs(thu_muc_tai_lieu, exist_ok=True)
        
        logger.info(f"Model loaded successfully from {model_path}")
        logger.info(f"Scanning directory: {thu_muc_tai_lieu}")
        
        for ten_file in os.listdir(thu_muc_tai_lieu):
            if ten_file.endswith(('.pdf', '.docx')):
                duong_dan = os.path.join(thu_muc_tai_lieu, ten_file)
                logger.info(f"Processing file: {ten_file}")
                ket_qua = xu_ly.phan_tich_tai_lieu(duong_dan)
                logger.info(f"Results: {ket_qua}")
                
    except Exception as e:
        logger.error(f"Critical error in main: {str(e)}")
        raise

if __name__ == "__main__":
    main()