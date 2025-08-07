import sys
from pathlib import Path
import fitz
import uuid
from datetime import datetime, timezone

from logger.custom_logger import CustomLogger
from exception.custom_exception import DocumentPortalException

class DocumentIngestion:

    def __init__(self,base_dir="data\\document_compare",session_id=None):
        self.log=CustomLogger().get_logger(__name__)
        self.base_dir=Path(base_dir)
        self.base_dir.mkdir(parents=True,exist_ok=True)
        self.session_id = session_id or f"session_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        self.session_path = self.base_dir / self.session_id
        self.session_path.mkdir(parents=True, exist_ok=True)
    
    def delete_existing_file(self):
        try:
            if self.base_dir.exists() and self.base_dir.is_dir():
                for file in self.base_dir.iterdir():
                    if file.is_file():
                        file.unlink()#for deleting the file
                        self.log.info("File deleted",path=str(file))
                self.log.info("Directory Cleaned",directory=str(self.base_dir))
        except Exception as e:
            self.log.error(f"Error deleting the existing files:{e}")
            raise DocumentPortalException("An error occured while deleting existing documents:", sys)


    def save_uploaded_files(self,ref_file,actual_file):
        try:
            self.delete_existing_file()
            self.log.info("Existing file deleted successfully")
            ref_path=self.base_dir/ref_file.name
            actual_path=self.base_dir/actual_file.name

            #ref_file: which means it's a V1(changes made file)
            #actual_file: which means V2
            print("ref_file.name", ref_file.name)
            print("actual_file.name", actual_file.name)

            if not ref_file.name.endswith(".pdf") or not actual_file.name.endswith(".pdf"):
                raise ValueError("Only PDF files allowed.")
            with open(ref_path, "wb") as f:
                f.write(ref_file.getbuffer())

            with open(actual_path, "wb") as f:
                f.write(actual_file.getbuffer())

            self.log.info("Files saved", reference=str(ref_path), actual=str(actual_path), session=self.session_id)
            return ref_path, actual_file
        except Exception as e:
            self.log.error(f"Error saved uploading files:{e}")
            raise DocumentPortalException("An error occured while saving and loading the documents:", sys)


    def read_pdf(self,pdf_path):
        try:
            with fitz.open(pdf_path) as doc:
                if doc.is_encrypted:
                    raise ValueError("PDF is encrypted: {pdf_path.name}")
                all_text=[]
                for page_num in range(doc.page_count):
                    page=doc.load_page(page_num)
                    text=page.get_text()
                    if text.strip():
                        all_text.append(f"\n --- Page {page_num+1} --- \n{text}")
                self.log.info("PDF read successfully",file=str(pdf_path), pages=len(all_text))
                return "\n".join(all_text)

        except Exception as e:
            self.log.error(f"Error while reading the PDF:{e}")
            raise DocumentPortalException("An error occured while reading the PDF:",sys)
        
    def combine_documents(self) -> str:
        """
        Combine content of all PDFs in session folder into a single string.
        """
        try:
            doc_parts = []
            for file in sorted(self.session_path.iterdir()):
                if file.is_file() and file.suffix.lower() == ".pdf":
                    content = self.read_pdf(file)
                    doc_parts.append(f"Document: {file.name}\n{content}")

            combined_text = "\n\n".join(doc_parts)
            self.log.info("Documents combined", count=len(doc_parts), session=self.session_id)
            return combined_text

        except Exception as e:
            self.log.error("Error combining documents", error=str(e), session=self.session_id)
            raise DocumentPortalException("Error combining documents", sys)

    def clean_old_sessions(self, keep_latest: int = 3):
        """
        Optional method to delete older session folders, keeping only the latest N.
        """
        try:
            session_folders = sorted(
                [f for f in self.base_dir.iterdir() if f.is_dir()],
                reverse=True
            )
            for folder in session_folders[keep_latest:]:
                for file in folder.iterdir():
                    file.unlink()
                folder.rmdir()
                self.log.info("Old session folder deleted", path=str(folder))

        except Exception as e:
            self.log.error("Error cleaning old sessions", error=str(e))
            raise DocumentPortalException("Error cleaning old sessions", sys)
        
