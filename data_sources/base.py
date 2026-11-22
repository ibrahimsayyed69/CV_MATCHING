"""
Abstract interface every CV source must implement.
 
When you get access to the real database, write ONE new class here
(e.g. SQLSource, SharePointSource, S3Source) that implements these
methods, then flip CV_SOURCE in config.py. Nothing else in the app changes.
"""
 
from abc import ABC, abstractmethod
from typing import List, Dict
 
 
class CVSource(ABC):
    @abstractmethod
    def list_cv_ids(self) -> List[str]:
        """Return a list of unique CV identifiers."""
        raise NotImplementedError
 
    @abstractmethod
    def get_cv_text(self, cv_id: str) -> str:
        """Return the raw extracted text of one CV. Can be slow (parsing/OCR)."""
        raise NotImplementedError
 
    @abstractmethod
    def get_cv_version(self, cv_id: str) -> str:
        """
        A CHEAP value that changes whenever the CV changes (file modified-time
        + size, a DB 'updated_at', an S3 ETag...). Must NOT read or parse the
        CV: the cache calls it for every CV on every search to decide what
        needs re-processing.
        """
        raise NotImplementedError
 
    @abstractmethod
    def get_cv_metadata(self, cv_id: str) -> Dict:
        """Cheap display info: {'name', 'category', 'source_ref'}. Must not parse the CV."""
        raise NotImplementedError