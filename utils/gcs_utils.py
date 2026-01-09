"""
Utilitários para acesso ao Google Cloud Storage.
"""
import os
import re
from datetime import datetime
from google.cloud import storage

class GCSHelper:
    """Helper para operações no GCS"""
    
    @staticmethod
    def get_storage_client():
        """Retorna cliente do GCS"""
        return storage.Client()
    
    @staticmethod
    def extract_timestamp_from_filename(filename: str) -> datetime:
        """Extrai timestamp do nome do arquivo"""
        try:
            # Procura padrão YYYYMMDD_HHMMSS
            match = re.search(r'(\d{8}_\d{6})', filename)
            if match:
                return datetime.strptime(match.group(1), '%Y%m%d_%H%M%S')
        except:
            pass
        return None