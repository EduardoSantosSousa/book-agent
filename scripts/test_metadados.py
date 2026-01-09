# Script para verificar o que há no bucket
from google.cloud import storage
import json

client = storage.Client()
bucket_name = "book-agent-embeddings-bucket"
prefix = "embeddings/"

print("📦 Verificando arquivos no bucket...")
blobs = list(client.list_blobs(bucket_name, prefix=prefix))

for blob in blobs[:20]:  # Mostrar primeiros 20 arquivos
    print(f"  📄 {blob.name} ({blob.size/1024/1024:.1f}MB)")