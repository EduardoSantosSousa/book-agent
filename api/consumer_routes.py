# D:\Django\book_agent\api\consumer_routes.py

from flask import Blueprint, jsonify
import logging
from config import config
from services.gcs_consumer_service import GCSEmbeddingConsumer
from datetime import datetime

logger = logging.getLogger(__name__)
consumer_bp = Blueprint('consumer', __name__, url_prefix='/consumer')

@consumer_bp.route('/status', methods=['GET'])
def get_consumer_status():
    """Status do consumidor GCS"""
    try:
        consumer = GCSEmbeddingConsumer(
            bucket_name=config.GCS_BUCKET_NAME,
            embeddings_prefix=config.GCS_EMBEDDINGS_PREFIX
        )
        
        # Tentar carregar para obter status atual
        loaded = consumer.load_latest_embeddings()
        stats = consumer.get_stats()
        
        return jsonify({
            'success': True,
            'data': {
                'status': 'operational' if loaded else 'failed',
                'bucket': config.GCS_BUCKET_NAME,
                'prefix': config.GCS_EMBEDDINGS_PREFIX,
                'environment': config.ENVIRONMENT,
                'version': stats.get('version'),
                'embeddings_loaded': stats.get('status') == 'loaded',
                'embeddings_shape': stats.get('embeddings', {}).get('shape'),
                'index_size': stats.get('index', {}).get('size'),
                'loaded_at': stats.get('loaded_at'),
                'mode': 'gcs_consumer'
            },
            'timestamp': datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"Erro ao obter status: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'mode': 'gcs_consumer'
        }), 500

@consumer_bp.route('/reload', methods=['POST'])
def reload_embeddings():
    """Recarrega embeddings mais recentes"""
    try:
        consumer = GCSEmbeddingConsumer(
            bucket_name=config.GCS_BUCKET_NAME,
            embeddings_prefix=config.GCS_EMBEDDINGS_PREFIX
        )
        
        success = consumer.load_latest_embeddings()
        
        if success:
            stats = consumer.get_stats()
            return jsonify({
                'success': True,
                'message': f"Recarregado para versão {stats.get('version')}",
                'data': stats
            }), 200
        else:
            return jsonify({
                'success': False,
                'error': 'Falha ao recarregar embeddings'
            }), 500
            
    except Exception as e:
        logger.error(f"Erro ao recarregar: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@consumer_bp.route('/check-update', methods=['GET'])
def check_for_update():
    """Verifica se há versão mais nova"""
    try:
        consumer = GCSEmbeddingConsumer(
            bucket_name=config.GCS_BUCKET_NAME,
            embeddings_prefix=config.GCS_EMBEDDINGS_PREFIX
        )
        
        # Carrega atual primeiro
        consumer.load_latest_embeddings()
        
        # Verifica se há nova versão
        has_update = consumer.check_for_new_version()
        
        stats = consumer.get_stats()
        
        return jsonify({
            'success': True,
            'data': {
                'has_update': has_update,
                'current_version': stats.get('version'),
                'current_loaded_at': stats.get('loaded_at')
            }
        }), 200
        
    except Exception as e:
        logger.error(f"Erro ao verificar atualização: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500