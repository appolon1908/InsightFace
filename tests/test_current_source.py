"""Exercise current numeric/API behavior with explicit synthetic constructors."""
import pytest
from conftest import main, detector, constructors


def test_only_synthetic_model_constructors_were_used():
    assert constructors == [('detector', '/synthetic-fixture/no-real-detector-model'),
                            ('recognizer', '/synthetic-fixture/no-real-recognizer-model')]


def test_health_and_model_independent_capabilities(client):
    assert client.get('/healthz').json() == {'ok': True}
    assert client.get('/v1/runtime/capabilities').json()['gpu'] is False
    assert client.get('/v1/model/status').json()['metric'] == 'cosine'


def test_missing_real_model_files_are_not_reported_ready(client):
    assert client.get('/readyz').json()['ok'] is False


@pytest.mark.parametrize('value', ['not-valid-base64', '', 'YWJj'])
def test_invalid_encoded_images_fail_truthfully(client, value):
    assert client.post('/v1/embeddings', json={'image_base64': value}).status_code == 400


def test_embedding_normalizes_features_and_selects_largest_face(client, image):
    response = client.post('/v1/embeddings', json={'image_base64': image})
    assert response.status_code == 200
    result = response.json()
    assert result['embedding'] == pytest.approx([0.6, 0.8])
    assert result['dimension'] == 2
    assert result['bbox'] == [5, 6, 8, 9]
    assert detector.input_size == (16, 12)


def test_no_detected_face_is_not_fabricated_success(client, image):
    detector.faces = None
    assert client.post('/v1/embeddings', json={'image_base64': image}).status_code == 422


@pytest.mark.parametrize('a,b,want', [([1, 0], [1, 0], 1), ([1, 0], [-1, 0], -1), ([1, 0], [0, 1], 0), ([0, 0], [1, 0], -1)])
def test_actual_cosine_numeric_semantics(a, b, want):
    assert main.cosine(a, b) == pytest.approx(want)


def test_embedding_dimension_mismatch_is_client_error(client):
    response = client.post('/v1/verify', json={'embedding_a': [1, 0], 'embedding_b': [1]})
    assert response.status_code == 400


def test_verify_matches_provided_vectors_without_inference(client):
    response = client.post('/v1/verify', json={'embedding_a': [1, 0], 'embedding_b': [1, 0], 'threshold': 0.9})
    assert response.json()['match'] is True


def test_missing_verify_input_is_not_fabricated(client):
    assert client.post('/v1/verify', json={}).status_code == 400


def test_gallery_search_sorts_scores_and_limits_results(client):
    response = client.post('/v1/search', json={'embedding': [1, 0], 'top_k': 1, 'gallery': [
        {'subject_id': 'opposite', 'embedding': [-1, 0]}, {'subject_id': 'same', 'embedding': [1, 0]}]})
    assert response.json()['hits'] == [{'subject_id': 'same', 'score': 1.0, 'match': True}]


def test_gallery_search_caps_returned_results(client):
    gallery = [{'subject_id': str(i), 'embedding': [1, 0]} for i in range(110)]
    assert len(client.post('/v1/search', json={'embedding': [1, 0], 'top_k': 999, 'gallery': gallery}).json()['hits']) == 100


def test_batch_search_preserves_independent_results(client):
    request = {'embedding': [1, 0], 'gallery': [{'subject_id': 'one', 'embedding': [1, 0]}]}
    result = client.post('/v1/batch/search', json=[request, request]).json()
    assert len(result) == 2
    assert result[0]['hits'][0]['subject_id'] == 'one'


def test_metrics_and_openapi_are_generated_from_actual_source(client):
    assert 'recognition_requests_total' in client.get('/metrics').text
    assert '/v1/batch/search' in client.get('/openapi.json').json()['paths']
