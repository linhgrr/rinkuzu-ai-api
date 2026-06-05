from api.routers.pipeline import ProcessDocumentRequest


def test_process_request_has_source_s3_key_field():
    assert "source_s3_key" in ProcessDocumentRequest.model_fields
    field = ProcessDocumentRequest.model_fields["source_s3_key"]
    assert field.default is None
