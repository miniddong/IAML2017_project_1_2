def validate_chunked_batch_X(chunked_batch_X, batch_size, chunk_size, image_height):
    assert len(chunked_batch_X) == batch_size
    assert chunked_batch_X[0].shape == (image_height, chunk_size)
