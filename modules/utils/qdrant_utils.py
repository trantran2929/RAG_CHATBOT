from rag_note.modules.utils.services import qdrant_services, embedder_services
from qdrant_client import models
import uuid

def add_doc(collection_name, dense_vector, sparse_vector, payload, point_id=None):
    point_id = point_id or str(uuid.uuid4())
    vectors = {
        "dense_vector": dense_vector,
        "binary": dense_vector,
    }
    if sparse_vector is not None:
        vectors["sparse_vector"] = {"sparse_vector": sparse_vector}
    qdrant_services.client.upsert(
        collection_name=collection_name,
        points=[
            models.PointStruct(
                id=point_id,
                vector=vectors,
                payload=payload
            )
        ]
    )
    return point_id

def update_doc(collection_name, point_id, payload_update):
    qdrant_services.client.set_payload(
        collection_name=collection_name,
        payload=payload_update,
        points=[point_id]
    )

def delete_doc(collection_name, point_id):
    qdrant_services.client.delete(
        collection_name=collection_name,
        points_selector=models.PointIdsList(points=[point_id])
    )

def delete_payload_key(collection_name, point_id, keys):
    qdrant_services.client.delete_payload(
        collection_name=collection_name,
        keys=keys,
        points=[point_id]
    )

def get_doc(collection_name, point_id):
    result = qdrant_services.client.retrieve(
        collection_name=collection_name,
        ids=[point_id],
        with_payload=True
    )
    return result[0].payload if result else None

def search_hybrid(collection_name,query, top_k=5):
    dense_vector = embedder_services.encode_dense(query)[0]
    sparse_vector = embedder_services.encode_sparse(query)
    binary_vector = embedder_services.encode_binary(dense_vector)
    prefetch_list = [
        models.Prefetch(
            query=dense_vector,
            using="dense_vector"
        ),
        models.Prefetch(
            query=sparse_vector,
            using="sparse_vector"   
        ),
        models.Prefetch(
            query=binary_vector,
            using="binary"
        )
    ]

    result= qdrant_services.client.query_points(
        collection_name=collection_name,
        prefetch=prefetch_list,
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=top_k,
        with_payload=True,
    )

    return [{"id": r.id, "score": r.score, "payload": r.payload} for r in result.points]
