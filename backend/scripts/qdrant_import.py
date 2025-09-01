"""Import Qdrant collection points from NDJSON for backup restoration or transfer.

Usage: python qdrant_import.py --input points.ndjson
"""
import json
import argparse
from qdrant_client import QdrantClient, models as qdrant_models
from typing import List

QDRANT_PATH = "./qdrant_storage"
COLLECTION = "documents"


def import_points(input_path: str, batch_size: int = 100):
    """Import points from NDJSON file to Qdrant collection."""
    client = QdrantClient(path=QDRANT_PATH)
    
    # Ensure collection exists
    collections = [c.name for c in client.get_collections().collections]
    if COLLECTION not in collections:
        print(f"Creating collection '{COLLECTION}'...")
        client.create_collection(
            collection_name=COLLECTION,
            vectors_config=qdrant_models.VectorParams(
                size=384, 
                distance=qdrant_models.Distance.COSINE
            )
        )
    
    batch = []
    total_imported = 0
    
    try:
        with open(input_path, "r", encoding="utf-8") as in_f:
            for line_num, line in enumerate(in_f, 1):
                try:
                    # Parse JSON line
                    point_data = json.loads(line.strip())
                    
                    # Handle both scroll API format and direct point format
                    if isinstance(point_data, list) and len(point_data) == 2:
                        # Scroll API returns [points, next_page_offset]
                        points_list, _ = point_data
                    elif isinstance(point_data, dict) and 'id' in point_data:
                        # Direct point format
                        points_list = [point_data]
                    else:
                        print(f"Skipping malformed line {line_num}: unexpected format")
                        continue
                    
                    # Convert to PointStruct
                    for point in points_list:
                        if not isinstance(point, dict) or 'id' not in point:
                            continue
                            
                        batch.append(qdrant_models.PointStruct(
                            id=point['id'],
                            vector=point.get('vector', []),
                            payload=point.get('payload', {})
                        ))
                        
                        # Batch upsert
                        if len(batch) >= batch_size:
                            client.upsert(collection_name=COLLECTION, points=batch)
                            total_imported += len(batch)
                            print(f"Imported {total_imported} points...")
                            batch = []
                            
                except json.JSONDecodeError as e:
                    print(f"Skipping malformed JSON on line {line_num}: {e}")
                    continue
                except Exception as e:
                    print(f"Error processing line {line_num}: {e}")
                    continue
            
            # Final batch
            if batch:
                client.upsert(collection_name=COLLECTION, points=batch)
                total_imported += len(batch)
                
        print(f"Successfully imported {total_imported} total points from {input_path}")
        
    except FileNotFoundError:
        print(f"Error: File '{input_path}' not found")
    except Exception as e:
        print(f"Import failed: {e}")
    finally:
        client.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Import Qdrant points from NDJSON")
    parser.add_argument("--input", "-i", required=True, help="Input NDJSON file path")
    parser.add_argument("--batch-size", "-b", type=int, default=100, help="Batch size for upserts")
    args = parser.parse_args()
    
    import_points(args.input, args.batch_size)
