from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from huffman_engine import AdaptiveHuffman, calculate_metrics
from cache_manager import CompressionCache

# Initialize the server and the cache
app = FastAPI(title="Stage 2: Compression Microservice")
cache = CompressionCache()

# Define the expected JSON payload format
class CompressionRequest(BaseModel):
    text: str

@app.post("/compress")
async def compress_data(request: CompressionRequest):
    input_text = request.text
    
    if not input_text:
        raise HTTPException(status_code=400, detail="Input text cannot be empty")

    # 1. THE CACHE INTERCEPTOR
    cached_result = cache.get(input_text)
    if cached_result:
        return {"status": "success", "source": "cache", "data": cached_result}

    try:
        # 2. THE CORE ENGINE (Cache Miss)
        encoder = AdaptiveHuffman()
        compressed_binary = encoder.encode(input_text)
        
        # Verify Lossless Recovery
        decoder = AdaptiveHuffman()
        decompressed_text = decoder.decode(compressed_binary)
        
        # Get Graduate Metrics
        ratio, entropy, eff = calculate_metrics(input_text, compressed_binary)
        
        # Format the response
        response_data = {
            "lossless_verification": input_text == decompressed_text,
            "original_size_bits": len(input_text) * 8,
            "compressed_size_bits": len(compressed_binary),
            "metrics": {
                "compression_ratio": round(ratio, 2),
                "text_entropy": round(entropy, 2),
                "encoding_efficiency_percent": round(eff, 2)
            },
            "compressed_payload": compressed_binary,
            "recovered_text": decompressed_text
        }
        
        # 3. UPDATE CACHE AND RETURN
        cache.set(input_text, response_data)
        return {"status": "success", "source": "computed", "data": response_data}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Engine failed: {str(e)}")
